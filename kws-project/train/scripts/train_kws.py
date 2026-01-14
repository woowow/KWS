import os, csv, random
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

SR = 16000
DUR = 1.0
NSAMP = int(SR * DUR)

LABELS = ["next","prev","stop","play","unknown","silence"]
LABEL2ID = {k:i for i,k in enumerate(LABELS)}

@dataclass
class Split:
    train_spk: list[str]
    val_spk: list[str]
    test_spk: list[str]

def load_manifest(path="data/manifest.csv"):
    items = []
    speakers = set()
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            items.append((r["path"], r["speaker"], r["label"]))
            speakers.add(r["speaker"])
    return items, sorted(list(speakers))

def make_split(speakers):
    random.shuffle(speakers)
    n = len(speakers)
    n_train = max(1, int(n*0.7))
    n_val = max(1, int(n*0.15))
    train = speakers[:n_train]
    val = speakers[n_train:n_train+n_val]
    test = speakers[n_train+n_val:] or speakers[-1:]
    return Split(train, val, test)

class KWSDataset(Dataset):
    def __init__(self, items, speakers, augment=False):
        self.items = [(p, LABEL2ID[lbl]) for (p, spk, lbl) in items if spk in set(speakers)]
        self.augment = augment

    def __len__(self): return len(self.items)

    def _fix_len(self, wav):
        # wav: (1, n)
        n = wav.shape[1]
        if n >= NSAMP:
            start = random.randint(0, n-NSAMP)
            return wav[:, start:start+NSAMP]
        else:
            pad = NSAMP - n
            return torch.nn.functional.pad(wav, (0, pad))

    def __getitem__(self, idx):
        path, y = self.items[idx]
        wav, sr = torchaudio.load(path)  # (C, n)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = self._fix_len(wav)  # (1, 16000)

        # normalize
        wav = wav - wav.mean()
        wav = wav / (wav.std() + 1e-6)

        # simple aug: gain + small noise
        if self.augment:
            gain = random.uniform(0.7, 1.3)
            wav = wav * gain
            noise = torch.randn_like(wav) * random.uniform(0.0, 0.02)
            wav = wav + noise

        x = wav.squeeze(0)  # (16000,)
        return x, torch.tensor(y, dtype=torch.long)

class RawKWSNet(nn.Module):
    def __init__(self, n_class=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, n_class)

    def forward(self, x):
        # x: (B, 16000)
        x = x.unsqueeze(1)           # (B, 1, 16000)
        z = self.net(x).squeeze(-1)  # (B, 64)
        return self.fc(z)

def run_epoch(model, loader, opt=None, device="cuda"):
    is_train = opt is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if is_train:
            opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        if is_train:
            loss.backward()
            opt.step()

        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum/total, correct/total

def main():
    random.seed(0)
    torch.manual_seed(0)

    items, speakers = load_manifest()
    if len(items) == 0:
        raise RuntimeError("manifest is empty. Record first, then run build_manifest.py")

    split = make_split(speakers)
    print("[Split]")
    print(" train:", split.train_spk)
    print(" val  :", split.val_spk)
    print(" test :", split.test_spk)

    train_ds = KWSDataset(items, split.train_spk, augment=True)
    val_ds   = KWSDataset(items, split.val_spk, augment=False)
    test_ds  = KWSDataset(items, split.test_spk, augment=False)

    train_ld = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
    val_ld   = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)
    test_ld  = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RawKWSNet(n_class=len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints", exist_ok=True)
    best_val = 0.0

    for epoch in range(1, 21):
        tr_loss, tr_acc = run_epoch(model, train_ld, opt=opt, device=device)
        va_loss, va_acc = run_epoch(model, val_ld, opt=None, device=device)
        print(f"Epoch {epoch:02d} | train acc={tr_acc:.3f} | val acc={va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), "checkpoints/best.pt")

    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
    te_loss, te_acc = run_epoch(model, test_ld, opt=None, device=device)
    print(f"[TEST] acc={te_acc:.3f}")

if __name__ == "__main__":
    main()
