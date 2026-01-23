import os, csv, random, json
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import soundfile as sf
import numpy as np
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
SR = 16000
DUR = 1.5
NSAMP = int(SR * DUR)

#LABELS = ["wake", "quit", "next", "prev", "stop", "play", "unknown", "silence"]
LABELS = ["next", "prev", "stop", "play", "unknown", "silence"]
LABEL2ID = {k: i for i, k in enumerate(LABELS)}

@dataclass
class Split:
    train_spk: List[str]
    val_spk: List[str]
    test_spk: List[str]

def load_manifest(path="data/manifest.csv"):
    items = []
    speakers = set()
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            items.append((r["path"], r["speaker"], r["label"]))
            speakers.add(r["speaker"])
    return items, sorted(list(speakers))

def make_split(speakers: List[str], seed: int = 0) -> Split:
    rng = random.Random(seed)
    spk = speakers[:]
    rng.shuffle(spk)
    n = len(spk)
    n_train = max(1, int(n * 0.7))
    n_val = max(1, int(n * 0.15))
    train = spk[:n_train]
    val = spk[n_train:n_train + n_val]
    test = spk[n_train + n_val:] or spk[-1:]
    return Split(train, val, test)

class RawKWSDataset(Dataset):
    def __init__(self, items, speakers, augment=False):
        spk_set = set(speakers)
        self.items = [(p, LABEL2ID[lbl]) for (p, spk, lbl) in items
                      if (spk in spk_set and lbl in LABEL2ID)]
        self.augment = augment

    def __len__(self): 
        return len(self.items)

    def _fix_len(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (1, n)
        n = wav.shape[1]
        if n >= NSAMP:
            start = random.randint(0, n - NSAMP)
            wav = wav[:, start:start + NSAMP]
        else:
            wav = F.pad(wav, (0, NSAMP - n))
        return wav

    def __getitem__(self, idx):
        path, y = self.items[idx]

        audio, sr = sf.read(path, dtype="float32")  # (n,) or (n, ch)

        # stereo -> mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # 이 프로젝트는 16k로 녹음했다고 가정(아니면 데이터부터 정리)
        if sr != SR:
            raise RuntimeError(f"Sample rate mismatch: expected {SR}, got {sr} for {path}")

        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, n)

        wav = self._fix_len(wav)  # (1, NSAMP)

        # per-utterance normalize
        wav = wav - wav.mean()
        wav = wav / (wav.std() + 1e-6)

        if self.augment:
            gain = random.uniform(0.7, 1.3)
            wav = wav * gain

            shift = random.randint(-int(0.1 * SR), int(0.1 * SR))
            wav = torch.roll(wav, shifts=shift, dims=1)

            wav = wav + torch.randn_like(wav) * random.uniform(0.0, 0.02)

        x = wav.squeeze(0)  # (NSAMP,)
        return x, torch.tensor(y, dtype=torch.long)


class RawKWSNet(nn.Module):
    """
    브라우저/실시간까지 염두: 너무 무겁지 않게, BN+Dropout 추가
    입력: (B, NSAMP)
    """
    def __init__(self, n_class=len(LABELS)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, n_class),
        )

    def forward(self, x):
        x = x.unsqueeze(1)      # (B,1,NSAMP)
        z = self.conv(x)        # (B,128,T')
        z = self.pool(z)        # (B,128,1)
        return self.fc(z)       # (B,n_class)

def compute_class_weights(items: List[Tuple[str, str, str]]) -> torch.Tensor:
    counts = torch.zeros(len(LABELS), dtype=torch.float)
    for _, _, lbl in items:
        if lbl in LABEL2ID:
            counts[LABEL2ID[lbl]] += 1
    w = 1.0 / (counts + 1e-6)
    w = w / w.mean()
    return w

def run_epoch(model, loader, opt=None, device="cuda", class_weights=None):
    is_train = opt is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss(weight=class_weights).to(device)

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

    return loss_sum / total, correct / total

def main():
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    items, speakers = load_manifest()
    if len(items) == 0:
        raise RuntimeError("manifest is empty. Record first, then run build_manifest.py")

    #split = make_split(speakers, seed=seed)
    split = Split(train_spk=["spk02"], val_spk=["spk02"], test_spk=["spk04"])

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/split_raw.json", "w", encoding="utf-8") as f:
        json.dump(split.__dict__, f, ensure_ascii=False, indent=2)

    print("[Split]")
    print(" train:", split.train_spk)
    print(" val  :", split.val_spk)
    print(" test :", split.test_spk)

    train_ds = RawKWSDataset(items, split.train_spk, augment=False)
    val_ds   = RawKWSDataset(items, split.val_spk, augment=False)
    test_ds  = RawKWSDataset(items, split.test_spk, augment=False)

    class_w = compute_class_weights(items)
    y_train = [y for _, y in train_ds.items]
    class_count = torch.bincount(torch.tensor(y_train), minlength=len(LABELS)).float()
    per_class_weight = 1.0 / (class_count + 1e-6)
    sample_weights = torch.tensor([per_class_weight[y] for y in y_train], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RawKWSNet(n_class=len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    class_w = class_w.to(device)

    best_val = 0.0
    patience = 5
    bad = 0

    for epoch in range(1, 51):
        tr_loss, tr_acc = run_epoch(model, train_ld, opt=opt, device=device, class_weights=class_w)
        va_loss, va_acc = run_epoch(model, val_ld, opt=None, device=device, class_weights=class_w)
        print(f"Epoch {epoch:02d} | train acc={tr_acc:.3f} | val acc={va_acc:.3f} | tr_loss={tr_loss:.3f} | va_loss={va_loss:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            bad = 0
            torch.save(model.state_dict(), "checkpoints/best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStopping] no improvement for {patience} epochs.")
                break

    model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
    te_loss, te_acc = run_epoch(model, test_ld, opt=None, device=device, class_weights=class_w)
    print(f"[TEST] acc={te_acc:.3f}")

if __name__ == "__main__":
    main()
