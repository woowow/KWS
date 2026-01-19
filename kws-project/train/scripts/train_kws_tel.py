# kws-project/train/scripts/train_kws_mel.py
import os, csv, json, random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
SR = 16000
DUR = 1.5                  # 너가 추천한 1.2~1.5s로 맞추는 걸 권장 (데이터/웹 윈도우와 일치)
NSAMP = int(SR * DUR)

# log-mel params (KWS에서 흔히 쓰는 값들)
N_FFT = 400                # 25ms @ 16k
HOP = 160                  # 10ms @ 16k
WIN = 400
N_MELS = 40
F_MIN = 20
F_MAX = 7600

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

class LogMelKWSDataset(Dataset):
    """
    wav -> (1, NSAMP) 고정 -> log-mel (1, n_mels, n_frames)
    """
    def __init__(self, items, speakers, augment=False, device="cpu"):
        spk_set = set(speakers)
        self.items = [(p, LABEL2ID[lbl]) for (p, spk, lbl) in items if spk in spk_set]
        self.augment = augment

        # torchaudio transforms (CPU에서 하는 게 일반적)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_fft=N_FFT,
            win_length=WIN,
            hop_length=HOP,
            f_min=F_MIN,
            f_max=F_MAX,
            n_mels=N_MELS,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.device = device

    def __len__(self):
        return len(self.items)

    def _fix_len(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (1, n)
        n = wav.shape[1]
        if n >= NSAMP:
            start = random.randint(0, n - NSAMP)
            wav = wav[:, start:start + NSAMP]
        else:
            pad = NSAMP - n
            wav = F.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):
        path, y = self.items[idx]
        wav, sr = torchaudio.load(path)  # (C, n)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        wav = self._fix_len(wav)  # (1, NSAMP)

        # normalize (per-utterance)
        wav = wav - wav.mean()
        wav = wav / (wav.std() + 1e-6)

        # waveform aug (간단하지만 효과 있음)
        if self.augment:
            gain = random.uniform(0.7, 1.3)
            wav = wav * gain
            # time shift (±100ms 정도)
            shift = random.randint(-int(0.1 * SR), int(0.1 * SR))
            wav = torch.roll(wav, shifts=shift, dims=1)
            # small noise
            wav = wav + torch.randn_like(wav) * random.uniform(0.0, 0.02)

        # mel: (1, n_mels, n_frames)
        mel = self.mel(wav)
        logmel = torch.log(mel + 1e-6)

        # log-mel normalize (권장: per-sample)
        logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)

        return logmel, torch.tensor(y, dtype=torch.long)

class KWS2DCNN(nn.Module):
    """
    입력: (B, 1, n_mels, n_frames)
    작은 2D CNN + global pooling
    """
    def __init__(self, n_class=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (mels/2, frames/2)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, n_class)
        )

    def forward(self, x):
        z = self.features(x)
        return self.classifier(z)

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

def compute_class_weights(items: List[Tuple[str, str, str]]):
    # items: (path, speaker, label)
    counts = torch.zeros(len(LABELS), dtype=torch.float)
    for _, _, lbl in items:
        if lbl in LABEL2ID:
            counts[LABEL2ID[lbl]] += 1
    # inverse freq (간단 가중치)
    w = 1.0 / (counts + 1e-6)
    w = w / w.mean()
    return w

def main():
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    items, speakers = load_manifest()
    if len(items) == 0:
        raise RuntimeError("manifest is empty. Record first, then run build_manifest.py")

    split = make_split(speakers, seed=seed)
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/split.json", "w", encoding="utf-8") as f:
        json.dump(split.__dict__, f, ensure_ascii=False, indent=2)

    print("[Split]")
    print(" train:", split.train_spk)
    print(" val  :", split.val_spk)
    print(" test :", split.test_spk)

    # datasets
    train_ds = LogMelKWSDataset(items, split.train_spk, augment=True)
    val_ds   = LogMelKWSDataset(items, split.val_spk, augment=False)
    test_ds  = LogMelKWSDataset(items, split.test_spk, augment=False)

    # class weights (오탐 방지에 중요: unknown/silence 비율이 달라질 수 있음)
    class_w = compute_class_weights(items).to("cuda" if torch.cuda.is_available() else "cpu")

    # (선택) imbalance가 심하면 sampler 쓰는 것도 좋음
    # 여기서는 간단히 weighted sampler 예시만 제공 (원하면 고도화 가능)
    y_train = [y for _, y in train_ds.items]
    class_count = torch.bincount(torch.tensor(y_train), minlength=len(LABELS)).float()
    per_class_weight = 1.0 / (class_count + 1e-6)
    sample_weights = torch.tensor([per_class_weight[y] for y in y_train], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_ld = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=2, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KWS2DCNN(n_class=len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

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
            torch.save(model.state_dict(), "checkpoints/best_mel2d.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStopping] no improvement for {patience} epochs.")
                break

    model.load_state_dict(torch.load("checkpoints/best_mel2d.pt", map_location=device))
    te_loss, te_acc = run_epoch(model, test_ld, opt=None, device=device, class_weights=class_w)
    print(f"[TEST] acc={te_acc:.3f}")

if __name__ == "__main__":
    main()
