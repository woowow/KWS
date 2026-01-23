import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import random

from train_kws import RawKWSNet, LABELS, SR, NSAMP

CKPT = "checkpoints/best.pt"
TEST_WAV = "data/raw/spk06/next/next_0013.wav"

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean()
    x = x / (x.std() + 1e-6)
    return x

def crop_or_pad(audio: np.ndarray) -> np.ndarray:
    n = len(audio)
    if n == NSAMP:
        return audio
    if n < NSAMP:
        return np.pad(audio, (0, NSAMP - n))
    # n > NSAMP: 랜덤 크롭 (학습과 유사)
    start = random.randint(0, n - NSAMP)
    return audio[start:start+NSAMP]

def main():
    audio, sr = sf.read(TEST_WAV, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise RuntimeError(f"SR mismatch: expected {SR}, got {sr}")

    model = RawKWSNet(n_class=len(LABELS))
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    probs = []
    with torch.no_grad():
        for _ in range(10):  # 멀티크롭 횟수
            seg = crop_or_pad(audio)
            seg = normalize(seg)
            x = torch.from_numpy(seg).unsqueeze(0)  # (1, NSAMP)
            logits = model(x)
            prob = F.softmax(logits, dim=1)[0].numpy()
            probs.append(prob)

    mean_prob = np.mean(probs, axis=0)
    idx = int(mean_prob.argmax())

    print("WAV:", TEST_WAV)
    print("Pred:", LABELS[idx], "/ prob:", float(mean_prob[idx]))
    top3 = np.argsort(-mean_prob)[:3]
    for i in top3:
        print(f"  - {LABELS[int(i)]}: {float(mean_prob[int(i)]):.4f}")

    print("CKPT:", CKPT)
    print("LABELS:", LABELS)


if __name__ == "__main__":
    main()
