# scripts/eval_kws.py
import os
import glob
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from train_kws import RawKWSNet, LABELS, SR, NSAMP

CKPT = "checkpoints/best.pt"
ROOT = "data/raw"

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean()
    x = x / (x.std() + 1e-6)
    return x

def fix_len(audio: np.ndarray) -> np.ndarray:
    n = len(audio)
    if n >= NSAMP:
        # 평가에서는 "앞에서부터" 자르는 고정 크롭(재현성↑)
        return audio[:NSAMP]
    return np.pad(audio, (0, NSAMP - n), mode="constant")

def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise RuntimeError(f"SR mismatch: expected {SR}, got {sr} ({path})")
    audio = fix_len(audio)
    audio = normalize(audio)
    return audio

@torch.no_grad()
def predict(model: torch.nn.Module, wav_1d: np.ndarray) -> tuple[str, float, np.ndarray]:
    x = torch.from_numpy(wav_1d).unsqueeze(0)  # (1, NSAMP)
    logits = model(x)
    prob = F.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(prob.argmax())
    return LABELS[idx], float(prob[idx]), prob

def iter_files(speakers=None, labels=None):
    if speakers is None:
        speakers = sorted([d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))])
    if labels is None:
        labels = LABELS

    for spk in speakers:
        for lbl in labels:
            pattern = os.path.join(ROOT, spk, lbl, "*.wav")
            for path in glob.glob(pattern):
                yield spk, lbl, path.replace("\\", "/")

def print_confusion(conf: dict, labels: list[str]):
    # conf[true][pred] = count
    print("\n[Confusion Matrix] rows=true, cols=pred")
    header = "true\\pred".ljust(12) + " ".join([c.rjust(8) for c in labels])
    print(header)
    for t in labels:
        row = [conf[t].get(p, 0) for p in labels]
        print(t.ljust(12) + " ".join([str(v).rjust(8) for v in row]))

def main():
    # ====== 설정: 여기만 바꿔서 실험 ======
    speakers = ["spk03"]            # 예: ["spk06"] 또는 ["spk01","spk02","spk03"]
    labels_to_eval = ["next", "prev", "stop", "play"]  # unknown/silence 빼고 보고 싶으면 이렇게
    # labels_to_eval = LABELS        # 전부 평가하려면 이걸로
    # ====================================

    print("[Config]")
    print(" CKPT:", CKPT)
    print(" LABELS:", LABELS)
    print(" speakers:", speakers)
    print(" eval_labels:", labels_to_eval)

    model = RawKWSNet(n_class=len(LABELS))
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    conf = defaultdict(Counter)  # conf[true][pred] = count
    per_spk = defaultdict(lambda: {"correct": 0, "total": 0})
    per_true = defaultdict(lambda: {"correct": 0, "total": 0})
    preds_hist = Counter()

    total = 0
    correct = 0
    errors = []

    for spk, true_lbl, path in iter_files(speakers=speakers, labels=labels_to_eval):
        wav = load_audio(path)
        pred_lbl, p, prob = predict(model, wav)

        conf[true_lbl][pred_lbl] += 1
        preds_hist[pred_lbl] += 1

        total += 1
        per_spk[spk]["total"] += 1
        per_true[true_lbl]["total"] += 1

        if pred_lbl == true_lbl:
            correct += 1
            per_spk[spk]["correct"] += 1
            per_true[true_lbl]["correct"] += 1
        else:
            # 상위 3개 확률도 같이 기록
            top3 = np.argsort(-prob)[:3]
            top3_str = ", ".join([f"{LABELS[i]}:{prob[i]:.2f}" for i in top3])
            errors.append((spk, true_lbl, pred_lbl, p, path, top3_str))

    if total == 0:
        print("\n[ERROR] No files found. Check ROOT/speakers/labels path.")
        return

    print(f"\n[Overall] acc = {correct/total:.3f} ({correct}/{total})")
    print("\n[Pred distribution]")
    for k, v in preds_hist.most_common():
        print(f"  {k}: {v}")

    print("\n[Per-speaker acc]")
    for spk in sorted(per_spk.keys()):
        c, t = per_spk[spk]["correct"], per_spk[spk]["total"]
        print(f"  {spk}: {c/t:.3f} ({c}/{t})")

    print("\n[Per-label acc]")
    for t_lbl in labels_to_eval:
        c, t = per_true[t_lbl]["correct"], per_true[t_lbl]["total"]
        if t > 0:
            print(f"  {t_lbl}: {c/t:.3f} ({c}/{t})")

    print_confusion(conf, labels_to_eval)

    # 오분류 상위 몇 개 출력
    if errors:
        print("\n[Top misclassifications] (showing up to 20)")
        for e in errors[:20]:
            spk, t_lbl, p_lbl, p, path, top3 = e
            print(f"  {spk} true={t_lbl} pred={p_lbl} p={p:.2f} | {top3} | {path}")

if __name__ == "__main__":
    main()
