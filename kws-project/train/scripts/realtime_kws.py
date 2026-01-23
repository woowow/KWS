import time
import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd

from train_kws import RawKWSNet, LABELS, SR, NSAMP

CKPT = "checkpoints/best.pt"

# 실시간 동작 파라미터(시연용 추천값)
STEP_SEC = 0.10       # 100ms마다 추론
CONFIRM_N = 2         # 같은 결과가 연속 2번 나오면 확정
COOLDOWN_SEC = 1.0    # 확정 후 1초간 재트리거 방지
THRESH = 0.70         # 확률 임계치 (시연이면 0.6~0.8 사이로 튜닝)

IGNORE = {"unknown", "silence"}  # 이런 라벨은 트리거로 취급 안 함

def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.mean()
    x = x / (x.std() + 1e-6)
    return x

def main():
    model = RawKWSNet(n_class=len(LABELS))
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    ring = np.zeros(NSAMP, dtype=np.float32)
    step = int(SR * STEP_SEC)

    last_label = None
    streak = 0
    cooldown_until = 0.0

    print("[KWS] start (Ctrl+C to stop)")
    print("labels:", LABELS)

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=step) as stream:
        while True:
            audio, _ = stream.read(step)        # (step, 1)
            chunk = audio[:, 0]

            # 링버퍼 갱신
            ring[:-step] = ring[step:]
            ring[-step:] = chunk

            # cooldown 중이면 스킵(중복 트리거 방지)
            now = time.time()
            if now < cooldown_until:
                continue

            x = normalize(ring.copy())
            xt = torch.from_numpy(x).unsqueeze(0)  # (1, NSAMP)

            with torch.no_grad():
                logits = model(xt)
                prob = F.softmax(logits, dim=1)[0]
                idx = int(prob.argmax().item())
                label = LABELS[idx]
                p = float(prob[idx])

            # 임계치 미달이면 streak 리셋
            if p < THRESH or label in IGNORE:
                last_label = None
                streak = 0
                continue

            # 연속 감지 로직
            if label == last_label:
                streak += 1
            else:
                last_label = label
                streak = 1

            # 확정
            if streak >= CONFIRM_N:
                print(f"[TRIGGER] {label} (p={p:.2f})")
                cooldown_until = time.time() + COOLDOWN_SEC
                last_label = None
                streak = 0

if __name__ == "__main__":
    main()
