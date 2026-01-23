import os, time, argparse, re
import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000

ALL_LABELS = ["next","prev","stop","play","wake","quit","unknown","silence"]
PROMPT = {"next":"다음 단계","prev":"이전 단계","stop":"일시 정지","play":"이어 하기","wake":"셰프님", "quit":"채팅 종료"}

DEFAULT_TARGET = {
    "next": 40,
    "prev": 40,
    "stop": 40,
    "play": 40,
    "wake": 40,
    "quit": 40,
    "unknown": 120,
    "silence": 60,
}

IDX_RE = re.compile(r"^(?P<label>[a-z]+)_(?P<idx>\d{4})\.wav$")


def beep():
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass


def record_one(n_samples: int) -> np.ndarray:
    audio = sd.rec(n_samples, samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


def save_wav(path: str, audio: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, SR)


def parse_counts(counts_list):
    if not counts_list:
        return {}
    out = {}
    for x in counts_list:
        if "=" not in x:
            raise ValueError(f"Invalid --counts item: {x} (expected label=num)")
        k, v = x.split("=", 1)
        k = k.strip()
        v = int(v.strip())
        if k not in ALL_LABELS:
            raise ValueError(f"Unknown label in --counts: {k}")
        out[k] = v
    return out


def next_index_for_label(dir_path: str, label: str) -> int:
    if not os.path.exists(dir_path):
        return 0
    max_idx = -1
    for name in os.listdir(dir_path):
        m = IDX_RE.match(name)
        if not m:
            continue
        if m.group("label") != label:
            continue
        idx = int(m.group("idx"))
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


def countdown(seconds: float = 1.2):
    steps = [3, 2, 1]
    per = seconds / len(steps)
    for s in steps:
        print(f"  {s}...", flush=True)
        time.sleep(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spk", required=True, help="speaker id e.g. spk01")
    ap.add_argument("--root", default="data/raw", help="output root")
    ap.add_argument("--dur", type=float, default=1.5, help="record duration seconds (e.g., 1.5)")
    ap.add_argument(
        "--labels",
        nargs="+",
        default=ALL_LABELS,
        help=f"labels to record (default: all). options: {ALL_LABELS}",
    )
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--counts", nargs="*", default=None)
    ap.add_argument("--cooldown", type=float, default=0.25, help="rest seconds between clips")

    ap.add_argument("--append", action="store_true", help="append after existing files")
    ap.add_argument("--start", type=int, default=None, help="force starting index (overrides --append)")
    ap.add_argument("--no-beep", action="store_true", help="disable beep sound")

    args = ap.parse_args()

    labels = [x.strip() for x in args.labels]
    for l in labels:
        if l not in ALL_LABELS:
            raise ValueError(f"Unknown label: {l}. Choose from {ALL_LABELS}")

    per_label = dict(DEFAULT_TARGET)
    per_override = parse_counts(args.counts)

    if args.count is not None:
        for l in labels:
            per_label[l] = args.count

    for k, v in per_override.items():
        per_label[k] = v

    n_samples = int(SR * args.dur)
    base = os.path.join(args.root, args.spk)

    print("\n[Guide]")
    print(" next=다음, prev=이전, stop=중지, play=재생, wake=셰프님, quit=채팅 종료")
    print(" unknown=아무 말(짧은 문장/감탄사), silence=무음\n")
    print(f"[Config] spk={args.spk}, dur={args.dur}s, sr={SR}")
    print(f"[Config] append={args.append}, start={args.start}, cooldown={args.cooldown}s")
    print("[Config] labels & counts:")
    for l in labels:
        print(f"  - {l}: {per_label[l]}")

    for label in labels:
        target = per_label[label]
        label_dir = os.path.join(base, label)

        # start index
        if args.start is not None:
            start_idx = args.start
        elif args.append:
            start_idx = next_index_for_label(label_dir, label)
        else:
            start_idx = 0

        if args.append and args.start is None:
            print(f"\n[Append] {label}: start from {start_idx:04d}")

        for i in range(target):
            cur_idx = start_idx + i

            if label in PROMPT:
                title = f"{label} ({PROMPT[label]})"
            elif label == "unknown":
                title = "unknown (아무 말)"
            else:
                title = "silence (무음)"

            print(f"\n[{args.spk}] {title}  #{i+1}/{target}")
            print("말할 준비!", flush=True)
            countdown(1.2)

            if not args.no_beep:
                beep()

            print(f"RECORDING ({args.dur:.2f}s)... 깨어나라.. 봉인된 휘민이여... ", flush=True)
            audio = record_one(n_samples)
            print("⏹ DONE", flush=True)

            out = os.path.join(label_dir, f"{label}_{cur_idx:04d}.wav")
            save_wav(out, audio)
            print(f"✅ saved: {out}", flush=True)

            time.sleep(args.cooldown)

    print("\n[Done] Recorded:", base)


if __name__ == "__main__":
    main()
