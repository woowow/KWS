import os, argparse, shutil

def assert_nonempty(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    size = os.path.getsize(path)
    if size <= 0:
        raise RuntimeError(f"Empty file (0 bytes): {path}")
    return size

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="exports/kws.onnx")
    ap.add_argument("--copy-to", default=None)  # ../web/public/models
    args = ap.parse_args()

    size = assert_nonempty(args.onnx)
    print(f"[OK] ONNX exists: {args.onnx} ({size} bytes)")

    if args.copy_to:
        os.makedirs(args.copy_to, exist_ok=True)
        dst = os.path.join(args.copy_to, "kws.onnx")
        shutil.copy2(args.onnx, dst)
        dsize = assert_nonempty(dst)
        print(f"[OK] Copied to web: {dst} ({dsize} bytes)")

if __name__ == "__main__":
    main()
