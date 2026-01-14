import os

LABELS = ["next","prev","stop","play","unknown","silence"]

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("exports", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)
    print("[OK] Created folders:")
    print(" - data/raw")
    print(" - checkpoints")
    print(" - exports")
    print("\nLabels:", LABELS)
    print("\nNext commands:")
    print("  pip install -r requirements.txt")
    print("  python scripts/record_kws.py --spk spk01")
    print("  python scripts/build_manifest.py")
    print("  python scripts/train_kws.py")
    print("  python scripts/export_onnx.py")
    print("  python scripts/sanity_check.py --copy-to ../web/public/models")
if __name__ == '__main__':
    main()
