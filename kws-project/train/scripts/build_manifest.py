import os, glob, csv

LABELS = ["next","prev","stop","play","unknown","silence"]
#LABELS = ["next","prev","stop","play","unknown","silence","wake","quit"]

def main():
    root = "data/raw"
    rows = []
    speakers = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])

    for spk in speakers:
        for lbl in LABELS:
            pattern = os.path.join(root, spk, lbl, "*.wav")
            for path in glob.glob(pattern):
                rows.append([path.replace("\\","/"), spk, lbl])

    os.makedirs("data", exist_ok=True)
    out = "data/manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path","speaker","label"])
        wr.writerows(rows)

    print(f"[OK] Wrote {out} with {len(rows)} rows")
    if len(rows) == 0:
        print("No data found. Record first: python scripts/record_kws.py --spk spk01")

if __name__ == "__main__":
    main()
