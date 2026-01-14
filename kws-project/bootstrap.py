import os, textwrap, subprocess, sys

NPM = r"C:\Program Files\nodejs\npm.cmd"

def w(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(content).lstrip())

def run(cmd, cwd=None):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd, shell=False)

def main():
    # ---- train files ----
    w("train/requirements.txt", """
    torch
    torchaudio
    numpy
    tqdm
    sounddevice
    soundfile
    scikit-learn
    onnx
    """)

    w("train/scripts/setup_project.py", r"""
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
    """)

    # record script
    w("train/scripts/record_kws.py", r"""
    import os, time, argparse
    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    SR = 16000
    DUR = 1.0
    N = int(SR * DUR)

    LABELS = ["next","prev","stop","play","unknown","silence"]
    PROMPT = {"next":"다음","prev":"이전","stop":"중지","play":"재생"}

    DEFAULT_TARGET = {
        "next": 40,
        "prev": 40,
        "stop": 40,
        "play": 40,
        "unknown": 120,
        "silence": 60,
    }

    def record_one():
        audio = sd.rec(N, samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        return audio.squeeze()

    def save_wav(path, audio):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, audio, SR)

    def countdown():
        time.sleep(0.6); print("  2...")
        time.sleep(0.6); print("  1...")
        time.sleep(0.3)

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--spk", required=True, help="speaker id e.g. spk01")
        ap.add_argument("--root", default="data/raw")
        args = ap.parse_args()

        base = os.path.join(args.root, args.spk)

        print("\n[Guide]")
        print(" next=다음, prev=이전, stop=중지, play=재생")
        print(" unknown=아무 말(짧은 문장/감탄사), silence=무음\n")

        for label in LABELS:
            target = DEFAULT_TARGET[label]
            for i in range(target):
                if label in PROMPT:
                    print(f"[{args.spk}] {label} ({PROMPT[label]})  #{i+1}/{target}  → 녹음")
                elif label == "unknown":
                    print(f"[{args.spk}] unknown (아무 말) #{i+1}/{target} → 녹음")
                else:
                    print(f"[{args.spk}] silence (무음) #{i+1}/{target} → 녹음")

                countdown()
                audio = record_one()

                out = os.path.join(base, label, f"{label}_{i:04d}.wav")
                save_wav(out, audio)
                time.sleep(0.2)

        print("\n[Done] Recorded:", base)

    if __name__ == "__main__":
        main()
    """)

    # build manifest
    w("train/scripts/build_manifest.py", r"""
    import os, glob, csv

    LABELS = ["next","prev","stop","play","unknown","silence"]

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
    """)

    # train (raw waveform 1D CNN) - 웹 추론 쉽게 하려고 mel 없이 raw로 학습
    w("train/scripts/train_kws.py", r"""
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
    """)

    # export onnx
    w("train/scripts/export_onnx.py", r"""
    import os
    import torch
    from train_kws import RawKWSNet, LABELS

    os.makedirs("exports", exist_ok=True)

    model = RawKWSNet(n_class=len(LABELS))
    model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 16000)  # raw waveform 1 sec @ 16k
    out_path = "exports/kws.onnx"

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0:"batch"}},
        opset_version=17
    )

    print("[OK] Exported:", out_path)
    """)

    # sanity check + copy to web
    w("train/scripts/sanity_check.py", r"""
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
    """)

    # ---- web files (React via Vite) ----
    # Create Vite React app if not exists
    if not os.path.exists("web/package.json"):
        # requires npm
        run([sys.executable, "-c", "print('Creating web (Vite React)...')"])
        run([NPM, "create", "vite@latest", "web", "--", "--template", "react"])
    # install deps
    run([NPM, "install"], cwd="web")
    run([NPM, "install", "onnxruntime-web"], cwd="web")

    w("web/src/VoiceKwsOnnx.jsx", r"""
    import React, { useEffect, useRef, useState } from "react";
    import * as ort from "onnxruntime-web";

    const LABELS = ["next","prev","stop","play","unknown","silence"];

    function softmax(arr) {
      const m = Math.max(...arr);
      const exps = arr.map((x) => Math.exp(x - m));
      const s = exps.reduce((a,b) => a+b, 0);
      return exps.map((e) => e / (s || 1));
    }

    // Simple linear resampler (from inputRate -> 16000)
    function resampleTo16k(float32, inputRate) {
      const targetRate = 16000;
      if (inputRate === targetRate) return float32;

      const ratio = inputRate / targetRate;
      const outLen = Math.floor(float32.length / ratio);
      const out = new Float32Array(outLen);

      for (let i = 0; i < outLen; i++) {
        const idx = i * ratio;
        const i0 = Math.floor(idx);
        const i1 = Math.min(i0 + 1, float32.length - 1);
        const t = idx - i0;
        out[i] = float32[i0] * (1 - t) + float32[i1] * t;
      }
      return out;
    }

    export default function VoiceKwsOnnx() {
      const [ready, setReady] = useState(false);
      const [listening, setListening] = useState(false);
      const [status, setStatus] = useState("idle");
      const [logs, setLogs] = useState([]);

      const sessionRef = useRef(null);
      const audioCtxRef = useRef(null);
      const procRef = useRef(null);

      const ringRef = useRef(new Float32Array(16000)); // 1 sec @ 16k
      const ringPosRef = useRef(0);
      const lastFireRef = useRef(0);

      async function loadModel() {
        setStatus("loading model...");
        const sess = await ort.InferenceSession.create("/models/kws.onnx", { executionProviders: ["wasm"] });
        sessionRef.current = sess;
        setReady(true);
        setStatus("model loaded");
      }

      function pushToRing(samples16k) {
        const ring = ringRef.current;
        let pos = ringPosRef.current;

        for (let i = 0; i < samples16k.length; i++) {
          ring[pos] = samples16k[i];
          pos = (pos + 1) % ring.length;
        }
        ringPosRef.current = pos;
      }

      function readRing1s() {
        const ring = ringRef.current;
        const pos = ringPosRef.current;
        const out = new Float32Array(ring.length);
        // out = ring[pos..end] + ring[0..pos-1]
        out.set(ring.subarray(pos));
        out.set(ring.subarray(0, pos), ring.length - pos);
        return out;
      }

      async function inferOnce() {
        const sess = sessionRef.current;
        if (!sess) return;

        // 1 sec waveform
        const x = readRing1s();

        // normalize (same as training)
        let mean = 0;
        for (let i = 0; i < x.length; i++) mean += x[i];
        mean /= x.length;
        let varSum = 0;
        for (let i = 0; i < x.length; i++) {
          const d = x[i] - mean;
          varSum += d * d;
        }
        const std = Math.sqrt(varSum / x.length) + 1e-6;
        for (let i = 0; i < x.length; i++) x[i] = (x[i] - mean) / std;

        const input = new ort.Tensor("float32", x, [1, 16000]);
        const out = await sess.run({ x: input });
        const logits = out.logits.data; // Float32Array
        const probs = softmax(Array.from(logits));

        // pick
        let best = 0;
        for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;

        const label = LABELS[best];
        const p = probs[best];

        // trigger rule
        const now = Date.now();
        const cooldownMs = 1200;
        const threshold = 0.70;

        if ((label === "next" || label === "prev" || label === "stop" || label === "play") &&
            p >= threshold &&
            now - lastFireRef.current > cooldownMs) {
          lastFireRef.current = now;
          setLogs((prev) => [{ t: new Date().toLocaleTimeString(), label, p: p.toFixed(2) }, ...prev].slice(0, 20));
        }
      }

      async function start() {
        if (!ready) return;
        setStatus("requesting mic...");
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        audioCtxRef.current = ctx;

        const src = ctx.createMediaStreamSource(stream);

        // ScriptProcessorNode is deprecated but OK for quick demo
        const proc = ctx.createScriptProcessor(4096, 1, 1);
        procRef.current = proc;

        proc.onaudioprocess = (e) => {
          const input = e.inputBuffer.getChannelData(0);
          const inputRate = ctx.sampleRate;
          const s16 = resampleTo16k(input, inputRate);
          pushToRing(s16);
        };

        src.connect(proc);
        proc.connect(ctx.destination);

        setListening(true);
        setStatus("listening");

        // inference loop
        const loop = async () => {
          if (!sessionRef.current || !procRef.current) return;
          await inferOnce();
          if (procRef.current) requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
      }

      async function stop() {
        if (procRef.current) {
          try { procRef.current.disconnect(); } catch {}
          procRef.current = null;
        }
        if (audioCtxRef.current) {
          try { await audioCtxRef.current.close(); } catch {}
          audioCtxRef.current = null;
        }
        setListening(false);
        setStatus("stopped");
      }

      useEffect(() => {
        loadModel();
        return () => { stop(); };
        // eslint-disable-next-line react-hooks/exhaustive-deps
      }, []);

      return (
        <div style={{ padding: 16, fontFamily: "sans-serif" }}>
          <h2>KWS (PyTorch → ONNX → Web)</h2>
          <p>Model: /models/kws.onnx</p>
          <p>Status: {status} / Ready: {String(ready)} / Listening: {String(listening)}</p>

          <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
            <button onClick={start} disabled={!ready || listening}>🎤 Start</button>
            <button onClick={stop} disabled={!listening}>⏹ Stop</button>
          </div>

          <h3>Triggers</h3>
          <ul>
            {logs.map((x, i) => (
              <li key={i}>[{x.t}] {x.label} (p={x.p})</li>
            ))}
          </ul>

          <p style={{ fontSize: 12, opacity: 0.8 }}>
            Tip: unknown/silence 데이터가 많을수록 오탐이 줄어듭니다.
          </p>
        </div>
      );
    }
    """)

    w("web/src/App.jsx", r"""
    import VoiceKwsOnnx from "./VoiceKwsOnnx.jsx";
    export default function App() {
      return <VoiceKwsOnnx />;
    }
    """)

    print("\n[OK] Bootstrapped everything.")
    print("\nNext:")
    print("  cd train")
    print("  pip install -r requirements.txt")
    print("  python scripts/setup_project.py")
    print("  python scripts/record_kws.py --spk spk01")
    print("  python scripts/record_kws.py --spk spk02")
    print("  python scripts/build_manifest.py")
    print("  python scripts/train_kws.py")
    print("  python scripts/export_onnx.py")
    print("  python scripts/sanity_check.py --copy-to ../web/public/models")
    print("  cd ../web && npm run dev")

if __name__ == "__main__":
    main()
