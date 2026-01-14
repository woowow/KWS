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
