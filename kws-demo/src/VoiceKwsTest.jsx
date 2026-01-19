import React, { useEffect, useRef, useState } from "react";
import { usePorcupine } from "@picovoice/porcupine-react";

// .env: VITE_PICOVOICE_ACCESS_KEY=xxxxx
const ACCESS_KEY = import.meta.env.VITE_PICOVOICE_ACCESS_KEY;

export default function VoiceKwsTest() {
  const {
    keywordDetection,
    isLoaded,
    isListening,
    error,
    init,
    start,
    stop,
    release,
  } = usePorcupine();

  const lastFiredAtRef = useRef(0);
  const [log, setLog] = useState([]);

  const porcupineKeyword = {
    publicPath: "/porcupine/next.ppn", 
    label: "NEXT",                     
  };

  const porcupineModel = {
    publicPath: "/porcupine/ko.pv",
  };

  useEffect(() => {
    if (!ACCESS_KEY) return;

    init(ACCESS_KEY, porcupineKeyword, porcupineModel);

    return () => {
      release();
    };
  }, []);

  useEffect(() => {
    if (keywordDetection === null) return;

    const now = Date.now();
    if (now - lastFiredAtRef.current < 1200) return;
    lastFiredAtRef.current = now;

    setLog((prev) => [
      { t: new Date().toLocaleTimeString(), label: keywordDetection.label },
      ...prev,
    ]);
  }, [keywordDetection]);

  return (
    <div style={{ padding: 16, fontFamily: "sans-serif" }}>
      <h2>Porcupine KWS Test</h2>

      {error && (
        <pre style={{ color: "crimson", whiteSpace: "pre-wrap" }}>
          {String(error)}
        </pre>
      )}

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button onClick={start} disabled={!isLoaded || isListening}>
          🎤 Start
        </button>
        <button onClick={stop} disabled={!isLoaded || !isListening}>
          ⏹ Stop
        </button>
      </div>

      <p>Status: {isListening ? "Listening" : "Idle"} / Loaded: {String(isLoaded)}</p>

      <h3>Detections</h3>
      <ul>
        {log.slice(0, 10).map((x, i) => (
          <li key={i}>
            [{x.t}] {x.label}
          </li>
        ))}
      </ul>
    </div>
  );
}
