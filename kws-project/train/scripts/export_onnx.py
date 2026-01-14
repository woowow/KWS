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
