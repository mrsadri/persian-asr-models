"""Download Shenava Koochik (sherpa-onnx bundle) from Hugging Face, make an
int8 copy for fast CPU inference, keep the model metadata, and pack release
archives. Runs inside GitHub Actions (Hugging Face is reachable there)."""
import os, shutil, tarfile
import onnx
from huggingface_hub import snapshot_download
from onnxruntime.quantization import quantize_dynamic, QuantType

REPO = "Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx"
src = snapshot_download(REPO, local_dir="hf")
out = "shenava"; os.makedirs(out, exist_ok=True)
for f in ["tokens.txt", "persian_itn.py", "hotwords_fa.txt", "LICENSE", "README.md"]:
    if os.path.exists(f"hf/{f}"):
        shutil.copy(f"hf/{f}", f"{out}/{f}")

quantize_dynamic("hf/model.onnx", f"{out}/model.int8.onnx",
                 op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8)
# quantization can drop custom metadata; sherpa-onnx needs it (normalize_type etc.)
orig = onnx.load("hf/model.onnx", load_external_data=False)
q = onnx.load(f"{out}/model.int8.onnx")
have = {p.key for p in q.metadata_props}
for p in orig.metadata_props:
    if p.key not in have:
        q.metadata_props.add(key=p.key, value=p.value)
onnx.save(q, f"{out}/model.int8.onnx")
print({p.key: p.value[:60] for p in q.metadata_props})

with tarfile.open("shenava-koochik-v1.0-int8.tar.bz2", "w:bz2") as t:
    t.add(out, arcname="shenava")
shutil.copy("hf/model.onnx", "shenava-koochik-v1.0-fp32.onnx")
for f in os.listdir("."):
    if f.endswith((".tar.bz2", ".onnx")):
        print(f, os.path.getsize(f) // 2**20, "MB")
