# persian-asr-models

Mirror of open Persian speech-recognition models, packaged for
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), so they can be downloaded
from GitHub Releases in environments where Hugging Face is not reachable.

| Release asset | What |
|---|---|
| `shenava-koochik-v1.0-int8.tar.bz2` | Shenava Koochik v1.0 (114M, FastConformer CTC), int8 MatMul quantized, with `tokens.txt` and `persian_itn.py` |
| `shenava-koochik-v1.0-fp32.onnx` | Original fp32 model |

Source: [Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx](https://huggingface.co/Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx), Apache-2.0. All credit to the original authors.

Rebuilt by `.github/workflows/mirror.yml` (run it from the Actions tab).
