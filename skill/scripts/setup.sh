#!/usr/bin/env bash
# Idempotent setup: Python deps + models into ~/.cache/pmt-models.
# Usage: bash setup.sh [shenava|whisper|diarize|all]   (default: shenava)
# All downloads come from GitHub Releases (Hugging Face / OpenAI are blocked
# in Claude's sandbox). Safe to re-run; skips what already exists.
set -euo pipefail
WHAT="${1:-shenava}"
M="${PMT_MODELS:-$HOME/.cache/pmt-models}"
K2=https://github.com/k2-fsa/sherpa-onnx/releases/download
MIRROR=https://github.com/mrsadri/persian-asr-models/releases/download/shenava-koochik-v1.0
mkdir -p "$M"; cd "$M"

python3 -c "import sherpa_onnx, numpy" 2>/dev/null || \
  pip install -q --break-system-packages sherpa-onnx numpy 2>&1 | grep -v WARNING || true
command -v ffmpeg >/dev/null || { echo "ffmpeg missing: apt-get install -y ffmpeg"; exit 1; }

get() { # url dest
  [ -s "$2" ] && return 0
  echo "downloading $(basename "$2") ..."
  curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

get $K2/asr-models/silero_vad.onnx silero_vad.onnx

if [[ "$WHAT" == shenava || "$WHAT" == all ]]; then
  if [ ! -s shenava/model.int8.onnx ]; then
    get $MIRROR/shenava-koochik-v1.0-int8.tar.bz2 shenava.tar.bz2
    tar xjf shenava.tar.bz2 && rm -f shenava.tar.bz2
  fi
fi
if [[ "$WHAT" == whisper || "$WHAT" == all ]]; then
  if [ ! -s sherpa-onnx-whisper-turbo/turbo-encoder.int8.onnx ]; then
    get $K2/asr-models/sherpa-onnx-whisper-turbo.tar.bz2 whisper.tar.bz2
    tar xjf whisper.tar.bz2 && rm -f whisper.tar.bz2
    rm -f sherpa-onnx-whisper-turbo/turbo-*.onnx.weights 2>/dev/null || true
    find sherpa-onnx-whisper-turbo -name "*.onnx" ! -name "*int8*" -delete
  fi
fi
if [[ "$WHAT" == diarize || "$WHAT" == all ]]; then
  [ -s sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx ] || \
    curl -fsSL $K2/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 | tar xj
  get $K2/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx campplus.onnx
fi
echo "READY ($WHAT) in $M"; ls "$M"
