#!/usr/bin/env bash
# One command for the whole pipeline.
# Usage: bash run.sh AUDIO [--engine shenava|whisper] [--speakers N|auto] [--glossary FILE] [--out DIR] [--max-minutes M] [--tempo T]
#   --speakers: omit = no speaker labels; N = known head count; auto = guess.
# Resumable: re-running continues where it stopped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AUDIO="$1"; shift
ENGINE=shenava; SPK=""; TEMPO=""; OUT=""; MAXM=""; GLOS=""
while [ $# -gt 0 ]; do case "$1" in
  --engine) ENGINE="$2"; shift 2;; --speakers) SPK="$2"; shift 2;;
  --tempo) TEMPO="$2"; shift 2;; --out) OUT="$2"; shift 2;;
  --max-minutes) MAXM="$2"; shift 2;; --glossary) GLOS="$2"; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done

NAME="$(basename "${AUDIO%.*}")"
OUT="${OUT:-$PWD/pmt-out/$NAME}"; mkdir -p "$OUT"
M="${PMT_MODELS:-$HOME/.cache/pmt-models}"

# engine fallback: if the Shenava mirror is unreachable, use Whisper
if [ "$ENGINE" = shenava ] && ! bash "$HERE/setup.sh" shenava >"$OUT/setup.log" 2>&1; then
  echo "Shenava unavailable (see $OUT/setup.log) -> falling back to whisper"; ENGINE=whisper
fi
[ "$ENGINE" = whisper ] && bash "$HERE/setup.sh" whisper >>"$OUT/setup.log" 2>&1
[ -n "$SPK" ] && bash "$HERE/setup.sh" diarize >>"$OUT/setup.log" 2>&1

ARGS=(--engine "$ENGINE" --out "$OUT" --models "$M")
[ -n "$TEMPO" ] && ARGS+=(--tempo "$TEMPO")
[ -n "$MAXM" ] && ARGS+=(--max-minutes "$MAXM")
python3 "$HERE/transcribe.py" "$AUDIO" "${ARGS[@]}"

JL="$OUT/$NAME.segments.jsonl"
if [ -n "$SPK" ] && ! grep -q '"spk"' "$JL"; then
  D=(--models "$M"); [ "$SPK" != auto ] && D+=(--speakers "$SPK")
  [ -n "$MAXM" ] && D+=(--max-minutes "$MAXM")
  python3 "$HERE/diarize.py" "$AUDIO" "$JL" "${D[@]}"
fi
python3 "$HERE/compact.py" "$JL" $([ -n "$GLOS" ] && echo --extra-glossary "$GLOS")
echo "PIPELINE_DONE engine=$ENGINE out=$OUT"
