#!/usr/bin/env bash
# One command for the whole pipeline.
# Usage: bash run.sh AUDIO [--speakers auto|N] [--voiceprints FILE] [--glossary FILE]
#                          [--engine shenava|whisper] [--out DIR] [--max-minutes M] [--tempo T]
#   --speakers: omit = plain transcript; auto = detect count; N = known head count.
#   --voiceprints: JSON {name: vector} of known voices (labels matching speakers).
# Resumable: re-running continues where it stopped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AUDIO="$1"; shift
ENGINE=shenava; SPK=""; TEMPO=""; OUT=""; MAXM=""; GLOS=""; VP=""
while [ $# -gt 0 ]; do case "$1" in
  --engine) ENGINE="$2"; shift 2;; --speakers) SPK="$2"; shift 2;;
  --tempo) TEMPO="$2"; shift 2;; --out) OUT="$2"; shift 2;;
  --max-minutes) MAXM="$2"; shift 2;; --glossary) GLOS="$2"; shift 2;;
  --voiceprints) VP="$2"; shift 2;; *) echo "unknown arg $1"; exit 1;; esac; done

NAME="$(basename "${AUDIO%.*}")"
OUT="${OUT:-$PWD/pmt-out/$NAME}"; mkdir -p "$OUT"
M="${PMT_MODELS:-$HOME/.cache/pmt-models}"

if [ "$ENGINE" = shenava ] && ! bash "$HERE/setup.sh" shenava >"$OUT/setup.log" 2>&1; then
  echo "Shenava unavailable (see $OUT/setup.log) -> falling back to whisper"; ENGINE=whisper
fi
[ "$ENGINE" = whisper ] && bash "$HERE/setup.sh" whisper >>"$OUT/setup.log" 2>&1

ARGS=(--engine "$ENGINE" --out "$OUT" --models "$M")
[ -n "$MAXM" ] && ARGS+=(--max-minutes "$MAXM")
CARGS=()
[ -n "$GLOS" ] && CARGS+=(--extra-glossary "$GLOS")

if [ -n "$SPK" ]; then
  bash "$HERE/setup.sh" speakers >>"$OUT/setup.log" 2>&1
  TURNS="$OUT/$NAME.turns.json"
  if [ ! -s "$TURNS" ]; then
    SA=(--speakers "$SPK" --models "$M"); [ -n "$VP" ] && SA+=(--voiceprints "$VP")
    [ -n "$MAXM" ] && SA+=(--max-minutes "$MAXM")
    python3 "$HERE/speakers.py" "$AUDIO" "$TURNS" "${SA[@]}"
  fi
  ARGS+=(--turns "$TURNS"); CARGS+=(--turns "$TURNS")
else
  [ -n "$TEMPO" ] && ARGS+=(--tempo "$TEMPO")
fi

python3 "$HERE/transcribe.py" "$AUDIO" "${ARGS[@]}"
python3 "$HERE/compact.py" "$OUT/$NAME.segments.jsonl" "${CARGS[@]}"
echo "PIPELINE_DONE engine=$ENGINE out=$OUT"
