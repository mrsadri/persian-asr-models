#!/usr/bin/env python3
"""Optional: label who speaks when, then attach speaker labels to segments.

  python3 diarize.py AUDIO SEGMENTS.jsonl [--speakers N] [--threshold 0.6]

Uses pyannote-3.0 segmentation + 3D-Speaker CAM++ embeddings (sherpa-onnx). Writes a
"spk" field (1, 2, ...) into every row of SEGMENTS.jsonl (in place, backup kept).
Pass --speakers when the number of people is known: it is much more reliable.
"""
import argparse, json, os, shutil, subprocess, time
import numpy as np
import sherpa_onnx

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("jsonl")
    ap.add_argument("--speakers", type=int, default=-1)
    ap.add_argument("--threshold", type=float, default=1.15, help="only used when --speakers is not given; lower = more speakers")
    ap.add_argument("--models", default=os.path.expanduser("~/.cache/pmt-models"))
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args()

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=f"{a.models}/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"),
            num_threads=a.threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=f"{a.models}/campplus.onnx", num_threads=a.threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=a.speakers, threshold=a.threshold),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    cmd = ["ffmpeg", "-v", "error"] + (["-t", str(a.max_minutes * 60)] if a.max_minutes else []) + \
          ["-i", a.audio, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.int16).astype(np.float32) / 32768
    t = time.time()
    turns = [(r.start, r.end, r.speaker) for r in sd.process(x).sort_by_start_time()]
    print(f"diarization {len(x)/SR/60:.1f} min audio in {(time.time()-t)/60:.1f} min, "
          f"{len(set(s for *_, s in turns))} speakers, {len(turns)} turns")

    # renumber speakers by first appearance: 1, 2, 3 ...
    order = {}
    for _, _, s in turns:
        order.setdefault(s, len(order) + 1)

    rows = [json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()]
    for r in rows:
        best, ov = None, 0.0
        for st, en, s in turns:
            o = min(en, r["end"]) - max(st, r["start"])
            if o > ov:
                best, ov = s, o
        r["spk"] = order.get(best, 0)
    shutil.copy(a.jsonl, a.jsonl + ".bak")
    with open(a.jsonl, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("speaker labels written to", a.jsonl)


if __name__ == "__main__":
    main()
