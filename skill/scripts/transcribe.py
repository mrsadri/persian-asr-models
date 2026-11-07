#!/usr/bin/env python3
"""Persian meeting transcription pipeline (sherpa-onnx, CPU only).

Steps: ffmpeg clean-up (mono 16 kHz, high-pass, loudness levelling, optional
tempo) -> Silero VAD (drops silence) -> ASR per speech segment -> JSONL
(resumable) -> compact Markdown transcript with original-time timestamps.

Usage:
  python3 transcribe.py AUDIO [--engine shenava|whisper] [--tempo 1.0]
                        [--out DIR] [--models DIR] [--threads N]
                        [--max-minutes N]
Outputs in --out: <name>.segments.jsonl, <name>.raw.txt, progress.log
(compact.py turns the JSONL into the transcript Claude reads.)
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import sherpa_onnx

SR = 16000


def log(msg, fh=None):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def load_audio(path, tempo, max_minutes):
    filters = ["highpass=f=80", "dynaudnorm=f=150:g=15"]
    if abs(tempo - 1.0) > 1e-3:
        filters.append(f"atempo={tempo}")
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_minutes:
        cmd = ["ffmpeg", "-v", "error", "-t", str(max_minutes * 60), "-i", path]
    cmd += ["-af", ",".join(filters), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0


def vad_segments(samples, vad_model, max_len):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = vad_model
    cfg.silero_vad.threshold = 0.45
    cfg.silero_vad.min_silence_duration = 0.5
    cfg.silero_vad.min_speech_duration = 0.3
    cfg.silero_vad.max_speech_duration = max_len
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max_len * 4 + 60)
    segs, win = [], cfg.silero_vad.window_size
    def drain():
        while not vad.empty():
            segs.append((vad.front.start, np.array(vad.front.samples, dtype=np.float32)))
            vad.pop()
    for i in range(0, len(samples), win):
        vad.accept_waveform(samples[i:i + win]); drain()
    vad.flush(); drain()
    return split_long(segs, max_len)


def split_long(segs, max_len, extra=None):
    """Split pieces longer than max_len at the quietest 0.2 s window in their last 40%.
    segs: [(start_sample, samples)]; extra: parallel list of tags (e.g. speaker)."""
    out, hop = [], int(0.2 * SR)
    for n_, (start, s) in enumerate(segs):
        tag = extra[n_] if extra else None
        while len(s) > max_len * SR:
            lo, hi = int(max_len * 0.6 * SR), int(max_len * SR)
            win = s[lo:hi]
            n = len(win) // hop
            energy = [float(np.mean(win[k*hop:(k+1)*hop] ** 2)) for k in range(n)]
            cut = lo + int(np.argmin(energy)) * hop + hop // 2
            out.append((start, s[:cut], tag)); start += cut; s = s[cut:]
        if len(s) > 0.3 * SR:
            out.append((start, s, tag))
    return out


def make_recognizer(engine, models, threads):
    if engine == "whisper":
        d = os.path.join(models, "sherpa-onnx-whisper-turbo")
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=f"{d}/turbo-encoder.int8.onnx", decoder=f"{d}/turbo-decoder.int8.onnx",
            tokens=f"{d}/turbo-tokens.txt", language="fa", task="transcribe",
            num_threads=threads, tail_paddings=800)
    d = os.path.join(models, "shenava")
    model = f"{d}/model.int8.onnx" if os.path.exists(f"{d}/model.int8.onnx") else f"{d}/model.onnx"
    return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=model, tokens=f"{d}/tokens.txt", num_threads=threads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--engine", default="shenava", choices=["shenava", "whisper"])
    ap.add_argument("--tempo", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=os.path.expanduser("~/.cache/pmt-models"))
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-minutes", type=float, default=None, help="only first N minutes (tests)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--turns", default=None, help="speakers.py JSON: transcribe per speaker turn")
    a = ap.parse_args()

    name = os.path.splitext(os.path.basename(a.audio))[0]
    out = a.out or os.path.join(os.getcwd(), "pmt-out", name)
    os.makedirs(out, exist_ok=True)
    jl_path = os.path.join(out, f"{name}.segments.jsonl")
    logf = open(os.path.join(out, "progress.log"), "a")

    t0 = time.time()
    samples = load_audio(a.audio, a.tempo, a.max_minutes)
    dur = len(samples) / SR
    log(f"audio {dur*a.tempo/60:.1f} min (processed {dur/60:.1f} min at tempo {a.tempo})", logf)

    max_len = 20 if a.engine == "whisper" else 15
    if a.turns:
        if abs(a.tempo - 1.0) > 1e-3:
            sys.exit("--turns cannot be combined with --tempo")
        turns = json.load(open(a.turns, encoding="utf-8"))["turns"]
        raw = [(int(st * SR), samples[int(st * SR):int(en * SR)]) for st, en, _ in turns]
        segs = split_long(raw, max_len, [k for _, _, k in turns])
    else:
        segs = vad_segments(samples, os.path.join(a.models, "silero_vad.onnx"), max_len)
    speech = sum(len(s) for _, s, _ in segs) / SR
    log(f"{'speaker turns' if a.turns else 'VAD'}: {len(segs)} pieces, speech {speech/60:.1f} min ({100*speech/max(dur,1):.0f}% kept)", logf)

    done = set()
    if os.path.exists(jl_path):
        for line in open(jl_path, encoding="utf-8"):
            try: done.add(json.loads(line)["i"])
            except Exception: pass
        if done: log(f"resuming: {len(done)} segments already done", logf)

    rec = make_recognizer(a.engine, a.models, a.threads)
    todo = [(i, st, s, k) for i, (st, s, k) in enumerate(segs) if i not in done]
    jl = open(jl_path, "a", encoding="utf-8")
    t1 = time.time(); proc = 0.0
    for b in range(0, len(todo), a.batch):
        chunk = todo[b:b + a.batch]
        streams = []
        for _, _, s, _ in chunk:
            st = rec.create_stream(); st.accept_waveform(SR, s); streams.append(st)
        rec.decode_streams(streams)
        for (i, start, s, k), st in zip(chunk, streams):
            txt = st.result.text.strip()
            row = {"i": i, "start": round(start / SR * a.tempo, 2),
                   "end": round((start + len(s)) / SR * a.tempo, 2), "text": txt}
            if k is not None:
                row["spk"] = k
            jl.write(json.dumps(row, ensure_ascii=False) + "\n")
            proc += len(s) / SR
        jl.flush()
        el = time.time() - t1
        log(f"{min(b + a.batch, len(todo))}/{len(todo)} segments, "
            f"speed x{proc/max(el,1e-6):.1f} realtime, eta {(speech-proc)/(proc/max(el,1e-6))/60 if proc else 0:.1f} min", logf)
    jl.close()

    rows = sorted((json.loads(l) for l in open(jl_path, encoding="utf-8")), key=lambda r: r["i"])
    with open(os.path.join(out, f"{name}.raw.txt"), "w", encoding="utf-8") as fh:
        for r in rows:
            m, s = divmod(int(r["start"]), 60)
            fh.write(f"[{m:02d}:{s:02d}] {r['text']}\n")
    log(f"DONE in {(time.time()-t0)/60:.1f} min -> {out}", logf)


if __name__ == "__main__":
    main()
