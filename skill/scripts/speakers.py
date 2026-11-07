#!/usr/bin/env python3
"""Who speaks when: estimate the number of speakers, diarize, and match known voices.

  python3 speakers.py AUDIO OUT_TURNS.json [--speakers N|auto] [--voiceprints FILE]
                       [--enroll NAME:SPK --enroll-out FILE]

1. Speaker count (if auto): CAM++ embeddings of speech windows -> average-linkage
   clustering at several thresholds; tiny clusters (<4% talk time) are merged.
   The count is "stable" when the thresholds agree.
2. Diarization: 1.5 s windows (0.75 s hop) -> CAM++ embeddings -> nearest
   speaker centroid, smoothed over 5 windows, merged into turns.
3. Per-speaker voiceprint (cluster centroid) compared with
   known voiceprints (JSON {name: [192 floats]}); cosine >= --match labels them.
4. --enroll NAME:K saves speaker K's voiceprint under NAME into --enroll-out.

Writes JSON: {"n": N, "n_stable": bool, "n_candidates": {...},
              "speakers": {"1": {"seconds", "share", "match", "sim"}, ...},
              "turns": [[start, end, spk], ...]}
"""
import argparse, json, os, subprocess, time
import numpy as np
import sherpa_onnx

SR = 16000


def load(path, max_minutes=None):
    cmd = ["ffmpeg", "-v", "error"] + (["-t", str(max_minutes * 60)] if max_minutes else []) + \
          ["-i", path, "-af", "highpass=f=80", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.int16).astype(np.float32) / 32768


def embedder(models, threads):
    return sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=f"{models}/campplus.onnx", num_threads=threads))


def embed(ext, x):
    st = ext.create_stream(); st.accept_waveform(SR, x); st.input_finished()
    v = np.array(ext.compute(st), dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def speech_windows(x, models, win=1.5, hop=0.75):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = f"{models}/silero_vad.onnx"
    cfg.silero_vad.threshold = 0.45
    cfg.silero_vad.min_silence_duration = 0.5
    cfg.silero_vad.min_speech_duration = 0.3
    cfg.silero_vad.max_speech_duration = 15
    cfg.sample_rate = SR
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)
    out, w = [], cfg.silero_vad.window_size
    def drain():
        while not vad.empty():
            s = np.array(vad.front.samples, dtype=np.float32); st = vad.front.start
            n, h = int(win * SR), int(hop * SR)
            vad.pop()
            if len(s) < n:
                if len(s) >= 0.5 * SR:
                    out.append((st, s))
                continue
            for k in range(0, len(s) - n + 1, h):
                out.append((st + k, s[k:k + n]))
    for i in range(0, len(x), w):
        vad.accept_waveform(x[i:i + w]); drain()
    vad.flush(); drain()
    return out


def agglomerate(E, dur, thr=None, n=None, min_share=0.04):
    """Average-linkage agglomerative clustering on cosine distance. Stops at distance
    thr or at n clusters. Returns centroids of clusters holding >= min_share of time."""
    clusters = [[i] for i in range(len(E))]
    cents = [E[i].copy() for i in range(len(E))]
    weights = [dur[i] for i in range(len(E))]
    while len(clusters) > 1:
        C = np.array([c / (np.linalg.norm(c) + 1e-9) for c in cents])
        D = 1 - C @ C.T
        np.fill_diagonal(D, 9)
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if (thr is not None and D[i, j] > thr) or (n is not None and len(clusters) <= n):
            break
        wi, wj = weights[i], weights[j]
        cents[i] = (cents[i] * wi + cents[j] * wj) / (wi + wj); weights[i] = wi + wj
        clusters[i] += clusters[j]
        del clusters[j], cents[j], weights[j]
    total = sum(weights)
    keep = [c / (np.linalg.norm(c) + 1e-9) for c, w in zip(cents, weights) if w / total >= min_share]
    return keep or [cents[int(np.argmax(weights))]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("out")
    ap.add_argument("--speakers", default="auto")
    ap.add_argument("--voiceprints", default=None)
    ap.add_argument("--match", type=float, default=0.5)
    ap.add_argument("--enroll", default=None, help="NAME:SPK, e.g. مسیح:2")
    ap.add_argument("--enroll-out", default=None)
    ap.add_argument("--models", default=os.path.expanduser("~/.cache/pmt-models"))
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-minutes", type=float, default=None)
    a = ap.parse_args()

    t0 = time.time()
    x = load(a.audio, a.max_minutes)
    ext = embedder(a.models, a.threads)
    res = {"n_candidates": {}}

    wins = speech_windows(x, a.models)
    E = np.array([embed(ext, s) for _, s in wins]); dur = np.array([len(s) / SR for _, s in wins])
    idx = np.linspace(0, len(wins) - 1, min(350, len(wins))).astype(int)   # clustering sample
    if a.speakers == "auto":
        for thr in (0.35, 0.45, 0.55):
            res["n_candidates"][str(thr)] = len(agglomerate(E[idx], dur[idx], thr=thr))
        vals = list(res["n_candidates"].values())
        n = max(set(vals), key=vals.count)
        res["n_stable"] = len(set(vals)) == 1
    else:
        n = int(a.speakers); res["n_stable"] = True
    print(f"speakers: {n} ({'stable' if res['n_stable'] else 'UNSTABLE ' + str(res['n_candidates'])})", flush=True)

    # centroids from the sample, refined k-means style on all windows
    C = np.array(agglomerate(E[idx], dur[idx], n=n, min_share=0.0))
    for _ in range(5):
        lab = np.argmax(E @ C.T, axis=1)
        C = np.array([E[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(len(C))])
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    lab = np.argmax(E @ C.T, axis=1)
    # smooth: majority over 5 neighbouring windows
    sm = lab.copy()
    for i in range(len(lab)):
        nb = lab[max(0, i - 2):i + 3]; sm[i] = np.bincount(nb, minlength=len(C)).argmax()
    # windows -> turns (each window owns its first hop, last one its full length)
    order, turns = {}, []
    for (st, s), k in zip(wins, sm):
        st_s = st / SR; en_s = st_s + len(s) / SR
        spk = order.setdefault(int(k), len(order) + 1)
        if turns and turns[-1][2] == spk and st_s - turns[-1][1] < 0.8:
            turns[-1][1] = max(turns[-1][1], en_s)
        else:
            if turns and st_s < turns[-1][1]:
                turns[-1][1] = st_s + 0.375          # split overlapping windows midway
                st_s = turns[-1][1]
            turns.append([st_s, en_s, spk])
    turns = [[round(a_, 2), round(b_, 2), k] for a_, b_, k in turns if b_ - a_ >= 0.3]
    res["n"] = len(order)
    res["turns"] = turns
    cent = {order[k]: C[k] for k in order}

    known = {}
    if a.voiceprints and os.path.exists(a.voiceprints):
        known = {k: np.array(v, dtype=np.float32) for k, v in json.load(open(a.voiceprints, encoding="utf-8")).items()}
    total = sum(en - st for st, en, _ in turns) or 1
    speakers, prints = {}, {}
    for k in sorted({t[2] for t in turns}):
        sec = sum(en - st for st, en, s in turns if s == k)
        info = {"seconds": round(sec), "share": round(sec / total, 2), "match": None, "sim": None}
        c = cent[k]; prints[k] = c
        if known:
            name, sim = max(((nm, float(c @ v / (np.linalg.norm(v) + 1e-9))) for nm, v in known.items()), key=lambda t: t[1])
            info["sim"] = round(sim, 2)
            if sim >= a.match:
                info["match"] = name
        speakers[str(k)] = info
    res["speakers"] = speakers

    if a.enroll and a.enroll_out:
        name, k = a.enroll.rsplit(":", 1)
        db = json.load(open(a.enroll_out, encoding="utf-8")) if os.path.exists(a.enroll_out) else {}
        db[name] = [round(float(v), 4) for v in prints[int(k)]]
        json.dump(db, open(a.enroll_out, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"enrolled {name} from S{k} -> {a.enroll_out}")

    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in speakers.items():
        print(f"S{k}: {v['seconds']}s ({int(v['share']*100)}%)" + (f" = {v['match']} (sim {v['sim']})" if v["match"] else (f" best sim {v['sim']}" if v["sim"] else "")))
    print(f"speakers done in {(time.time()-t0)/60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
