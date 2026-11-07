#!/usr/bin/env python3
"""Turn <name>.segments.jsonl into transcripts Claude can read cheaply.

  python3 compact.py SEGMENTS.jsonl [--glossary FILE] [--para-sec 60] [--no-itn]

Writes next to the JSONL:
  <name>.transcript.md   compact: fillers and stutters removed, glossary fixed,
                         spoken numbers -> digits, one paragraph per ~60 s
                         with a [MM:SS] stamp. THIS is what Claude reads.
  <name>.verbatim.md     every segment with its own stamp, glossary fixed only
                         (for quoting / checking a passage).
Prints word counts and a rough token estimate.
"""
import argparse, json, os, re, sys

ZW = "‌"
FILLERS = {"اِ", "اه", "اوم", "ام", "ممم", "مم", "اا", "ااا", "آآ", "هوم", "اوه"}

# ---------- glossary ----------
def load_glossary(path):
    rules = []
    if not path or not os.path.exists(path):
        return rules
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        good, bad = line.split("|", 1)
        good = good.strip()
        for v in (x.strip() for x in bad.split(",")):
            if v:
                pat = re.escape(v).replace(r"\ ", r"[\s‌]*").replace(ZW, r"[\s‌]*")
                rules.append((len(v), re.compile(r"(?<![\w‌])" + pat + r"(?![\w‌])"), good))
    rules.sort(key=lambda r: -r[0])
    return [(rx, good) for _, rx, good in rules]


def apply_glossary(text, rules):
    for rx, good in rules:
        text = rx.sub(good, text)
    return text

# ---------- spoken numbers -> digits ----------
UNITS = {"صفر": 0, "یک": 1, "یه": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "شیش": 6,
         "هفت": 7, "هشت": 8, "نه": 9, "ده": 10, "یازده": 11, "دوازده": 12, "سیزده": 13,
         "چهارده": 14, "پانزده": 15, "پونزده": 15, "شانزده": 16, "شونزده": 16, "هفده": 17,
         "هیفده": 17, "هجده": 18, "هیجده": 18, "نوزده": 19, "بیست": 20, "سی": 30, "چهل": 40,
         "پنجاه": 50, "شصت": 60, "هفتاد": 70, "هشتاد": 80, "نود": 90, "صد": 100, "یکصد": 100,
         "دویست": 200, "سیصد": 300, "چهارصد": 400, "پانصد": 500, "پونصد": 500, "ششصد": 600,
         "شیشصد": 600, "هفتصد": 700, "هشتصد": 800, "نهصد": 900}
SCALES = {"هزار": 1000, "میلیون": 10**6, "میلیارد": 10**9, "هزارتا": 1000}
FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def itn(text):
    toks = text.split(" ")
    out, i = [], 0
    while i < len(toks):
        j, total, cur, used, words = i, 0, 0, False, 0
        while j < len(toks):
            w = toks[j]
            if w in UNITS:
                cur += UNITS[w]; used = True; words += 1; j += 1
            elif w in SCALES and (used or w != "هزارتا"):
                total += max(cur, 1) * SCALES[w]; cur = 0; used = True; words += 1; j += 1
            elif w == "و" and used and j + 1 < len(toks) and (toks[j+1] in UNITS or toks[j+1] in SCALES):
                j += 1
            else:
                break
        val = total + cur
        # keep a lone small number as a word ("یه کارشناس", "دو تا") - reads better
        if used and (words > 1 or val > 10):
            out.append(str(val).translate(FA_DIGITS)); i = j
        else:
            out.append(toks[i]); i += 1
    return " ".join(out)

# ---------- clean-up ----------
def dedupe(text):
    words = [w for w in text.split() if w not in FILLERS]
    # collapse any n-gram (n=1..6) repeated back to back
    changed = True
    while changed:
        changed = False
        for n in range(1, 7):
            k = 0
            while k + 2 * n <= len(words):
                if words[k:k+n] == words[k+n:k+2*n]:
                    del words[k+n:k+2*n]; changed = True
                else:
                    k += 1
    return " ".join(words)


def zwnj(text):
    """Join می/نمی prefixes and ها suffixes with ZWNJ (می شه -> می‌شه)."""
    text = re.sub(r"(?<![\w\u200c])(ن?می) (?=[\u0600-\u06FF])", "\\1" + ZW, text)
    text = re.sub(r"(?<=[\u0600-\u06FF]) (ها|های|هایی|هام|هات|هاش|هامون|هاتون|هاشون)(?![\w\u200c])", ZW + "\\1", text)
    return text


def stamp(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--glossary", default=os.path.join(here, "..", "references", "glossary.txt"))
    ap.add_argument("--extra-glossary", default=None, help="second glossary file (session-specific)")
    ap.add_argument("--para-sec", type=float, default=60)
    ap.add_argument("--no-itn", action="store_true")
    ap.add_argument("--turns", default=None, help="speakers.py JSON (adds speaker stats header)")
    ap.add_argument("--names", default=None, help='JSON {"1": "مسیح (Design Lead)", ...} to label speakers')
    a = ap.parse_args()

    rules = load_glossary(a.glossary) + load_glossary(a.extra_glossary)
    spk_info = json.load(open(a.turns, encoding="utf-8"))["speakers"] if a.turns and os.path.exists(a.turns) else {}
    names = json.load(open(a.names, encoding="utf-8")) if a.names and os.path.exists(a.names) else {}
    def label(k):
        k = str(k)
        if k in names:
            return names[k]
        m = spk_info.get(k, {}).get("match")
        return f"S{k} ({m})" if m else f"S{k}"
    rows = sorted((json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()), key=lambda r: r["i"])
    base = a.jsonl.replace(".segments.jsonl", "")
    name = os.path.basename(base)

    verb_lines, paras, cur, cur_start, cur_spk = [], [], [], None, None
    raw_words = 0
    for r in rows:
        t = re.sub(r"\s+", " ", r["text"]).strip()
        if not t:
            continue
        raw_words += len(t.split())
        spk = r.get("spk")
        tag = f"{label(spk)}: " if spk else ""
        g = zwnj(apply_glossary(t, rules))
        verb_lines.append(f"[{stamp(r['start'])}] {tag}{g}")
        c = dedupe(g)
        if not a.no_itn:
            c = itn(c)
        if cur and spk != cur_spk:
            paras.append((cur_start, cur_spk, " ".join(cur))); cur, cur_start = [], None
        if cur_start is None:
            cur_start, cur_spk = r["start"], spk
        cur.append(c)
        if r["end"] - cur_start >= a.para_sec:
            paras.append((cur_start, cur_spk, " ".join(cur))); cur, cur_start = [], None
    if cur:
        paras.append((cur_start, cur_spk, " ".join(cur)))

    dur = rows[-1]["end"] if rows else 0
    with open(base + ".verbatim.md", "w", encoding="utf-8") as fh:
        fh.write(f"# {name} (verbatim)\n\n" + "\n".join(verb_lines) + "\n")
    with open(base + ".transcript.md", "w", encoding="utf-8") as fh:
        fh.write(f"# {name}\n\nDuration: {stamp(dur)} | auto-transcribed, may contain errors\n\n")
        if spk_info:
            fh.write("| Speaker | Talk time | Voice match |\n|---|---|---|\n")
            for k, v in spk_info.items():
                vm = f"{v['match']} ({v['sim']})" if v.get("match") else (f"none (best {v['sim']})" if v.get("sim") else "-")
                fh.write(f"| {label(k)} | {stamp(v['seconds'])} ({int(v['share']*100)}%) | {vm} |\n")
            fh.write("\n")
        for st, spk, txt in paras:
            fh.write(f"[{stamp(st)}] {label(spk)+': ' if spk else ''}{txt}\n\n")
    comp_words = sum(len(t.split()) for _, _, t in paras)
    print(f"segments={len(rows)} duration={stamp(dur)} raw_words={raw_words} "
          f"compact_words={comp_words} (~{int(comp_words*2.2)} tokens to read)")
    print(base + ".transcript.md")
    print(base + ".verbatim.md")


if __name__ == "__main__":
    main()
