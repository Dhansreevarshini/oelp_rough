"""
promise_pipeline.py  -  promise-level comparison: speech promises vs manifesto promises

Steps (run from the project root, in this order):
  python promise_pipeline.py manifesto     1) extract specific promises from manifesto chunks
  python promise_pipeline.py speech        2) extract specific promises from speech chunks
  python promise_pipeline.py match         3) match every speech promise to its party's manifesto
  python promise_pipeline.py report        4) tables + charts
  python promise_pipeline.py all           all four
Limit to some parties:  python promise_pipeline.py speech TVK
                        python promise_pipeline.py match DMK TVK

Every step saves after each call and resumes where it stopped, so you can stop and re-run.
Outputs go to output/promises/.

Verdicts for each speech promise
  IN_MANIFESTO : same commitment as a manifesto promise (wording may differ, key details agree)
  RELATED      : same scheme/topic, but key details differ (amount, group, scale, timeline)
  NEW          : not in the manifesto
"""
import os
import re
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================== CONFIG ==============================
ROOT = Path(__file__).resolve().parent
PARTIES = ["AIADMK", "DMK", "TVK"]

MANIFESTO_CHUNKS_BASE = ROOT / "chunks"            # chunks/<PARTY>/*.txt  (auto-searched if different)
SPEECH_CHUNKS_BASE = ROOT / "chunks_speech"        # chunks_speech/<PARTY>/*.txt
SPEECH_RESULTS_BASE = ROOT / "output" / "speech_results"   # <PARTY>/gemini.csv (category filter)
OUT = ROOT / "output" / "promises"

SLEEP_BETWEEN_CALLS = 8      # seconds - same as your classifier; lower it if your quota allows
MAX_ATTEMPTS = 3
RESUME = True

# Speech chunks whose MAIN category is one of these (and have no secondary topic) are skipped
# when extracting promises - saves many API calls. Set FILTER_BY_CLASSIFICATION = False to read everything.
FILTER_BY_CLASSIFICATION = True
SKIP_PRIMARY = {"Content about the Opposition party", "Others", "Already executed promises"}

TOP_K = 5                    # manifesto candidates shown to the judge for each speech promise
BATCH = 5                    # speech promises judged per Gemini call
USE_EMBEDDINGS = True        # uses sentence-transformers if installed, else TF-IDF
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CATEGORIES = [
    "Agriculture and Farmers", "Employment and Jobs", "Women's Development", "Education",
    "Health", "Governance", "Infrastructure", "Religion and Culture",
    "Social justice and Deprived communities", "Law and order", "Welfare and Funds",
    "Divyang welfare", "Others",
]
CAT_LOOKUP = {c.lower(): c for c in CATEGORIES}

COLORS = {"IN_MANIFESTO": "#00D4A0", "RELATED": "#FFD23F", "NEW": "#FF6B35"}
VERDICTS = ["IN_MANIFESTO", "RELATED", "NEW"]
BG, SURFACE = "#0a0a0f", "#12121a"
# ====================================================================

MAN_COLS = ["party", "chunk", "promise_no", "promise", "category", "specificity", "status"]
SP_COLS = ["party", "video", "chunk", "promise_no", "promise", "category", "specificity", "status"]
MATCH_COLS = ["party", "video", "chunk", "promise_no", "promise", "category", "specificity",
              "verdict", "matched_promise", "matched_chunk", "top_similarity", "reason", "status"]


# ------------------------------ helpers ------------------------------
def natural_key(s):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


def strip_header(raw):
    lines = raw.splitlines()
    for i, line in enumerate(lines[:15]):
        if re.fullmatch(r"-{10,}", line.strip()):
            return "\n".join(lines[i + 1:]).strip()
    return raw.strip()


def read_chunk(path):
    for enc in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return strip_header(path.read_text(encoding=enc))
        except UnicodeError:
            continue
    return ""


def norm_cat(v):
    return CAT_LOOKUP.get(re.sub(r"[*_`]", "", str(v)).strip().lower(), "Others")


_model = None


def get_model():
    global _model
    if _model is None:
        from dotenv import load_dotenv
        import google.generativeai as genai
        load_dotenv(ROOT / ".env")
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(f"GEMINI_API_KEY not found. Check {ROOT / '.env'}")
        genai.configure(api_key=key)
        _model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"))
    return _model


def parse_json(text):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        for a, b in (("[", "]"), ("{", "}")):
            i, j = t.find(a), t.rfind(b)
            if i != -1 and j > i:
                try:
                    return json.loads(t[i:j + 1])
                except json.JSONDecodeError:
                    pass
    raise ValueError(f"Could not parse JSON from: {text[:200]!r}")


def call_json(prompt):
    last = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            cfg = {"response_mime_type": "application/json", "temperature": 0} if attempt == 1 else None
            resp = get_model().generate_content(prompt, generation_config=cfg) if cfg \
                else get_model().generate_content(prompt)
            return parse_json(resp.text)
        except Exception as e:
            last = e
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {type(e).__name__}: {str(e)[:150]}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(5 * attempt)
    raise RuntimeError(str(last))


def load_store(path, key_cols):
    """Previously saved rows grouped by key (only completed ones)."""
    store = {}
    if RESUME and path.exists():
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for row in df.to_dict("records"):
            if row.get("status") in ("ok", "skipped_by_category"):
                store.setdefault(tuple(row[k] for k in key_cols), []).append(row)
    return store


def save_store(path, store, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for rows in store.values() for r in rows]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False, encoding="utf-8-sig")


def eta(n_calls):
    mins = n_calls * (SLEEP_BETWEEN_CALLS + 3) / 60
    return f"~{mins:.0f} min" if mins < 120 else f"~{mins / 60:.1f} h"


# ------------------------------ 1 & 2: promise extraction ------------------------------
def extraction_prompt(text, party, kind):
    cats = "\n".join(f"- {c}" for c in CATEGORIES)
    if kind == "manifesto":
        intro = (f"You are reading one chunk of the {party} manifesto for the 2026 Tamil Nadu election. "
                 "The text is Tamil, scanned by OCR, and may contain spelling errors.")
        skip = "history, praise of leaders, party ideology, criticism of opponents, achievements already completed"
    else:
        intro = (f"You are reading one chunk of a {party} election rally speech (2026 Tamil Nadu election). "
                 "The text is a Tamil speech transcribed by software and may contain errors.")
        skip = ("history, praise of leaders, introducing candidates, vote appeals, criticism of opponents, "
                "jokes or anecdotes, and work the party says it has ALREADY completed")
    return f"""{intro}

Extract every SPECIFIC FORWARD-LOOKING promise or commitment: something the party says it WILL do, give, build,
provide, increase, reduce, waive or introduce. Ignore: {skip}.
Do not invent anything. If the chunk has no such promise, return an empty list.

For each promise return:
- "promise": one self-contained English sentence. Keep scheme names, amounts, numbers, beneficiaries and time frames exactly.
- "category": exactly one of:
{cats}
- "specificity": "specific" if it names a concrete scheme, amount, number or project, otherwise "general".

Return ONLY a JSON list, e.g. [{{"promise": "...", "category": "Education", "specificity": "specific"}}]

Text:
{text}
"""


def extract_items(text, party, kind):
    data = call_json(extraction_prompt(text, party, kind))
    if isinstance(data, dict):
        data = data.get("promises") or data.get("items") or []
    out = []
    for d in data or []:
        p = str(d.get("promise", "")).strip() if isinstance(d, dict) else ""
        if p:
            spec = "specific" if str(d.get("specificity", "")).lower().startswith("spec") else "general"
            out.append({"promise": p, "category": norm_cat(d.get("category")), "specificity": spec})
    return out


def find_manifesto_chunks(party):
    for cand in (MANIFESTO_CHUNKS_BASE / party, MANIFESTO_CHUNKS_BASE / f"{party}_manifesto",
                 MANIFESTO_CHUNKS_BASE / "manifesto" / party):
        if cand.is_dir() and list(cand.glob("*.txt")):
            return sorted(cand.glob("*.txt"), key=lambda p: natural_key(p.name))
    pat = re.compile(rf"(?<![A-Za-z]){party}(?![A-Za-z])")
    hits = [p for p in MANIFESTO_CHUNKS_BASE.rglob("*.txt") if pat.search(str(p.relative_to(MANIFESTO_CHUNKS_BASE)))]
    return sorted(hits, key=lambda p: natural_key(p.name))


def run_manifesto(parties):
    path = OUT / "manifesto_promises.csv"
    store = load_store(path, ("party", "chunk"))
    for party in parties:
        files = find_manifesto_chunks(party)
        print(f"\n=== Manifesto {party}: {len(files)} chunks "
              f"({files[0].parent if files else 'NOT FOUND - set MANIFESTO_CHUNKS_BASE'}) ===")
        todo = [f for f in files if (party, f.name) not in store]
        print(f"to process: {len(todo)}  ({eta(len(todo))})")
        for i, f in enumerate(todo, 1):
            text = read_chunk(f)
            if not text:
                continue
            print(f"[{i}/{len(todo)}] {f.name}", end=" ")
            try:
                items = extract_items(text, party, "manifesto")
            except Exception as e:
                print(f"-> FAILED ({e}); will retry next run")
                continue
            print(f"-> {len(items)} promises")
            rows = [{"party": party, "chunk": f.name, "promise_no": str(n), "promise": it["promise"],
                     "category": it["category"], "specificity": it["specificity"], "status": "ok"}
                    for n, it in enumerate(items, 1)]
            store[(party, f.name)] = rows or [{"party": party, "chunk": f.name, "promise_no": "0", "promise": "",
                                               "category": "", "specificity": "", "status": "ok"}]
            save_store(path, store, MAN_COLS)
            time.sleep(SLEEP_BETWEEN_CALLS)
    print(f"\nSaved {path}")


def classification_filter(party):
    """chunk filename -> True if it should be skipped (not promise-bearing)."""
    p = SPEECH_RESULTS_BASE / party / "gemini.csv"
    skip = {}
    if FILTER_BY_CLASSIFICATION and p.exists():
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if {"Chunk", "Primary_Category", "Secondary_Category", "Status"}.issubset(df.columns):
            for r in df.to_dict("records"):
                if (r["Status"].strip() == "success" and r["Primary_Category"].strip() in SKIP_PRIMARY
                        and (r["Secondary_Category"].strip() == "" or r["Secondary_Category"].strip() in SKIP_PRIMARY)):
                    skip[r["Chunk"].strip()] = True
    return skip


def run_speech(parties):
    path = OUT / "speech_promises.csv"
    store = load_store(path, ("party", "chunk"))
    for party in parties:
        folder = SPEECH_CHUNKS_BASE / party
        files = sorted(folder.glob("*.txt"), key=lambda p: natural_key(p.name)) if folder.is_dir() else []
        skip = classification_filter(party)
        print(f"\n=== Speeches {party}: {len(files)} chunks in {folder} ===")
        todo = []
        for f in files:
            if (party, f.name) in store:
                continue
            if skip.get(f.name):
                store[(party, f.name)] = [{"party": party, "video": video_of(f.name), "chunk": f.name,
                                           "promise_no": "0", "promise": "", "category": "", "specificity": "",
                                           "status": "skipped_by_category"}]
            else:
                todo.append(f)
        print(f"skipped by category filter: {sum(1 for f in files if skip.get(f.name))} | "
              f"to process: {len(todo)}  ({eta(len(todo))})")
        for i, f in enumerate(todo, 1):
            text = read_chunk(f)
            if not text:
                continue
            print(f"[{i}/{len(todo)}] {f.name}", end=" ")
            try:
                items = extract_items(text, party, "speech")
            except Exception as e:
                print(f"-> FAILED ({e}); will retry next run")
                continue
            print(f"-> {len(items)} promises")
            video = video_of(f.name)
            rows = [{"party": party, "video": video, "chunk": f.name, "promise_no": str(n),
                     "promise": it["promise"], "category": it["category"],
                     "specificity": it["specificity"], "status": "ok"} for n, it in enumerate(items, 1)]
            store[(party, f.name)] = rows or [{"party": party, "video": video, "chunk": f.name, "promise_no": "0",
                                               "promise": "", "category": "", "specificity": "", "status": "ok"}]
            save_store(path, store, SP_COLS)
            time.sleep(SLEEP_BETWEEN_CALLS)
        save_store(path, store, SP_COLS)
    print(f"\nSaved {path}")


def video_of(chunk_name):
    return re.sub(r"\.txt$", "", chunk_name).rsplit("_", 1)[0]


# ------------------------------ 3: matching ------------------------------
_STOP = set("the a an of to for and in on with will be is are by at from as that this their our all "
            "every each shall provide give create implement ensure per new tamil nadu government state "
            "people will also more under scheme".split())
_st_model = None
_warned = False


def tfidf_sim(a, b):
    tok = lambda s: [w for w in re.findall(r"[a-z0-9₹%]+", s.lower()) if w not in _STOP and len(w) > 1]
    docs = [tok(x) for x in a + b]
    vocab = {}
    for d in docs:
        for w in d:
            vocab.setdefault(w, len(vocab))
    M = np.zeros((len(docs), max(len(vocab), 1)), dtype=np.float32)
    for i, d in enumerate(docs):
        for w in d:
            M[i, vocab[w]] += 1
    df = (M > 0).sum(axis=0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1
    M = np.log1p(M) * idf
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return M[:len(a)] @ M[len(a):].T


def similarity(a, b):
    global _st_model, _warned
    if USE_EMBEDDINGS:
        try:
            from sentence_transformers import SentenceTransformer
            if _st_model is None:
                _st_model = SentenceTransformer(EMBEDDING_MODEL)
            ea = _st_model.encode(a, normalize_embeddings=True, show_progress_bar=False)
            eb = _st_model.encode(b, normalize_embeddings=True, show_progress_bar=False)
            return ea @ eb.T
        except ImportError:
            if not _warned:
                print("(sentence-transformers not installed -> using TF-IDF similarity. "
                      "Better matching: pip install sentence-transformers)")
                _warned = True
    return tfidf_sim(a, b)


def judge_prompt(batch):
    parts = []
    for n, item in enumerate(batch, 1):
        cands = "\n".join(f"    {chr(65 + k)}. {c}" for k, c in enumerate(item["cands"]))
        parts.append(f"SPEECH PROMISE {n} [{item['category']}]: {item['promise']}\n  Candidate manifesto promises:\n{cands}")
    body = "\n\n".join(parts)
    return f"""You compare promises made in a rally speech with the SAME party's manifesto.
For every numbered speech promise you get candidate manifesto promises (A, B, C...). Decide:

- "IN_MANIFESTO": the speech promise is the same commitment as one of the candidates: same action/scheme and the
  same key details (amount, number, beneficiaries, timeline). Different wording is fine. If the speech leaves out
  details but clearly names the same specific scheme without contradicting it, still IN_MANIFESTO.
- "RELATED": same topic or scheme, but a key detail differs (different amount, group, scale or timeline), or it is
  a clearly broader/narrower version of a candidate.
- "NEW": none of the candidates covers it, so it is not in the manifesto.

Be strict: a shared topic alone (e.g. both mention "farmers") is NOT enough for IN_MANIFESTO.

Return ONLY a JSON list with one object per speech promise:
[{{"id": 1, "verdict": "IN_MANIFESTO", "match": "A", "reason": "<max 25 words>"}}]
Use "match": null for NEW.

{body}
"""


def run_match(parties):
    man_path, sp_path = OUT / "manifesto_promises.csv", OUT / "speech_promises.csv"
    for p in (man_path, sp_path):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run the manifesto and speech steps first.")
    man = pd.read_csv(man_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sp = pd.read_csv(sp_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    man = man[(man["promise"] != "")]
    sp = sp[(sp["promise"] != "") & (sp["status"] == "ok")]

    out_path = OUT / "speech_promises_matched.csv"
    key = ("party", "video", "chunk", "promise_no")
    store = {}
    if RESUME and out_path.exists():
        old = pd.read_csv(out_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for r in old.to_dict("records"):
            if r["status"] == "ok":
                store[tuple(r[k] for k in key)] = [r]

    for party in parties:
        mp = man[man["party"] == party].reset_index(drop=True)
        todo = sp[sp["party"] == party]
        todo = todo[[tuple(r) not in store for r in todo[list(key)].itertuples(index=False, name=None)]]
        todo = todo.reset_index(drop=True)
        print(f"\n=== Matching {party}: {len(todo)} speech promises vs {len(mp)} manifesto promises "
              f"({eta(int(np.ceil(len(todo) / BATCH)))}) ===")
        if mp.empty or todo.empty:
            continue
        sims = similarity(todo["promise"].tolist(), mp["promise"].tolist())
        k = min(TOP_K, len(mp))
        topk = np.argsort(-sims, axis=1)[:, :k]

        for start in range(0, len(todo), BATCH):
            idx = list(range(start, min(start + BATCH, len(todo))))
            batch = [{"promise": todo.at[i, "promise"], "category": todo.at[i, "category"],
                      "cands": [mp.at[j, "promise"] for j in topk[i]]} for i in idx]
            print(f"  promises {idx[0] + 1}-{idx[-1] + 1}/{len(todo)}", end=" ")
            try:
                res = call_json(judge_prompt(batch))
                if isinstance(res, dict):
                    res = res.get("results") or list(res.values())
                by_id = {int(r["id"]): r for r in res if isinstance(r, dict) and "id" in r}
            except Exception as e:
                print(f"-> FAILED ({e}); will retry next run")
                continue
            done = 0
            for n, i in enumerate(idx, 1):
                r = by_id.get(n)
                verdict = str(r.get("verdict", "")).strip().upper().replace(" ", "_") if r else ""
                if verdict not in VERDICTS:
                    continue
                letter = str(r.get("match") or "").strip().upper()
                j = topk[i][ord(letter) - 65] if len(letter) == 1 and 0 <= ord(letter) - 65 < k else None
                if verdict != "NEW" and j is None:
                    verdict, j = "NEW", None
                if verdict == "NEW":
                    j = None
                src = todo.loc[i]
                row = {c: src[c] for c in ("party", "video", "chunk", "promise_no", "promise", "category", "specificity")}
                row.update({"verdict": verdict,
                            "matched_promise": mp.at[j, "promise"] if j is not None else "",
                            "matched_chunk": mp.at[j, "chunk"] if j is not None else "",
                            "top_similarity": f"{sims[i][j if j is not None else topk[i][0]]:.3f}",
                            "reason": str(r.get("reason", ""))[:300], "status": "ok"})
                store[tuple(row[c] for c in key)] = [row]
                done += 1
            print(f"-> {done}/{len(idx)} judged")
            save_store(out_path, store, MATCH_COLS)
            time.sleep(SLEEP_BETWEEN_CALLS)
    print(f"\nSaved {out_path}")


# ------------------------------ 4: report ------------------------------
def video_fixer():
    """Use the corrected locations from fix_locations.py / videos_with_locations.json if available."""
    try:
        from fix_locations import find_json, load_locations, split_video
        locs, _ = load_locations(find_json())

        def fix(v):
            parts = split_video(v)
            if parts and parts[3] in locs:
                return f"{parts[0]}_{parts[1]}_{locs[parts[3]]}_{parts[3]}"
            return v
        return fix
    except (Exception, SystemExit):
        return lambda v: v


def run_report(parties):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m_path, mp_path = OUT / "speech_promises_matched.csv", OUT / "manifesto_promises.csv"
    if not m_path.exists():
        raise SystemExit("Run the match step first.")
    m = pd.read_csv(m_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    m = m[(m["status"] == "ok") & (m["party"].isin(parties))].copy()
    man = pd.read_csv(mp_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    man = man[(man["promise"] != "") & (man["party"].isin(parties))]
    if m.empty:
        raise SystemExit("No matched promises yet.")
    fix = video_fixer()
    m["video"] = m["video"].map(fix)
    parties = [p for p in parties if p in set(m["party"])]
    plt.style.use("dark_background")
    plt.rcParams["axes.facecolor"] = SURFACE
    charts = OUT / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    def pct_table(df, by):
        c = df.groupby(by + ["verdict"]).size().unstack(fill_value=0).reindex(columns=VERDICTS, fill_value=0)
        c["total"] = c.sum(axis=1)
        for v in VERDICTS:
            c[f"{v}_%"] = (c[v] / c["total"] * 100).round(1)
        return c.reset_index()

    # --- tables ---
    by_party = pct_table(m, ["party"]).set_index("party").reindex(parties).reset_index()
    by_video = pct_table(m, ["party", "video"]).sort_values(["party", "IN_MANIFESTO_%"], ascending=[True, False])
    by_cat = pct_table(m, ["party", "category"])

    # manifesto coverage: how many manifesto promises were repeated in speeches
    man = man.rename(columns={"promise": "manifesto_promise", "chunk": "manifesto_chunk"})
    hits = m[m["verdict"] != "NEW"]
    cov_rows = []
    for _, r in man.iterrows():
        h = hits[(hits["party"] == r["party"]) & (hits["matched_promise"] == r["manifesto_promise"])]
        cov_rows.append({"party": r["party"], "manifesto_promise": r["manifesto_promise"],
                         "category": r["category"], "times_in_manifesto_verdict": int((h["verdict"] == "IN_MANIFESTO").sum()),
                         "times_related_verdict": int((h["verdict"] == "RELATED").sum()),
                         "speeches": h["video"].nunique()})
    cov = pd.DataFrame(cov_rows)
    cov["repeated_exact"] = cov["times_in_manifesto_verdict"] > 0
    cov["repeated_any"] = (cov["times_in_manifesto_verdict"] + cov["times_related_verdict"]) > 0
    cov_party = cov.groupby("party").agg(manifesto_promises=("manifesto_promise", "size"),
                                         repeated_exact=("repeated_exact", "sum"),
                                         repeated_any=("repeated_any", "sum")).reindex(parties).reset_index()
    cov_party["repeated_exact_%"] = (cov_party["repeated_exact"] / cov_party["manifesto_promises"] * 100).round(1)
    cov_party["repeated_any_%"] = (cov_party["repeated_any"] / cov_party["manifesto_promises"] * 100).round(1)
    summary = by_party.merge(cov_party, on="party")
    summary.insert(1, "speeches", summary["party"].map(m.groupby("party")["video"].nunique()))

    with pd.ExcelWriter(OUT / "promise_analysis.xlsx", engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Party_Summary", index=False)
        by_video.to_excel(xw, sheet_name="Per_Speech", index=False)
        by_cat.to_excel(xw, sheet_name="By_Category", index=False)
        cov.sort_values(["party", "times_in_manifesto_verdict"], ascending=[True, False]).to_excel(
            xw, sheet_name="Manifesto_Coverage", index=False)
        m.to_excel(xw, sheet_name="All_Matches", index=False)
    summary.to_csv(OUT / "party_summary.csv", index=False)
    by_video.to_csv(OUT / "per_speech.csv", index=False)

    def save(fig, name):
        fig.patch.set_facecolor(BG)
        fig.tight_layout()
        fig.savefig(charts / name, dpi=150, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print("  saved", charts / name)

    pretty = lambda c: c.replace("Social justice and Deprived communities", "Social justice") \
                        .replace("Agriculture and Farmers", "Agriculture").replace("Employment and Jobs", "Employment") \
                        .replace("Women's Development", "Women").replace("Religion and Culture", "Religion")

    # 1) verdict split per party (100% stacked, counts shown)
    fig, ax = plt.subplots(figsize=(9, 6))
    bottom = np.zeros(len(by_party))
    for v in VERDICTS:
        vals = by_party[f"{v}_%"].values
        bars = ax.bar(by_party["party"], vals, bottom=bottom, color=COLORS[v], label=v.replace("_", " ").title(), alpha=0.9)
        for b, val, cnt in zip(bars, vals, by_party[v]):
            if val >= 4:
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + val / 2, f"{val:.0f}%\n(n={cnt})",
                        ha="center", va="center", fontsize=9, color="black", weight="bold")
        bottom += vals
    ax.set_ylabel("% of promises made in speeches")
    ax.set_title("Which speech promises are in the manifesto?", fontsize=14, weight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3)
    save(fig, "1_promises_by_party.png")

    # 2) per speech
    v = by_video.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.38 * len(v))))
    left = np.zeros(len(v))
    for ver in VERDICTS:
        ax.barh(range(len(v)), v[f"{ver}_%"], left=left, color=COLORS[ver], label=ver.replace("_", " ").title(), alpha=0.9)
        left += v[f"{ver}_%"].values
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels([f"{r.video}  (n={r.total})" for r in v.itertuples()], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% of that speech's promises")
    ax.set_title("Per speech: promises that are in the manifesto", fontsize=13, weight="bold")
    ax.legend(loc="lower right")
    save(fig, "2_promises_per_speech.png")

    # 3) per category (one panel per party)
    fig, axes = plt.subplots(len(parties), 1, figsize=(13, 4.8 * len(parties)))
    for ax, party in zip(np.atleast_1d(axes), parties):
        d = by_cat[by_cat["party"] == party].sort_values("total", ascending=False)
        bottom = np.zeros(len(d))
        for ver in VERDICTS:
            ax.bar(range(len(d)), d[f"{ver}_%"], bottom=bottom, color=COLORS[ver], alpha=0.9,
                   label=ver.replace("_", " ").title())
            bottom += d[f"{ver}_%"].values
        for x, n in enumerate(d["total"]):
            ax.text(x, 101, f"n={n}", ha="center", fontsize=8)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([pretty(c) for c in d["category"]], rotation=25, ha="right")
        ax.set_ylim(0, 110)
        ax.set_ylabel("%")
        ax.set_title(f"{party}: speech promises by topic", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
    save(fig, "3_promises_by_category.png")

    # 4) how much of each manifesto was repeated in speeches
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(cov_party))
    for off, col, lab, colr in ((-0.2, "repeated_exact_%", "Repeated (same promise)", COLORS["IN_MANIFESTO"]),
                                (0.2, "repeated_any_%", "Repeated or related", COLORS["RELATED"])):
        bars = ax.bar(x + off, cov_party[col], 0.4, label=lab, color=colr, alpha=0.9)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.party}\n({r.manifesto_promises} promises)" for r in cov_party.itertuples()])
    ax.set_ylabel("% of manifesto promises")
    ax.set_title("How much of the manifesto was spoken about in rallies?", fontsize=13, weight="bold")
    ax.legend()
    save(fig, "4_manifesto_promises_repeated.png")

    print("\n=== Party summary ===")
    print(summary[["party", "speeches", "total", "IN_MANIFESTO_%", "RELATED_%", "NEW_%",
                   "manifesto_promises", "repeated_exact_%", "repeated_any_%"]].to_string(index=False))
    print(f"\nTables: {OUT / 'promise_analysis.xlsx'}")


# ------------------------------ main ------------------------------
def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    step = args[0].lower()
    asked = [a.upper() for a in args[1:]]
    bad = [a for a in asked if a not in PARTIES]
    if bad:
        raise SystemExit(f"Unknown party {bad}. Use one or more of: {', '.join(PARTIES)}")
    parties = asked or PARTIES
    OUT.mkdir(parents=True, exist_ok=True)
    steps = {"manifesto": run_manifesto, "speech": run_speech, "match": run_match, "report": run_report}
    if step == "all":
        for s in ("manifesto", "speech", "match", "report"):
            steps[s](parties)
    elif step in steps:
        steps[step](parties)
    else:
        print(f"Unknown step '{step}'.\n{__doc__}")


if __name__ == "__main__":
    main()