"""
Manifesto vs speech comparison (Gemini only), same logic/style as your Kaggle notebook:
  * reads *_summary.csv files (a 'category' column + a 'percent' column)
  * speech average per party = simple mean of that party's speeches (as in the notebook)
  * grouped dark-theme bar charts

Looks in:
  output/speech_summary/**/*.csv      (one summary csv per speech/video)
  output/manifesto_summary/**/*.csv   (one summary csv per party manifesto)
The party is detected from the folder/file name (AIADMK / DMK / TVK).
If auto-detection is wrong, fill SPEECH_FILES / MANIFESTO_FILES like in the notebook.

Run from project root:  python compare_manifesto_vs_speech.py
Outputs -> output/comparison/
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------ CONFIG ------------------------------
SPEECH_DIR = os.path.join("output", "speech_summary")
MANIFESTO_DIR = os.path.join("output", "manifesto_summary")
OUT_DIR = os.path.join("output", "comparison")
PARTIES = ["AIADMK", "DMK", "TVK"]
MODEL_HINT = "gemini"     # if any path contains this, only those files are used

# Optional manual override, same format as the notebook: {'name': ('PARTY', 'path.csv')}
SPEECH_FILES = {}
MANIFESTO_FILES = {}

# Not policy topics -> excluded from the manifesto-match score (still shown in raw %)
NON_POLICY = {"none", "nan", "unclassified", "already executed promises", "other", "others", "content about the opposition party",
              "opposition party", "general", "greetings", "irrelevant"}
# Rename categories so speech and manifesto labels line up, e.g.
CATEGORY_ALIASES = {
    # "agriculture and farmers": "agriculture",
}

BG, SURFACE = "#0a0a0f", "#12121a"
COLORS = {"AIADMK": "#00D4A0", "DMK": "#FF6B35", "TVK": "#FFD23F"}
# --------------------------------------------------------------------


def norm_cat(c):
    c = str(c).strip().lower()
    return CATEGORY_ALIASES.get(c, c)


def load_summary_csv(path):
    """Same as notebook: returns {category: percentage}."""
    df = pd.read_csv(path)
    cat_col = next((c for c in df.columns if "categ" in c.lower()), None)
    pct_col = next((c for c in df.columns if "percent" in c.lower()), None)
    if cat_col is None or pct_col is None:
        raise ValueError(f"needs 'category' and 'percent' columns, found {list(df.columns)}")
    out = {}
    for _, r in df.iterrows():
        try:
            pct = float(str(r[pct_col]).replace("%", "").strip())
        except ValueError:
            pct = 0.0
        cat = norm_cat(r[cat_col])
        out[cat] = out.get(cat, 0.0) + pct
    return out


def _num(v):
    try:
        return float(str(v).replace("%", "").strip())
    except ValueError:
        return 0.0


def load_table(path):
    """Read a summary csv in either layout. Returns a list of dicts:
       {label, party, data={category: pct}, per_item}
       WIDE : columns like 'Governance (%)' -> one row per video / per party
       LONG : 'category' + 'percent' columns -> the whole file is one item"""
    df = pd.read_csv(path)
    pct_cols = [c for c in df.columns if str(c).strip().endswith("(%)")]
    if pct_cols:
        id_col = next((c for c in df.columns if str(c).lower() in ("video", "speech", "name", "file")), None)
        party_col = next((c for c in df.columns if str(c).lower() == "party"), None)
        out = []
        for i, r in df.iterrows():
            d = {}
            for c in pct_cols:
                cat = norm_cat(re.sub(r"\s*\(%\)\s*$", "", str(c)))
                d[cat] = d.get(cat, 0.0) + _num(r[c])
            label = str(r[id_col]) if id_col else (str(r[party_col]) if party_col else f"row{i}")
            party = str(r[party_col]).strip().upper() if party_col else None
            out.append({"label": label, "party": party, "data": d, "per_item": id_col is not None})
        return out
    return [{"label": os.path.splitext(os.path.basename(path))[0], "party": None,
             "data": load_summary_csv(path), "per_item": True}]


def detect_party(path):
    for part in reversed(path.replace("\\", "/").split("/")):
        for p in PARTIES:
            if re.search(rf"(?<![A-Za-z]){p}(?![A-Za-z])", part, re.I):
                return p
    return None


def discover(folder):
    files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if any(MODEL_HINT in f.lower() for f in files):
        files = [f for f in files if MODEL_HINT in f.lower()]
    found = {}
    for f in sorted(files):
        party = detect_party(os.path.relpath(f, folder)) or detect_party(f)
        found[os.path.relpath(f, folder)] = (party, f)
    return found


def policy(d):
    return {k: v for k, v in d.items() if k not in NON_POLICY}


def match(speech, manifesto):
    sp, mp = policy(speech), policy(manifesto)
    cats = sorted(set(sp) | set(mp))
    total_speech = sum(speech.values())
    policy_share = sum(sp.values()) / total_speech * 100 if total_speech else 0
    if not sp or not mp or sum(sp.values()) == 0 or sum(mp.values()) == 0:
        return 0.0, 0.0, 0.0, policy_share
    p = np.array([sp.get(c, 0) for c in cats], float); p /= p.sum()
    q = np.array([mp.get(c, 0) for c in cats], float); q /= q.sum()
    overlap = np.minimum(p, q).sum() * 100
    cosine = float(p @ q / (np.linalg.norm(p) * np.linalg.norm(q))) * 100
    promised = [c for c, v in mp.items() if v > 0]      # only topics the manifesto actually has
    covered = [c for c in promised if sp.get(c, 0) > 0]
    coverage = len(covered) / len(promised) * 100 if promised else 0.0
    return overlap, cosine, coverage, policy_share


def pretty(c):
    return (c.replace("content about the opposition party", "opposition")
             .replace("social justice and deprived communities", "social justice")
             .replace("agriculture and farmers", "agriculture")
             .replace("employment and jobs", "employment")
             .replace("women's development", "women")
             .replace("religion and culture", "religion")
             .title())


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", alpha=0.15, color="white")
    for s in ax.spines.values():
        s.set_color("#333")


def grouped(ax, cats, series, colors, label_fs=7):
    """series: {name: {cat: value}}"""
    n, x = len(series), np.arange(len(cats))
    w = 0.8 / n
    for i, (name, vals) in enumerate(series.items()):
        v = [vals.get(c, 0) for c in cats]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, v, w, label=name, color=colors[i], alpha=0.8)
        for b in bars:
            if b.get_height() > 0:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                        f"{b.get_height():.1f}", ha="center", fontsize=label_fs, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(c) for c in cats], rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("% of content")
    ax.legend()
    style_ax(ax)


def save(fig, name, title=None):
    fig.patch.set_facecolor(BG)
    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", color="white")
    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("  saved", path)


def main():
    plt.style.use("dark_background")
    os.makedirs(OUT_DIR, exist_ok=True)

    sp_src = SPEECH_FILES or discover(SPEECH_DIR)
    mf_src = MANIFESTO_FILES or discover(MANIFESTO_DIR)

    speeches = {}      # "video [PARTY]" -> (party, {cat: pct})
    for name, (party, path) in sp_src.items():
        try:
            entries = load_table(path)
        except Exception as e:
            print(f"✗ speech file {name} — {e}")
            continue
        n = 0
        for e in entries:
            if not e["per_item"]:
                continue
            p = e["party"] if e["party"] in PARTIES else (party or detect_party(e["label"]))
            if p is None:
                continue
            speeches[f"{e['label']} [{p}]"] = (p, e["data"])
            n += 1
        print(f"✓ speech file {name}: {n} speeches")

    manifestos = {}    # party -> {cat: pct}
    for name, (party, path) in mf_src.items():
        try:
            entries = load_table(path)
        except Exception as e:
            print(f"✗ manifesto file {name} — {e}")
            continue
        for e in entries:
            p = e["party"] if e["party"] in PARTIES else (party or detect_party(e["label"]))
            if p is None:
                continue
            manifestos.setdefault(p, e["data"])
            print(f"✓ manifesto {p}  ({name})")

    print(f"\nSpeeches per party: { {p: sum(v[0]==p for v in speeches.values()) for p in PARTIES} }")
    print(f"Manifestos found for: {sorted(manifestos)}")

    parties = [p for p in PARTIES if p in manifestos and any(v[0] == p for v in speeches.values())]
    if not parties:
        print("No party has both a manifesto and speeches. Check folders / SPEECH_FILES.")
        return
    cats_all = sorted({c for _, d in speeches.values() for c in d} |
                      {c for d in manifestos.values() for c in d})
    pol_cats = [c for c in cats_all if c not in NON_POLICY]

    # ---- per-party average speech (simple mean over speeches, as in the notebook) ----
    speech_avg = {}
    for p in parties:
        df = pd.DataFrame([d for (pp, d) in speeches.values() if pp == p]).fillna(0)
        speech_avg[p] = df.mean().to_dict()

    # ---- per-speech match ----
    rows = []
    for name, (p, d) in speeches.items():
        if p not in manifestos:
            continue
        ov, cs, cov, pol = match(d, manifestos[p])
        top = max(policy(d), key=policy(d).get) if policy(d) else "-"
        rows.append({"party": p, "speech": name.split(" [")[0], "match_overlap_%": round(ov, 2),
                     "match_cosine_%": round(cs, 2), "manifesto_coverage_%": round(cov, 2),
                     "policy_content_%": round(pol, 2), "top_speech_topic": top})
    per_speech = pd.DataFrame(rows).sort_values(["party", "match_overlap_%"], ascending=[True, False])

    # ---- per-party overall ----
    prow = []
    for p in parties:
        ov, cs, cov, pol = match(speech_avg[p], manifestos[p])
        prow.append({"party": p, "speeches": sum(v[0] == p for v in speeches.values()),
                     "match_overlap_%": round(ov, 2), "match_cosine_%": round(cs, 2),
                     "manifesto_coverage_%": round(cov, 2), "policy_content_%": round(pol, 2),
                     "opposition_talk_%": round(speech_avg[p].get("content about the opposition party", 0), 2)})
    per_party = pd.DataFrame(prow)

    # ---- category table (each side re-normalised over policy topics to 100%) ----
    def renorm(d):
        s = sum(policy(d).values())
        return {c: v / s * 100 for c, v in policy(d).items()} if s else {}
    crow = []
    for p in parties:
        ms, ss = renorm(manifestos[p]), renorm(speech_avg[p])
        for c in pol_cats:
            crow.append({"party": p, "category": c, "manifesto_%": round(ms.get(c, 0), 2),
                         "speech_%": round(ss.get(c, 0), 2),
                         "gap_(speech-manifesto)": round(ss.get(c, 0) - ms.get(c, 0), 2)})
    cat_table = pd.DataFrame(crow)

    for df, n in ((per_speech, "per_speech_match"), (per_party, "party_summary"), (cat_table, "category_shares")):
        df.to_csv(os.path.join(OUT_DIR, n + ".csv"), index=False)
    with pd.ExcelWriter(os.path.join(OUT_DIR, "comparison.xlsx"), engine="openpyxl") as xw:
        per_party.to_excel(xw, sheet_name="Party_Summary", index=False)
        per_speech.to_excel(xw, sheet_name="Per_Speech", index=False)
        cat_table.to_excel(xw, sheet_name="Category_Shares", index=False)

    # ================= CHARTS =================
    print("\nCharts")
    # 1) per party: manifesto vs speech (raw % like the notebook)
    fig, axes = plt.subplots(len(parties), 1, figsize=(15, 5.5 * len(parties)))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, parties):
        grouped(ax, cats_all, {f"{p} manifesto": manifestos[p], f"{p} speeches (avg)": speech_avg[p]},
                ["#8888aa", COLORS[p]])
        ax.set_title(f"{p}: manifesto vs speeches")
    save(fig, "1_manifesto_vs_speech_per_party.png", "Manifesto promises vs what was said in speeches")

    # 2) parties compared: speeches | manifestos
    fig, axes = plt.subplots(2, 1, figsize=(15, 11))
    grouped(axes[0], cats_all, {p: speech_avg[p] for p in parties}, [COLORS[p] for p in parties])
    axes[0].set_title("Speeches (average per party)")
    grouped(axes[1], cats_all, {p: manifestos[p] for p in parties}, [COLORS[p] for p in parties])
    axes[1].set_title("Manifestos")
    save(fig, "2_parties_compared.png", "Parties compared: topic share in speeches and manifestos")

    # 3) overall match per party
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["match_overlap_%", "manifesto_coverage_%", "policy_content_%"]
    grouped(ax, metrics,
            {r["party"]: {m: r[m] for m in metrics} for _, r in per_party.iterrows()},
            [COLORS[p] for p in per_party["party"]], label_fs=9)
    ax.set_xticklabels(["Topic-mix match with manifesto", "Manifesto topics covered", "Speech that is policy content"],
                       rotation=0, ha="center")
    ax.set_ylabel("%")
    save(fig, "3_overall_match_by_party.png", "How closely do speeches follow the manifesto?")

    # 4) per-speech match
    v = per_speech.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(v))))
    ax.barh(range(len(v)), v["match_overlap_%"], color=[COLORS[p] for p in v["party"]], alpha=0.85)
    for i, val in enumerate(v["match_overlap_%"]):
        ax.text(val + 0.5, i, f"{val:.1f}%", va="center", fontsize=7, color="white")
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels(v["speech"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Match with own party's manifesto (%)")
    ax.set_facecolor(SURFACE)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[p]) for p in parties], parties, loc="lower right")
    save(fig, "4_per_speech_match.png", "Per speech: match with manifesto")

    # 5) gap heatmap
    g = cat_table.pivot(index="category", columns="party", values="gap_(speech-manifesto)").reindex(columns=parties).fillna(0)
    fig, ax = plt.subplots(figsize=(7.5, max(4, 0.45 * len(g))))
    lim = max(abs(g.values).max(), 1)
    im = ax.imshow(g.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(g.columns)), g.columns)
    ax.set_yticks(range(len(g.index)), [pretty(c) for c in g.index])
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            ax.text(j, i, f"{g.values[i, j]:+.1f}", ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im)
    ax.set_title("Speech % minus manifesto % (blue = talked about more than promised)", fontsize=9)
    save(fig, "5_gap_heatmap.png")

    print("\n=== Party summary ===")
    print(per_party.to_string(index=False))
    print(f"\nAll outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()