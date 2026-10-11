# summarize_percentages_fixed.py
# Replacement for summarize_percentages.py - handles BOTH:
#   speeches   : output/speech_results/<PARTY>/<model>.csv     -> output/speech_summary/<PARTY>/<model>_per_video_percentages.csv
#   manifestos : output/manifesto_results/**/<model>.csv       -> output/manifesto_summary/<PARTY>/<model>_manifesto_percentages.csv
#
# What is fixed: the old script always read the 4th column as "category". In the new speech
# files (Party, Chunk, Primary_Category, Secondary_Category, Status, Error) the 4th column is the
# SECONDARY category, which is why most chunks became "Unclassified".
#
# How the category column is chosen for each file (it prints the choice):
#   1. file has a header row and a "Primary_Category" column  -> that column
#   2. file has a header row and a "category" column          -> that column
#   3. no header row                                          -> your old layout
#                                                                party,file,promise_type,category,status,error
#
# SECONDARY_WEIGHT = 0   -> primary only (default)
# SECONDARY_WEIGHT = 0.5 -> a Secondary_Category (if the file has one) also counts at half weight
#
# Run from the project root:  python summarize_percentages_fixed.py

from pathlib import Path

import pandas as pd

SPEECH_DIR = Path("output/speech_results")
MANIFESTO_DIR = Path("output/manifesto_results")
OUT_SPEECH = Path("output/speech_summary")
OUT_MANIFESTO = Path("output/manifesto_summary")

OLD_COLS = ["party", "file", "promise_type", "category", "status", "error"]
SECONDARY_WEIGHT = 0.2


def load(path):
    """Return (df with columns party, file, category, secondary, status), description of columns used."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", on_bad_lines="skip")
    low = {str(c).strip().lower(): c for c in raw.columns}
    file_col = low.get("chunk") or low.get("file")
    cat_col = low.get("primary_category") or low.get("main_category") or low.get("category")

    if file_col and cat_col:                                   # file has a header row
        df = pd.DataFrame({
            "party": raw[low["party"]] if "party" in low else "",
            "file": raw[file_col],
            "category": raw[cat_col],
            "secondary": raw[low["secondary_category"]] if "secondary_category" in low else "",
            "status": raw[low["status"]] if "status" in low else "success",
        })
        used = f"header row, category column = '{cat_col}'"
    else:                                                      # no header: old layout
        raw = pd.read_csv(path, header=None, names=OLD_COLS, dtype=str, keep_default_na=False,
                          encoding="utf-8-sig", on_bad_lines="skip")
        df = raw[["party", "file", "category", "status"]].copy()
        df["secondary"] = ""
        used = "no header row, old layout (4th column = category)"

    df["file"] = df["file"].str.strip()
    df = df[df["file"] != ""]
    df = df[df["status"].str.strip() == "success"]
    df = df.drop_duplicates(subset=["party", "file"], keep="last")
    df["category"] = df["category"].str.strip().replace("", "Unclassified")
    df["secondary"] = df["secondary"].str.strip()
    return df, used


def percentages(df, group_col):
    long = pd.DataFrame({"g": df[group_col], "category": df["category"], "w": 1.0})
    if SECONDARY_WEIGHT > 0:
        sec = df[df["secondary"] != ""]
        long = pd.concat([long, pd.DataFrame({"g": sec[group_col], "category": sec["secondary"],
                                              "w": SECONDARY_WEIGHT})])
    counts = long.groupby(["g", "category"])["w"].sum().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(2)
    pct.columns = [f"{c} (%)" for c in pct.columns]
    pct.insert(0, "total_chunks", df.groupby(group_col).size())
    out = pct.reset_index().rename(columns={"g": group_col})
    return out


def run(root, out_root, per_video):
    for csv_path in sorted(root.rglob("*.csv")):
        df, used = load(csv_path)
        if df.empty:
            continue
        df["party"] = df["party"].replace("", csv_path.parent.name)
        model = csv_path.stem
        for party, pdf in df.groupby("party"):
            out_dir = out_root / party
            out_dir.mkdir(parents=True, exist_ok=True)
            if per_video:
                pdf = pdf.assign(video=pdf["file"].str.rsplit("_", n=1).str[0])
                result = percentages(pdf, "video")
                name = f"{model}_per_video_percentages.csv"
            else:
                result = percentages(pdf.assign(party=party), "party")
                name = f"{model}_manifesto_percentages.csv"
            result.to_csv(out_dir / name, index=False)
            print(f"saved {out_dir / name}  ({len(pdf)} chunks)  [{used}]")


print("--- SPEECHES (per video) ---")
run(SPEECH_DIR, OUT_SPEECH, per_video=True)
print("\n--- MANIFESTOS (per party) ---")
run(MANIFESTO_DIR, OUT_MANIFESTO, per_video=False)