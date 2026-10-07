# summarize_tvk.py
# Builds output/speech_summary/TVK/gemini_per_video_percentages.csv
# from output/speech_results/TVK/gemini.csv   (TVK only, nothing else is touched)
#
# Run from the project root:  python summarize_tvk.py

import pandas as pd
from pathlib import Path

PARTY = "TVK"
MODEL = "gemini"
IN_CSV = Path("output/speech_results") / PARTY / f"{MODEL}.csv"
OUT_CSV = Path("output/speech_summary") / PARTY / f"{MODEL}_per_video_percentages.csv"

# Used only if the csv has no header row
COLS = ["party", "file", "promise_type", "category", "status", "error"]


def load(path):
    # 1) try with a header row
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")
    lower = {c.lower().strip(): c for c in raw.columns}
    cat_name = lower.get("category") or lower.get("main_category")

    if "file" in lower and cat_name:
        df = pd.DataFrame({
            "file": raw[lower["file"]],
            "category": raw[cat_name],
            "status": raw[lower["status"]] if "status" in lower else "success",
        })
    else:
        # 2) no header: same positional layout as summarize_percentages.py
        df = pd.read_csv(path, header=None, names=COLS, dtype=str,
                         keep_default_na=False, on_bad_lines="skip")
        df = df[["file", "category", "status"]]

    df["file"] = df["file"].str.strip()
    df = df[df["file"].str.endswith(".txt")]            # drops header/garbage rows
    total_rows = len(df)
    df = df[df["status"].str.strip() == "success"]      # only successful calls
    failed = total_rows - len(df)
    before = len(df)
    df = df.drop_duplicates(subset="file", keep="last") # a chunk classified twice counts once
    dupes = before - len(df)
    df["category"] = df["category"].str.strip().replace("", "Unclassified")
    return df, failed, dupes


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Not found: {IN_CSV}  (run from C:\\manifesto_analysis)")

    df, failed, dupes = load(IN_CSV)
    if df.empty:
        raise SystemExit("No successful rows found - check the csv columns.")

    # DMK_kanimozhi_x_qBXEjDhB4ng_0052.txt -> DMK_kanimozhi_x_qBXEjDhB4ng
    df["video"] = df["file"].str.rsplit("_", n=1).str[0]

    counts = df.groupby(["video", "category"]).size().unstack(fill_value=0)
    total = counts.sum(axis=1)
    pct = counts.div(total, axis=0).mul(100).round(2)
    pct.columns = [f"{c} (%)" for c in pct.columns]
    pct.insert(0, "total_chunks", total)
    result = pct.reset_index()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_CSV, index=False)

    print(f"Read            : {IN_CSV}")
    print(f"Chunks used     : {len(df)}")
    print(f"Failed (skipped): {failed}")
    print(f"Duplicates removed: {dupes}")
    print(f"Videos found    : {len(result)}")
    print(result[["video", "total_chunks"]].to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()