# summarize_percentages.py
import pandas as pd
from pathlib import Path

SPEECH_DIR    = Path("output/speech_results")      # AIADMK/gemini.csv, DMK/gemini.csv ...
MANIFESTO_DIR = Path("output/manifesto_results")   # adjust if the layout differs
OUT_SPEECH    = Path("output/speech_summary")
OUT_MANIFESTO = Path("output/manifesto_summary")

COLS = ["party", "file", "promise_type", "category", "status", "error"]


def load(path):
    df = pd.read_csv(path, header=None, names=COLS, dtype=str,
                     keep_default_na=False, on_bad_lines="skip")
    df = df[df["file"].str.endswith(".txt")]        # drops a header row, if any
    df = df[df["status"].str.strip() == "success"]  # only successful calls
    df["category"] = df["category"].str.strip().replace("", "Unclassified")
    return df


def percentages(df, group_cols):
    counts = df.groupby(group_cols + ["category"]).size().unstack(fill_value=0)
    total = counts.sum(axis=1)
    pct = counts.div(total, axis=0).mul(100).round(2)
    pct.columns = [f"{c} (%)" for c in pct.columns]
    pct.insert(0, "total_chunks", total)
    return pct.reset_index()


def run(root, out_root, per_video):
    for csv_path in sorted(root.rglob("*.csv")):
        df = load(csv_path)
        if df.empty:
            continue
        # party comes from the party column, falling back to the folder name
        df["party"] = df["party"].replace("", csv_path.parent.name)
        model = csv_path.stem                         # gemini / groq
        for party, pdf in df.groupby("party"):
            out_dir = out_root / party
            out_dir.mkdir(parents=True, exist_ok=True)
            if per_video:
                # DMK_kanimozhi_thoothukudi_qBXEjDhB4ng_0052.txt -> DMK_kanimozhi_thoothukudi_qBXEjDhB4ng
                pdf = pdf.assign(video=pdf["file"].str.rsplit("_", n=1).str[0])
                result = percentages(pdf, ["video"])
                name = f"{model}_per_video_percentages.csv"
            else:
                result = percentages(pdf.assign(party=party), ["party"])
                name = f"{model}_manifesto_percentages.csv"
            result.to_csv(out_dir / name, index=False)
            print("saved", out_dir / name)


run(SPEECH_DIR, OUT_SPEECH, per_video=True)
run(MANIFESTO_DIR, OUT_MANIFESTO, per_video=False)