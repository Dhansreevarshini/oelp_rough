
import sys
import pandas as pd

# reuse the paths and helpers from your pipeline file
from promise_level_classification import (
    OUT, SPEECH_CHUNKS_BASE, read_chunk, find_manifesto_chunks, PARTIES,
)

MAX_CELL = 30000   # Excel cells hold max 32,767 characters


def main():
    args = [a.upper() for a in sys.argv[1:]]
    parties = [a for a in args if a in PARTIES]
    verdicts = [a for a in args if a in ("IN_MANIFESTO", "RELATED", "NEW")]
    if not parties:
        raise SystemExit("Usage: python make_verification_csv.py AIADMK [IN_MANIFESTO] [RELATED] [NEW]")
    party = parties[0]

    path = OUT / "speech_promises_matched.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run the match step first.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df = df[(df["party"] == party) & (df["status"] == "ok")].copy()
    if verdicts:
        df = df[df["verdict"].isin(verdicts)]
    if df.empty:
        raise SystemExit("No rows to verify with those filters.")

    # find every chunk file once, then read the text
    speech_dir = SPEECH_CHUNKS_BASE / party
    man_files = {p.name: p for p in find_manifesto_chunks(party)}
    cache = {}

    def speech_text(name):
        key = ("s", name)
        if key not in cache:
            p = speech_dir / name
            cache[key] = read_chunk(p)[:MAX_CELL] if p.exists() else "(speech chunk file not found)"
        return cache[key]

    def man_text(name):
        if not name:
            return ""
        key = ("m", name)
        if key not in cache:
            p = man_files.get(name)
            cache[key] = read_chunk(p)[:MAX_CELL] if p else "(manifesto chunk file not found)"
        return cache[key]

    df["speech_chunk_text_TAMIL"] = df["chunk"].map(speech_text)
    df["manifesto_chunk_text_TAMIL"] = df["matched_chunk"].map(man_text)
    df["my_verdict"] = ""
    df["my_notes"] = ""

    order = {"IN_MANIFESTO": 0, "RELATED": 1, "NEW": 2}
    df["_o"] = df["verdict"].map(order)
    df["_s"] = pd.to_numeric(df["top_similarity"], errors="coerce")
    df = df.sort_values(["_o", "_s"], ascending=[True, False])

    cols = ["verdict", "my_verdict", "my_notes",
            "promise", "matched_promise", "reason", "top_similarity",
            "speech_chunk_text_TAMIL", "manifesto_chunk_text_TAMIL",
            "video", "chunk", "promise_no", "matched_chunk", "category", "specificity"]
    out = OUT / f"verification_{party}.csv"
    df[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {out}  ({len(df)} rows)")
    print(df["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()