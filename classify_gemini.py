import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ============================================================
# PATHS  (script lives in manifesto_analysis/, next to make_chunks_manifesto.py)
#
# manifesto_analysis/
# ├── chunks/
# │   ├── AIADMK/*.txt
# │   ├── DMK/*.txt
# │   └── TVK/*.txt
# ├── output/
# │   └── manifesto_results/<PARTY>/gemini.csv   <- created by this script
# ├── .env
# └── classify_manifesto.py
# ============================================================

ROOT = Path(__file__).resolve().parent                 # manifesto_analysis/
CHUNKS_BASE = ROOT / "chunks"
RESULTS_BASE = ROOT / "output" / "manifesto_results"

MANIFESTOS = {
    "AIADMK": CHUNKS_BASE / "AIADMK",
    "DMK": CHUNKS_BASE / "DMK",
    "TVK": CHUNKS_BASE / "TVK",
}

# Output: output/manifesto_results/<PARTY>/gemini.csv
OUTPUT_NAME = "gemini.csv"

# ============================================================
# API SETUP
# ============================================================

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError(f"GEMINI_API_KEY not found. Check {ROOT / '.env'}")

genai.configure(api_key=API_KEY)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
model = genai.GenerativeModel(MODEL_NAME)

SLEEP_BETWEEN_CALLS = 8      # seconds (rate limiting)
MAX_ATTEMPTS = 3

# ============================================================
# CATEGORIES (14)  -- unchanged
# ============================================================

CATEGORIES = [
    "Agriculture and Farmers",
    "Employment and Jobs",
    "Women's Development",
    "Education",
    "Health",
    "Governance",
    "Infrastructure",
    "Religion and Culture",
    "Social justice and Deprived communities",
    "Law and order",
    "Welfare and Funds",
    "Divyang welfare",
    "Content about the Opposition party",
    "Others",
]

CATEGORY_LOOKUP = {c.lower(): c for c in CATEGORIES}


def match_category(value):
    """Return exact official category name, or None if not valid."""
    if value is None:
        return None
    cleaned = re.sub(r"[*_`]", "", str(value)).strip().lower()
    return CATEGORY_LOOKUP.get(cleaned)


# ============================================================
# PROMPT (same as before, only the election name changed)
# ============================================================

def build_prompt(text):
    return f"""
You are a political analyst for Tamil Nadu 2026 elections.

Choose the **most dominant category** as the Main category and if there is a clear second important category, mention it as the Secondary category.

Categories:
{chr(10).join([f"- {cat}" for cat in CATEGORIES])}

Text:
{text}

Instructions:
- If the text clearly contains a substantial second theme that could be classified into a different category, also give that as the Secondary category.
- Provide a Secondary category only when there is genuine ambiguity or a clearly important second theme.
- Do NOT provide a Secondary category just because another category is mentioned briefly.
- The Main and Secondary categories must be different.
- Both categories MUST be exactly from the list above.
- Do not invent, modify, or combine category names.
- If there is no clear second category, write "Secondary: None".

Format:
Main: Category Name
Secondary: Category Name

Examples:

Main: Agriculture and Farmers
Secondary: Welfare and Funds
"""


def parse_response(response_text):
    """Parse 'Main: ...' / 'Secondary: ...' lines. Returns (main, secondary)."""
    main_raw, sec_raw = None, None
    for line in response_text.splitlines():
        line = re.sub(r"[*_`]", "", line).strip()
        if line.lower().startswith("main:"):
            main_raw = line.split(":", 1)[1].strip()
        elif line.lower().startswith("secondary:"):
            sec_raw = line.split(":", 1)[1].strip()

    main_cat = match_category(main_raw)
    if main_cat is None:
        raise ValueError(f"Invalid Main category: {main_raw!r}")

    sec_cat = None
    if sec_raw and sec_raw.lower() != "none":
        sec_cat = match_category(sec_raw)
        if sec_cat is None:
            print(f"  WARNING: invalid Secondary '{sec_raw}' -> ignored")
        elif sec_cat == main_cat:
            sec_cat = None

    return main_cat, sec_cat


def classify_chunk(text):
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = model.generate_content(build_prompt(text))
            main_cat, sec_cat = parse_response(response.text.strip())
            return {
                "Primary_Category": main_cat,
                "Secondary_Category": sec_cat or "",
                "Status": "success",
                "Error": "",
            }
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS} failed: {type(e).__name__}: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(5 * attempt)

    return {
        "Primary_Category": "",
        "Secondary_Category": "",
        "Status": "error",
        "Error": str(last_error)[:500],
    }


# ============================================================
# PROCESS ONE MANIFESTO (with resume support)
# ============================================================

def process_manifesto(party, folder_path):
    print("\n" + "=" * 60)
    print(f"Processing manifesto: {party}")
    print(f"Chunks folder: {folder_path}")
    print("=" * 60)

    if not folder_path.exists():
        print(f"WARNING: Folder not found: {folder_path}")
        return

    out_dir = RESULTS_BASE / party
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_NAME

    chunk_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    chunk_files.sort(key=lambda f: int(re.sub(r"\D", "", f) or 0))
    print(f"Chunks found: {len(chunk_files)}")

    # Resume: keep previously successful rows
    results = {}
    if out_path.exists():
        try:
            old = pd.read_csv(out_path, encoding="utf-8-sig", keep_default_na=False)
            for _, row in old.iterrows():
                if row.get("Status") == "success":
                    results[row["Chunk"]] = row.to_dict()
            print(f"Resuming: {len(results)} chunks already done")
        except Exception as e:
            print(f"Could not read old {OUTPUT_NAME}, starting fresh: {e}")

    for i, chunk_file in enumerate(chunk_files, start=1):
        if chunk_file in results:
            continue

        print(f"\nClassifying {chunk_file} ({i}/{len(chunk_files)})...")
        with open(folder_path / chunk_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print("  Empty chunk, skipped.")
            continue

        result = {"Party": party, "Chunk": chunk_file, **classify_chunk(text)}
        results[chunk_file] = result
        print(f"  Main: {result['Primary_Category']} | Secondary: {result['Secondary_Category'] or 'None'}")

        # checkpoint after every chunk
        pd.DataFrame(list(results.values())).to_csv(out_path, index=False, encoding="utf-8-sig")
        time.sleep(SLEEP_BETWEEN_CALLS)

    ordered = [results[c] for c in chunk_files if c in results]
    df = pd.DataFrame(ordered)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    errors = (df["Status"] != "success").sum() if len(df) else 0
    print(f"\nSaved: {out_path} ({len(df)} chunks, {errors} errors)")
    if errors:
        print("Re-run the script to retry the failed chunks.")


def main():
    for party, folder_path in MANIFESTOS.items():
        process_manifesto(party, folder_path)


if __name__ == "__main__":
    main()