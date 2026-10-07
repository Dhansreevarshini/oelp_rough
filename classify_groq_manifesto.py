import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq


# ============================================================
# PATHS
#
# manifesto_analysis/
# ├── chunks/
# │   ├── AIADMK/*.txt
# │   ├── DMK/*.txt
# │   └── TVK/*.txt
# ├── output/
# │   └── manifesto_results/
# │       └── <PARTY>/
# │           └── groq.csv
# ├── .env
# └── classify_manifesto.py
# ============================================================

ROOT = Path(__file__).resolve().parent

CHUNKS_BASE = ROOT / "chunks"
RESULTS_BASE = ROOT / "output" / "manifesto_results"


MANIFESTOS = {
    "AIADMK": CHUNKS_BASE / "AIADMK",
    "DMK": CHUNKS_BASE / "DMK",
    "TVK": CHUNKS_BASE / "TVK",
}


# Output file
OUTPUT_NAME = "groq.csv"


# ============================================================
# API SETUP
# ============================================================

load_dotenv(ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        f"GROQ_API_KEY not found. Check {ROOT / '.env'}"
    )


# Get model name from .env
MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


# Create Groq client
client = Groq(api_key=API_KEY)


# ============================================================
# SETTINGS
# ============================================================

SLEEP_BETWEEN_CALLS = 8
MAX_ATTEMPTS = 3


# ============================================================
# CATEGORIES (14)
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


CATEGORY_LOOKUP = {
    c.lower(): c
    for c in CATEGORIES
}


def match_category(value):
    """
    Return exact official category name,
    or None if not valid.
    """

    if value is None:
        return None

    cleaned = re.sub(
        r"[*_`]",
        "",
        str(value)
    ).strip().lower()

    return CATEGORY_LOOKUP.get(cleaned)


# ============================================================
# PROMPT
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


# ============================================================
# PARSE MODEL RESPONSE
# ============================================================

def parse_response(response_text):

    """
    Parse:

    Main: ...
    Secondary: ...

    Returns:
        (main_category, secondary_category)
    """

    main_raw = None
    sec_raw = None

    for line in response_text.splitlines():

        line = re.sub(
            r"[*_`]",
            "",
            line
        ).strip()

        if line.lower().startswith("main:"):
            main_raw = line.split(
                ":",
                1
            )[1].strip()

        elif line.lower().startswith("secondary:"):
            sec_raw = line.split(
                ":",
                1
            )[1].strip()


    # Check Main category
    main_cat = match_category(main_raw)

    if main_cat is None:
        raise ValueError(
            f"Invalid Main category: {main_raw!r}"
        )


    # Check Secondary category
    sec_cat = None

    if sec_raw and sec_raw.lower() != "none":

        sec_cat = match_category(sec_raw)

        if sec_cat is None:

            print(
                f"  WARNING: invalid Secondary "
                f"'{sec_raw}' -> ignored"
            )

        elif sec_cat == main_cat:

            sec_cat = None


    return main_cat, sec_cat


# ============================================================
# CLASSIFY ONE CHUNK
# ============================================================

def classify_chunk(text):

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        try:

            # Send prompt to Groq
            response = client.chat.completions.create(

                model=MODEL_NAME,

                messages=[
                    {
                        "role": "user",
                        "content": build_prompt(text)
                    }
                ],

                # Low temperature = more consistent
                temperature=0,

                # We only need a very short answer
                max_completion_tokens=100
            )


            # Get model's text response
            response_text = (
                response
                .choices[0]
                .message
                .content
                .strip()
            )


            # Parse:
            # Main: ...
            # Secondary: ...
            main_cat, sec_cat = parse_response(
                response_text
            )


            return {
                "Primary_Category": main_cat,
                "Secondary_Category": sec_cat or "",
                "Status": "success",
                "Error": "",
            }


        except Exception as e:

            last_error = e

            print(
                f"  Attempt {attempt}/{MAX_ATTEMPTS} "
                f"failed: {type(e).__name__}: {e}"
            )

            if attempt < MAX_ATTEMPTS:

                time.sleep(
                    5 * attempt
                )


    # If all attempts failed
    return {
        "Primary_Category": "",
        "Secondary_Category": "",
        "Status": "error",
        "Error": str(last_error)[:500],
    }


# ============================================================
# PROCESS ONE MANIFESTO
# ============================================================

def process_manifesto(
    party,
    folder_path
):

    print("\n" + "=" * 60)

    print(
        f"Processing manifesto: {party}"
    )

    print(
        f"Chunks folder: {folder_path}"
    )

    print("=" * 60)


    if not folder_path.exists():

        print(
            f"WARNING: Folder not found: {folder_path}"
        )

        return


    # Create output folder
    out_dir = (
        RESULTS_BASE / party
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    out_path = (
        out_dir / OUTPUT_NAME
    )


    # Find all .txt files
    chunk_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".txt")
    ]


    # Sort chunks numerically
    chunk_files.sort(
        key=lambda f:
        int(re.sub(r"\D", "", f) or 0)
    )


    print(
        f"Chunks found: {len(chunk_files)}"
    )


    # ========================================================
    # RESUME SUPPORT
    # ========================================================

    results = {}


    if out_path.exists():

        try:

            old = pd.read_csv(
                out_path,
                encoding="utf-8-sig",
                keep_default_na=False
            )


            for _, row in old.iterrows():

                if row.get("Status") == "success":

                    results[
                        row["Chunk"]
                    ] = row.to_dict()


            print(
                f"Resuming: {len(results)} "
                f"chunks already done"
            )


        except Exception as e:

            print(
                f"Could not read old "
                f"{OUTPUT_NAME}, "
                f"starting fresh: {e}"
            )


    # ========================================================
    # PROCESS EACH CHUNK
    # ========================================================

    for i, chunk_file in enumerate(
        chunk_files,
        start=1
    ):

        # Skip already completed chunks
        if chunk_file in results:
            continue


        print(
            f"\nClassifying {chunk_file} "
            f"({i}/{len(chunk_files)})..."
        )


        # Read chunk
        with open(
            folder_path / chunk_file,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read().strip()


        # Skip empty chunks
        if not text:

            print(
                "  Empty chunk, skipped."
            )

            continue


        # Classify
        result = {
            "Party": party,
            "Chunk": chunk_file,
            **classify_chunk(text)
        }


        results[
            chunk_file
        ] = result


        print(
            f"  Main: "
            f"{result['Primary_Category']} | "
            f"Secondary: "
            f"{result['Secondary_Category'] or 'None'}"
        )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        pd.DataFrame(
            list(results.values())
        ).to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig"
        )


        # Wait between API calls
        time.sleep(
            SLEEP_BETWEEN_CALLS
        )


    # ========================================================
    # FINAL SAVE
    # ========================================================

    ordered = [
        results[c]
        for c in chunk_files
        if c in results
    ]


    df = pd.DataFrame(
        ordered
    )


    df.to_csv(
        out_path,
        index=False,
        encoding="utf-8-sig"
    )


    errors = (
        (df["Status"] != "success").sum()
        if len(df)
        else 0
    )


    print(
        f"\nSaved: {out_path} "
        f"({len(df)} chunks, {errors} errors)"
    )


    if errors:

        print(
            "Re-run the script "
            "to retry the failed chunks."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    for party, folder_path in MANIFESTOS.items():

        process_manifesto(
            party,
            folder_path
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()