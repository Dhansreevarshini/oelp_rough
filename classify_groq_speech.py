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
# ├── chunks_speech/
# │   ├── AIADMK/*.txt
# │   ├── DMK/*.txt
# │   └── TVK/*.txt
# ├── output/
# │   └── speech_results/<PARTY>/groq.csv
# ├── .env
# └── classify_groq_speech.py
# ============================================================

ROOT = Path(__file__).resolve().parent
CHUNKS_BASE = ROOT / "chunks_speech"
RESULTS_BASE = ROOT / "output" / "speech_results"

PARTIES = ["AIADMK", "DMK", "TVK"]

SPEECHES = {
    party: CHUNKS_BASE / party
    for party in PARTIES
}

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

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "qwen/qwen3.8-27b"
)

client = Groq(api_key=API_KEY)

SLEEP_BETWEEN_CALLS = 8
MAX_ATTEMPTS = 3

# False = always start from first chunk
# True = skip chunks already classified successfully
RESUME = True


# ============================================================
# CATEGORIES
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
    "Already executed promises",
]

CATEGORY_LOOKUP = {
    c.lower(): c
    for c in CATEGORIES
}


def match_category(value):
    """Return exact official category name, or None if not valid."""

    if value is None:
        return None

    cleaned = re.sub(
        r"[*_`]",
        "",
        str(value)
    ).strip().lower()

    return CATEGORY_LOOKUP.get(cleaned)


# ============================================================
# CHUNK TEXT
# ============================================================

HEADER_SEPARATOR = "-" * 40


def strip_header(raw_text):
    """
    Remove the header made by make_chunks_speech.py.
    """

    lines = raw_text.splitlines()

    for i, line in enumerate(lines):

        if line.strip() == HEADER_SEPARATOR:
            return "\n".join(
                lines[i + 1:]
            ).strip()

    return raw_text.strip()


def natural_key(filename):
    """
    Sort:
    speech_0002.txt before speech_0010.txt
    speech2.txt before speech10.txt
    """

    return [
        int(p) if p.isdigit() else p.lower()
        for p in re.split(r"(\d+)", filename)
    ]


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

About the category "Already executed promises":
- Use it ONLY when the speaker claims that a specific promise, scheme, project or commitment made by their party or government has actually been delivered or completed (for example: "we promised free bus travel for women and we have implemented it", "the hospital we announced has been built and opened").
- When a chunk does describe such a delivered promise, choose "Already executed promises" as the Main category and, if the delivered work clearly belongs to a topic such as Education, Health or Infrastructure, give that topic as the Secondary category.
- Past tense alone does NOT make a chunk "Already executed promises". Classify by its topic instead in these cases:
  - recounting history, past events, or past leaders' lives without claiming a delivered promise
  - criticising the opposition's past actions or failures (use "Content about the Opposition party")
  - personal stories, anecdotes, or descriptions of past rallies or meetings
  - a past promise that the speaker says was NOT fulfilled or was broken
  - vague boasts with no identifiable promise, scheme, or work (e.g. "we did a lot of good work")
  - future promises or plans, even if worded with "will have done"

Format:
Main: Category Name
Secondary: Category Name

Examples:

Main: Agriculture and Farmers
Secondary: Welfare and Funds

Main: Already executed promises
Secondary: Education
"""


# ============================================================
# PARSE GROQ RESPONSE
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

    main_cat = match_category(main_raw)

    if main_cat is None:
        raise ValueError(
            f"Invalid Main category: {main_raw!r}"
        )

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
# CLASSIFY ONE CHUNK USING GROQ + QWEN
# ============================================================

def classify_chunk(text):

    last_error = None

    for attempt in range(
        1,
        MAX_ATTEMPTS + 1
    ):

        try:

            # ------------------------------------------------
            # GROQ API CALL
            # ------------------------------------------------

            response = client.chat.completions.create(

                model=MODEL_NAME,

                messages=[
                    {
                        "role": "user",
                        "content": build_prompt(text)
                    }
                ],

                reasoning_effort="none"
            )

            # ------------------------------------------------
            # GET TEXT FROM GROQ RESPONSE
            # ------------------------------------------------

            response_text = (
                response
                .choices[0]
                .message
                .content
            )

            if not response_text:
                raise ValueError(
                    "Empty response received from Groq"
                )

            # ------------------------------------------------
            # PARSE RESPONSE
            # ------------------------------------------------

            main_cat, sec_cat = parse_response(
                response_text.strip()
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

    # --------------------------------------------------------
    # ALL ATTEMPTS FAILED
    # --------------------------------------------------------

    return {
        "Primary_Category": "",
        "Secondary_Category": "",
        "Status": "error",
        "Error": str(last_error)[:500],
    }


# ============================================================
# PROCESS ONE PARTY
# ============================================================

def process_party(
    party,
    folder_path
):

    print("\n" + "=" * 60)

    print(
        f"Processing speeches: {party}"
    )

    print(
        f"Chunks folder: {folder_path}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # CHECK FOLDER
    # --------------------------------------------------------

    if not folder_path.exists():

        print(
            f"WARNING: Folder not found: "
            f"{folder_path}"
        )

        return


    # --------------------------------------------------------
    # OUTPUT FOLDER
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # GET CHUNK FILES
    # --------------------------------------------------------

    chunk_files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".txt")
    ]

    chunk_files.sort(
        key=natural_key
    )

    print(
        f"Chunks found: {len(chunk_files)}"
    )


    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    results = {}

    if RESUME and out_path.exists():

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


    # --------------------------------------------------------
    # PROCESS CHUNKS
    # --------------------------------------------------------

    for i, chunk_file in enumerate(
        chunk_files,
        start=1
    ):

        # ----------------------------------------------------
        # SKIP SUCCESSFULLY PROCESSED CHUNKS
        # ----------------------------------------------------

        if chunk_file in results:

            continue


        print(
            f"\nClassifying {chunk_file} "
            f"({i}/{len(chunk_files)})..."
        )


        # ----------------------------------------------------
        # READ CHUNK
        # ----------------------------------------------------

        with open(
            folder_path / chunk_file,
            "r",
            encoding="utf-8"
        ) as f:

            text = strip_header(
                f.read()
            )


        # ----------------------------------------------------
        # EMPTY CHUNK
        # ----------------------------------------------------

        if not text:

            print(
                "  Empty chunk, skipped."
            )

            continue


        # ----------------------------------------------------
        # CLASSIFY
        # ----------------------------------------------------

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
            f"{result['Primary_Category']} "
            f"| Secondary: "
            f"{result['Secondary_Category'] or 'None'}"
        )


        # ----------------------------------------------------
        # CHECKPOINT AFTER EVERY CHUNK
        # ----------------------------------------------------

        pd.DataFrame(
            list(results.values())
        ).to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig"
        )


        # ----------------------------------------------------
        # WAIT BEFORE NEXT API CALL
        # ----------------------------------------------------

        time.sleep(
            SLEEP_BETWEEN_CALLS
        )


    # --------------------------------------------------------
    # SAVE FINAL ORDERED CSV
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # COUNT ERRORS
    # --------------------------------------------------------

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
            "Re-run the script to retry "
            "the failed chunks."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Using Groq model: {MODEL_NAME}"
    )

    for party, folder_path in SPEECHES.items():

        process_party(
            party,
            folder_path
        )


if __name__ == "__main__":

    main()