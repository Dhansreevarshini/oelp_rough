
import os
import re
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# Speech chunks
SPEECH_CHUNKS_BASE = ROOT / "chunks_speech"
SPEECH_RESULTS_BASE = ROOT / "output" / "speech_results"

# Manifesto chunks
MANIFESTO_CHUNKS_BASE = ROOT / "chunks"
MANIFESTO_RESULTS_BASE = ROOT / "output" / "manifesto_results"

PARTIES = ["AIADMK", "DMK", "TVK"]

# Keep outputs separate from Gemini and Groq.
OUTPUT_NAME = "gemma.csv"


# ============================================================
# LOCAL OLLAMA SETUP
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:12b"

MAX_ATTEMPTS = 3
REQUEST_TIMEOUT = 600

# Resume successfully classified chunks.
RESUME = True

# Pause between chunks to avoid continuously loading the model.
SLEEP_BETWEEN_CHUNKS = 2


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
    """Return the exact official category name, or None if invalid."""

    if value is None:
        return None

    cleaned = re.sub(
        r"[\*\_\`]",
        "",
        str(value)
    ).strip().lower()

    return CATEGORY_LOOKUP.get(cleaned)


# ============================================================
# CHUNK TEXT
# ============================================================

HEADER_SEPARATOR = "-" * 40


def strip_header(raw_text):
    """Remove a chunk header separated by 40 hyphens."""

    lines = raw_text.splitlines()

    for i, line in enumerate(lines):
        if line.strip() == HEADER_SEPARATOR:
            return "\n".join(lines[i + 1:]).strip()

    return raw_text.strip()


def natural_key(filename):
    """Sort chunk names naturally, e.g. chunk_2 before chunk_10."""

    return [
        int(p) if p.isdigit() else p.lower()
        for p in re.split(r"(\d+)", filename)
    ]


# ============================================================
# PROMPT
# UNCHANGED FROM YOUR EXISTING SCRIPT
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
# PARSE MODEL RESPONSE
# ============================================================

def parse_response(response_text):
    """Parse Main and Secondary category labels."""

    main_raw = None
    sec_raw = None

    for line in response_text.splitlines():

        line = re.sub(
            r"[\*\_\`]",
            "",
            line
        ).strip()

        if line.lower().startswith("main:"):
            main_raw = line.split(":", 1)[1].strip()

        elif line.lower().startswith("secondary:"):
            sec_raw = line.split(":", 1)[1].strip()

    main_cat = match_category(main_raw)

    if main_cat is None:
        raise ValueError(
            f"Invalid Main category: {main_raw!r}. "
            f"Model response: {response_text[:500]!r}"
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
# CHECK OLLAMA CONNECTION
# ============================================================

def check_ollama():
    """Check the local Ollama server and selected model."""

    tags_url = "http://localhost:11434/api/tags"

    try:
        with urllib.request.urlopen(
            tags_url,
            timeout=10
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

    except Exception as e:
        raise RuntimeError(
            "Could not connect to Ollama at "
            "http://localhost:11434. Make sure Ollama is running. "
            + str(e)
        )

    available_models = [
        model.get("name", "")
        for model in data.get("models", [])
    ]

    model_available = any(
        name == MODEL_NAME
        for name in available_models
    )

    if not model_available:
        raise RuntimeError(
            f"Model '{MODEL_NAME}' was not found. "
            f"Available models: {available_models}"
        )

    print(f"Ollama connection successful: {MODEL_NAME}")


# ============================================================
# CLASSIFY ONE CHUNK
# ============================================================

def classify_chunk(text):

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:
            data = {
                "model": MODEL_NAME,
                "prompt": build_prompt(text),
                "stream": False,
                "options": {
                    "temperature": 0
                }
            }

            request = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(data).encode("utf-8"),
                headers={
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            with urllib.request.urlopen(
                request,
                timeout=REQUEST_TIMEOUT
            ) as response:

                response_data = json.loads(
                    response.read().decode("utf-8")
                )

            response_text = response_data.get(
                "response", ""
            ).strip()

            if not response_text:
                raise ValueError(
                    "Empty response received from Ollama"
                )

            main_cat, sec_cat = parse_response(response_text)

            return {
                "Primary_Category": main_cat,
                "Secondary_Category": sec_cat or "",
                "Status": "success",
                "Error": ""
            }

        except Exception as e:

            last_error = e

            print(
                f"  Attempt {attempt}/{MAX_ATTEMPTS} failed: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(5 * attempt)

    return {
        "Primary_Category": "",
        "Secondary_Category": "",
        "Status": "error",
        "Error": str(last_error)[:500]
    }


# ============================================================
# PROCESS ONE PARTY'S CHUNKS
# ============================================================

def process_party(party, folder_path, results_base, dataset_name):

    print("\n" + "=" * 65)
    print(f"Dataset: {dataset_name}")
    print(f"Party: {party}")
    print(f"Chunks folder: {folder_path}")
    print("=" * 65)

    if not folder_path.exists():
        print(f"WARNING: Folder not found: {folder_path}")
        print("Skipping this party.")
        return

    out_dir = results_base / party
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / OUTPUT_NAME

    chunk_files = [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(".txt")
    ]

    chunk_files.sort(key=natural_key)

    print(f"Chunks found: {len(chunk_files)}")

    if not chunk_files:
        print("No .txt files found. Skipping.")
        return

    # Keep only successful previous results for resuming.
    results = {}

    if RESUME and out_path.exists():

        try:
            old = pd.read_csv(
                out_path,
                encoding="utf-8-sig",
                keep_default_na=False
            )

            for _, row in old.iterrows():

                if (
                    row.get("Status") == "success"
                    and row.get("Chunk", "") != ""
                ):
                    results[row["Chunk"]] = row.to_dict()

            print(
                f"Resuming: {len(results)} previously successful "
                f"chunks found."
            )

        except Exception as e:
            print(f"Could not read previous results: {e}")
            print("Starting without previous checkpoints.")

    def save_results():

        ordered_results = [
            results[c]
            for c in chunk_files
            if c in results
        ]

        pd.DataFrame(ordered_results).to_csv(
            out_path,
            index=False,
            encoding="utf-8-sig"
        )

    total = len(chunk_files)
    skipped = 0
    processed = 0
    failed = 0

    for i, chunk_file in enumerate(chunk_files, start=1):

        if chunk_file in results:
            skipped += 1
            continue

        print(
            f"\n[{dataset_name} | {party}] "
            f"Chunk {i}/{total}: {chunk_file}"
        )

        try:
            with open(
                folder_path / chunk_file,
                "r",
                encoding="utf-8"
            ) as f:
                text = strip_header(f.read())

        except Exception as e:

            print(f"  File read error: {e}")

            results[chunk_file] = {
                "Party": party,
                "Chunk": chunk_file,
                "Primary_Category": "",
                "Secondary_Category": "",
                "Status": "error",
                "Error": f"File read error: {str(e)[:400]}"
            }

            failed += 1
            save_results()
            continue

        if not text:
            print("  Empty chunk. Skipping.")
            continue

        result = classify_chunk(text)

        result["Party"] = party
        result["Chunk"] = chunk_file

        results[chunk_file] = result
        processed += 1

        if result["Status"] == "success":

            print(
                f"  Main: {result['Primary_Category']} "
                f"| Secondary: "
                f"{result['Secondary_Category'] or 'None'}"
            )

        else:

            failed += 1
            print(f"  Classification failed: {result['Error']}")

        # Save after every chunk.
        save_results()

        if SLEEP_BETWEEN_CHUNKS > 0:
            time.sleep(SLEEP_BETWEEN_CHUNKS)

    # Final save.
    save_results()

    print("\n" + "-" * 65)
    print(f"Finished: {dataset_name} | {party}")
    print(f"Total chunks: {total}")
    print(f"Newly processed: {processed}")
    print(f"Previously successful/skipped: {skipped}")
    print(f"Failed this run: {failed}")
    print(f"Results saved to: {out_path}")
    print("-" * 65)


# ============================================================
# PROCESS A COMPLETE DATASET
# ============================================================

def process_dataset(dataset_name, chunks_base, results_base):

    print("\n\n" + "#" * 70)
    print(f"STARTING DATASET: {dataset_name}")
    print(f"Input: {chunks_base}")
    print(f"Output: {results_base}")
    print("#" * 70)

    if not chunks_base.exists():
        print(f"WARNING: Dataset folder does not exist: {chunks_base}")
        print("Skipping this dataset.")
        return

    for party in PARTIES:

        folder_path = chunks_base / party

        process_party(
            party,
            folder_path,
            results_base,
            dataset_name
        )


# ============================================================
# MAIN
# SPEECHES FIRST, THEN MANIFESTOS
# ============================================================

def main():

    check_ollama()

    print(f"\nModel: {MODEL_NAME}")
    print(f"Output filename: {OUTPUT_NAME}")
    print(f"Resume enabled: {RESUME}")

    # Stage 1: classify all speech chunks.
    process_dataset(
        "SPEECH",
        SPEECH_CHUNKS_BASE,
        SPEECH_RESULTS_BASE
    )

    # Stage 2: automatically start manifesto chunks.
    process_dataset(
        "MANIFESTO",
        MANIFESTO_CHUNKS_BASE,
        MANIFESTO_RESULTS_BASE
    )

    print("\n\n" + "=" * 65)
    print("ALL DATASETS FINISHED")
    print("Speech results:    output/speech_results/")
    print("Manifesto results: output/manifesto_results/")
    print("=" * 65)


if __name__ == "__main__":
    main()
