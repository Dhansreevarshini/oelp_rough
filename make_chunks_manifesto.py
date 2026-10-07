import re
import io
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import pytesseract
from PIL import Image


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================
#
# manifesto_analysis/
# ├── chunks/          <- chunks/AIADMK, DMK, TVK (.txt files)
# ├── data/            <- the 3 manifesto PDFs
# ├── output/          <- page CSVs + extraction_log.csv
# └── make_chunks_manifesto.py   <- this script


def find_base_dir():
    """Find the manifesto_analysis folder (the one that contains data/)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data").is_dir():
            return candidate
    return here


BASE_DIR = find_base_dir()
PDF_DIR = BASE_DIR / "data"
CHUNKS_DIR = BASE_DIR / "chunks"
OUTPUT_DIR = BASE_DIR / "output"

CHUNK_SIZE = 300  # max words per chunk

# Put party / page info at the top of each .txt file.
# Set to False for only the pure manifesto text.
INCLUDE_HEADER = True

# Tamil + English OCR.
OCR_LANGUAGE = "tam+eng"

# If Tesseract is not in your Windows PATH, uncomment and edit:
#
# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# ------------------------------------------------------------
# WHICH PARTIES TO (RE)PROCESS
# Only the parties listed here are touched. Their chunk .txt files
# and page CSVs are overwritten; every other party is left alone.
# ------------------------------------------------------------
PARTIES_TO_PROCESS = ["AIADMK"]

# Parties whose pages are ALWAYS OCR'd (the AIADMK PDF uses an old
# non-Unicode Tamil font, so its embedded text is garbage).
FORCE_OCR = {"AIADMK"}

# Each party is matched to the PDF in data/ whose filename starts
# with the prefix below (case-insensitive).
MANIFESTO_PREFIXES = {
    "TVK": "tvk",
    "DMK": "dmk",
    "AIADMK": "aiadmk",
}


def find_pdf(prefix):
    """Return the PDF in data/ whose name starts with the prefix."""
    matches = sorted(
        p for p in PDF_DIR.glob("*.pdf")
        if p.name.lower().startswith(prefix.lower())
    )

    if not matches:
        raise FileNotFoundError(
            f"No PDF starting with '{prefix}' found in {PDF_DIR}"
        )

    if len(matches) > 1:
        print(f"  Warning: several PDFs match '{prefix}', "
              f"using {matches[0].name}")

    return matches[0]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_spaces(text):
    """Normalize whitespace without changing Tamil wording."""
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Characters that appear when a PDF uses an old non-Unicode Tamil font
LEGACY_EXTRA = set("‹›œŒšŠžŽŸƒˆ˜†‡‰")


def legacy_font_ratio(text):
    """Share of characters that look like legacy-font Tamil garbage."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0

    bad = sum(
        1 for c in chars
        if 0xA1 <= ord(c) <= 0xFF or c in LEGACY_EXTRA
    )
    return bad / len(chars)


def needs_ocr(text):
    """
    Decide whether a page's embedded text is too sparse, corrupted,
    or in a legacy (non-Unicode) Tamil font. A heuristic only.
    """
    text = clean_spaces(text)

    if len(text) < 80:
        return True

    # Replacement characters usually indicate broken text.
    if "\ufffd" in text:
        return True

    letters = sum(char.isalpha() for char in text)
    if letters < 40:
        return True

    # Legacy Tamil font: Latin-looking garbage instead of Tamil Unicode.
    if legacy_font_ratio(text) > 0.05:
        return True

    return False


def run_ocr(page):
    """Convert one PDF page to an image and OCR it."""
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))

    return pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGE,
        config="--psm 3"
    )


def check_ocr_ready():
    """Stop early if Tesseract or the Tamil language data is missing."""
    try:
        available = pytesseract.get_languages(config="")
    except Exception as error:
        raise RuntimeError(
            "Tesseract could not be run. Install it and add it to PATH, "
            "or set pytesseract.pytesseract.tesseract_cmd.\n"
            f"Details: {error}"
        )

    missing = [
        lang for lang in OCR_LANGUAGE.split("+")
        if lang not in available
    ]
    if missing:
        raise RuntimeError(
            f"Tesseract language data missing: {missing}. "
            "Reinstall Tesseract and tick the Tamil language data."
        )


def normalize_header(line):
    """
    Normalize a line for comparing recurring headers/footers.
    Page numbers and punctuation are ignored.
    """
    line = re.sub(r"\d+", "", line)
    line = re.sub(r"[^\w\u0B80-\u0BFF]+", "", line)
    return line.strip().lower()


def remove_repeated_page_lines(pages):
    """
    Remove lines repeated at the top or bottom of multiple pages
    (headers/footers) and lines that are only page numbers.
    """
    line_counts = Counter()

    for page_data in pages:
        lines = page_data["raw_text"].splitlines()
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            continue

        edge_lines = lines[:3] + lines[-3:]

        for line in edge_lines:
            normalized = normalize_header(line)
            if normalized:
                line_counts[normalized] += 1

    # Must repeat on at least 3 pages and at least 20% of pages.
    page_count = len(pages)
    min_repetitions = max(3, int(page_count * 0.20))

    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= min_repetitions
    }

    cleaned_pages = []

    for page_data in pages:
        lines = page_data["raw_text"].splitlines()
        kept_lines = []

        for index, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                kept_lines.append("")
                continue

            normalized = normalize_header(stripped)

            is_edge = index < 3 or index >= len(lines) - 3

            if is_edge and normalized in repeated_lines:
                continue

            # Remove lines containing only page numbers.
            if re.fullmatch(r"[-–—| ]*\d{1,4}[-–—| ]*", stripped):
                continue

            kept_lines.append(line)

        cleaned_text = clean_spaces("\n".join(kept_lines))

        cleaned_pages.append({
            **page_data,
            "cleaned_text": cleaned_text,
        })

    return cleaned_pages


def split_into_units(text):
    """Split text at paragraph and sentence boundaries."""
    text = clean_spaces(text)

    paragraphs = re.split(r"\n+", text)
    units = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        sentences = re.split(r"(?<=[.!?।])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if sentence:
                units.append(sentence)

    return units


def make_chunk(party, chunk_number, start_page, end_page,
               word_count, parts):
    return {
        "chunk_id": f"{party}_{chunk_number:04d}",
        "party": party,
        "start_page": start_page,
        "end_page": end_page,
        "word_count": word_count,
        "text": " ".join(parts),
    }


def create_chunks(pages, party):
    """
    Pack sentence/paragraph units into chunks of at most
    CHUNK_SIZE words. Very long sentences are split by words.
    Page numbers are retained for every chunk.
    """
    chunks = []

    current_parts = []
    current_words = 0
    current_start_page = None
    current_end_page = None
    chunk_number = 1

    for page_data in pages:
        page_number = page_data["page_number"]
        text = page_data["cleaned_text"]

        if not text:
            continue

        for unit in split_into_units(text):
            words = unit.split()

            # Split any single unit longer than the chunk limit.
            unit_parts = [
                words[i:i + CHUNK_SIZE]
                for i in range(0, len(words), CHUNK_SIZE)
            ]

            for word_part in unit_parts:
                part_text = " ".join(word_part)
                part_count = len(word_part)

                # Current chunk would overflow: save it first.
                if (
                    current_words > 0
                    and current_words + part_count > CHUNK_SIZE
                ):
                    chunks.append(make_chunk(
                        party, chunk_number, current_start_page,
                        current_end_page, current_words, current_parts
                    ))
                    chunk_number += 1
                    current_parts = []
                    current_words = 0
                    current_start_page = None
                    current_end_page = None

                if current_start_page is None:
                    current_start_page = page_number

                current_end_page = page_number
                current_parts.append(part_text)
                current_words += part_count

                # Emit a full chunk immediately.
                if current_words >= CHUNK_SIZE:
                    chunks.append(make_chunk(
                        party, chunk_number, current_start_page,
                        current_end_page, current_words, current_parts
                    ))
                    chunk_number += 1
                    current_parts = []
                    current_words = 0
                    current_start_page = None
                    current_end_page = None

    # Save the final partial chunk.
    if current_words > 0:
        chunks.append(make_chunk(
            party, chunk_number, current_start_page,
            current_end_page, current_words, current_parts
        ))

    return chunks


def save_chunks_as_txt(chunks, party):
    """Write each chunk to chunks/<party>/<chunk_id>.txt (UTF-8)."""
    party_dir = CHUNKS_DIR / party
    party_dir.mkdir(parents=True, exist_ok=True)

    # Remove only THIS party's old chunk files.
    for old_file in party_dir.glob("*.txt"):
        old_file.unlink()

    for chunk in chunks:
        file_path = party_dir / f"{chunk['chunk_id']}.txt"

        if INCLUDE_HEADER:
            if chunk["start_page"] == chunk["end_page"]:
                pages = f"{chunk['start_page']}"
            else:
                pages = f"{chunk['start_page']}-{chunk['end_page']}"

            content = (
                f"Party: {chunk['party']}\n"
                f"Chunk: {chunk['chunk_id']}\n"
                f"Pages: {pages}\n"
                f"Words: {chunk['word_count']}\n"
                f"{'-' * 40}\n"
                f"{chunk['text']}\n"
            )
        else:
            content = chunk["text"] + "\n"

        file_path.write_text(content, encoding="utf-8")

    return party_dir


# ============================================================
# 3. PROCESS ONE MANIFESTO
# ============================================================

def process_manifesto(party, prefix):
    pdf_path = find_pdf(prefix)

    print(f"\nProcessing {party}: {pdf_path.name}")

    page_records = []

    with fitz.open(pdf_path) as document:
        total_pages = len(document)

        for page_index, page in enumerate(document):
            page_number = page_index + 1

            # Extract embedded PDF text first.
            embedded_text = clean_spaces(page.get_text("text"))

            # OCR if forced for this party, or if the embedded text
            # looks unusable (sparse, corrupted, or legacy Tamil font).
            if party in FORCE_OCR or needs_ocr(embedded_text):
                print(f"  Page {page_number}/{total_pages}: OCR")
                extracted_text = run_ocr(page)
                method = "OCR"
            else:
                extracted_text = embedded_text
                method = "PyMuPDF"

            page_records.append({
                "party": party,
                "page_number": page_number,
                "extraction_method": method,
                "raw_text": extracted_text,
            })

    # Save extraction before cleaning (useful for checking OCR quality).
    raw_df = pd.DataFrame(page_records)
    raw_path = OUTPUT_DIR / f"{party}_extracted_pages_raw.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    # Remove repeated headers/footers and page numbers.
    cleaned_pages = remove_repeated_page_lines(page_records)

    cleaned_df = pd.DataFrame([
        {
            "party": p["party"],
            "page_number": p["page_number"],
            "extraction_method": p["extraction_method"],
            "cleaned_text": p["cleaned_text"],
        }
        for p in cleaned_pages
    ])
    cleaned_path = OUTPUT_DIR / f"{party}_extracted_pages_cleaned.csv"
    cleaned_df.to_csv(cleaned_path, index=False, encoding="utf-8-sig")

    # Create chunks and save each as a .txt file.
    chunks = create_chunks(cleaned_pages, party)
    chunk_dir = save_chunks_as_txt(chunks, party)

    print(f"  Pages: {total_pages}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Raw text: {raw_path.name}")
    print(f"  Cleaned text: {cleaned_path.name}")
    print(f"  Chunk .txt files saved in: {chunk_dir}")

    return chunks, page_records


# ============================================================
# 4. MAIN
# ============================================================

def update_extraction_log(new_records, processed_parties):
    """
    Update output/extraction_log.csv without losing the log rows of
    parties that were not reprocessed in this run.
    """
    log_path = OUTPUT_DIR / "extraction_log.csv"
    new_df = pd.DataFrame(new_records)

    if log_path.exists():
        old_df = pd.read_csv(log_path, encoding="utf-8-sig")
        old_df = old_df[~old_df["party"].isin(processed_parties)]
        log_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        log_df = new_df

    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    return log_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    unknown = [p for p in PARTIES_TO_PROCESS if p not in MANIFESTO_PREFIXES]
    if unknown:
        raise ValueError(f"Unknown parties in PARTIES_TO_PROCESS: {unknown}")

    print(f"Project folder: {BASE_DIR}")
    print(f"PDF folder:     {PDF_DIR}")
    print(f"Processing only: {', '.join(PARTIES_TO_PROCESS)}")

    if any(party in FORCE_OCR for party in PARTIES_TO_PROCESS):
        check_ocr_ready()

    chunk_counts = {}
    all_log_records = []

    for party in PARTIES_TO_PROCESS:
        chunks, page_records = process_manifesto(
            party,
            MANIFESTO_PREFIXES[party]
        )
        chunk_counts[party] = len(chunks)

        for page in page_records:
            all_log_records.append({
                "party": page["party"],
                "page_number": page["page_number"],
                "extraction_method": page["extraction_method"],
                "raw_character_count": len(page["raw_text"]),
                "empty_or_short": len(page["raw_text"].strip()) < 80,
            })

    log_path = update_extraction_log(all_log_records, PARTIES_TO_PROCESS)

    print("\n========================================")
    print("EXTRACTION AND CHUNKING COMPLETED")
    print("========================================")
    print(f"Chunk folders: {CHUNKS_DIR}")
    print(f"Extraction log: {log_path}")
    print("\nChunks by party (this run):")
    for party, count in chunk_counts.items():
        print(f"  {party}: {count}")


if __name__ == "__main__":
    main()