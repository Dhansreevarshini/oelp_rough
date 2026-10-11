import re
from pathlib import Path


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================
#
# manifesto_analysis/
# ├── speech_transcripts/     <- INPUT  (AIADMK/, DMK/, TVK/ with .txt files)
# ├── chunks_speech/          <- OUTPUT (created automatically)
# │   ├── AIADMK/
# │   ├── DMK/
# │   └── TVK/
# └── make_chunks_speech.py   <- this script

BASE_DIR = Path(__file__).resolve().parent
SPEECH_DIR = BASE_DIR / "speech_transcripts" / "zone_wise"
OUTPUT_DIR = BASE_DIR / "chunks_zone"

PARTIES = ["admk", "dmk", "tvk"]

CHUNK_SIZE = 150  # max words per chunk

# Put source-file info at the top of each chunk .txt file.
# Set to False for only the pure speech text.
INCLUDE_HEADER = True


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def read_text_file(path):
    """Read a text file as UTF-8 (Tamil-safe), with fallbacks."""
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue

    print(f"    Warning: encoding problem in {path.name}, "
          "some characters may be replaced")
    return path.read_text(encoding="utf-8", errors="replace")


def clean_spaces(text):
    """Normalize whitespace without changing the wording."""
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_units(text):
    """Split text into lines, then sentences."""
    text = clean_spaces(text)
    units = []

    for line in re.split(r"\n+", text):
        line = line.strip()
        if not line:
            continue

        for sentence in re.split(r"(?<=[.!?।])\s+", line):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)

    return units


def create_chunks(text):
    """
    Pack sentences into chunks of at most CHUNK_SIZE words.
    Any single sentence/line longer than CHUNK_SIZE (common in
    transcripts without punctuation) is split by words.
    """
    chunks = []
    current_parts = []
    current_words = 0

    def flush():
        nonlocal current_parts, current_words
        if current_words > 0:
            chunks.append({
                "word_count": current_words,
                "text": " ".join(current_parts),
            })
        current_parts = []
        current_words = 0

    for unit in split_into_units(text):
        words = unit.split()

        pieces = [
            words[i:i + CHUNK_SIZE]
            for i in range(0, len(words), CHUNK_SIZE)
        ]

        for piece in pieces:
            # Would overflow the current chunk: save it first.
            if current_words > 0 and current_words + len(piece) > CHUNK_SIZE:
                flush()

            current_parts.append(" ".join(piece))
            current_words += len(piece)

            if current_words >= CHUNK_SIZE:
                flush()

    flush()
    return chunks


def save_chunks(chunks, party_out_dir, source_name, party):
    """Write chunks as <source_name>_0001.txt, _0002.txt, ..."""
    for number, chunk in enumerate(chunks, start=1):
        chunk_id = f"{source_name}_{number:04d}"
        file_path = party_out_dir / f"{chunk_id}.txt"

        if INCLUDE_HEADER:
            content = (
                f"Party folder: {party}\n"
                f"Source: {source_name}.txt\n"
                f"Chunk: {number}/{len(chunks)}\n"
                f"Words: {chunk['word_count']}\n"
                f"{'-' * 40}\n"
                f"{chunk['text']}\n"
            )
        else:
            content = chunk["text"] + "\n"

        file_path.write_text(content, encoding="utf-8")


# ============================================================
# 3. PROCESS ALL PARTIES
# ============================================================

def main():
    if not SPEECH_DIR.is_dir():
        raise FileNotFoundError(
            f"Speech folder not found: {SPEECH_DIR}\n"
            "Keep this script inside manifesto_analysis/ next to "
            "speech_transcripts/."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for party in PARTIES:
        party_in_dir = SPEECH_DIR / party
        party_out_dir = OUTPUT_DIR / party
        party_out_dir.mkdir(parents=True, exist_ok=True)

        # Remove old chunk files so stale ones don't remain on re-run.
        for old_file in party_out_dir.glob("*.txt"):
            old_file.unlink()

        if not party_in_dir.is_dir():
            print(f"\n{party}: folder not found, skipped "
                  f"({party_in_dir})")
            summary[party] = (0, 0)
            continue

        txt_files = sorted(party_in_dir.glob("*.txt"))
        print(f"\nProcessing {party}: {len(txt_files)} transcript files")

        total_chunks = 0

        for txt_file in txt_files:
            text = read_text_file(txt_file)
            chunks = create_chunks(text)

            if not chunks:
                print(f"  {txt_file.name}: empty, skipped")
                continue

            save_chunks(chunks, party_out_dir, txt_file.stem, party)
            total_chunks += len(chunks)
            print(f"  {txt_file.name}: {len(chunks)} chunks")

        summary[party] = (len(txt_files), total_chunks)

    print("\n========================================")
    print("SPEECH CHUNKING COMPLETED")
    print("========================================")
    print(f"Output folder: {OUTPUT_DIR}\n")
    for party, (files, chunks) in summary.items():
        print(f"  {party}: {files} files -> {chunks} chunks")


if __name__ == "__main__":
    main()