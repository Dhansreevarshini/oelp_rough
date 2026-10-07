from pathlib import Path

# Place this script in manifesto_analysis/ (next to the other scripts)
ROOT = Path(__file__).resolve().parent

# Folders searched (including all sub-folders)
SEARCH_FOLDERS = ["speech_transcripts", "chunks_speech"]

# Speeches to remove. Any .txt file whose NAME contains one of these is matched.
SPEECH_IDS = [
    "yOzSVsXvD0s",
    "lTnUJH5tk24",
    "1UdL6HSav2k",
    "dfgDeyV_k1k",
    "LxIuBGCVyqs",
]

# True  = only PRINT what would be deleted (safe preview)
# False = actually delete the files
DRY_RUN = False


def main():
    print(f"Script location (ROOT): {ROOT}\n")

    all_files = []
    for folder in SEARCH_FOLDERS:
        folder_path = ROOT / folder
        if not folder_path.is_dir():
            print(f"WARNING: folder not found: {folder_path}")
            continue
        files = [f for f in folder_path.rglob("*") if f.is_file()]
        print(f"{folder}: {len(files)} files scanned")
        all_files += files

    print()
    found = 0
    for speech_id in SPEECH_IDS:
        matches = [f for f in all_files if speech_id in f.name]
        print(f"{speech_id}: {len(matches)} file(s)")
        for f in matches:
            found += 1
            if DRY_RUN:
                print(f"   [preview] would delete: {f.relative_to(ROOT)}")
            else:
                f.unlink()
                print(f"   deleted: {f.relative_to(ROOT)}")

    print(f"\n{found} file(s) {'found' if DRY_RUN else 'deleted'}.")

    if found == 0 and all_files:
        print("\nNo file NAME contains those IDs. Example file names in your folders:")
        for f in all_files[:5]:
            print(f"   {f.relative_to(ROOT)}")
    elif DRY_RUN and found:
        print("Nothing was deleted. Set DRY_RUN = False and run again to delete.")


if __name__ == "__main__":
    main()