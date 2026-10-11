
import json
import re
import shutil
from pathlib import Path

import pandas as pd

SUMMARY_DIR = Path("output/speech_summary")
BACKUP_DIR = Path("output/speech_summary_backup_before_location_fix")
LOG_CSV = Path("output/location_fix_log.csv")
JSON_NAME = "videos_with_locations.json"
FILE_PATTERN = "*per_video_percentages.csv"      # gemini_ and groq_ files both

# Speakers whose name has an underscore in it (needed to find where the place starts).
# Single-word speakers like eps / vijay / kanimozhi need no entry. Add yours if needed.
MULTIWORD_SPEAKERS = {"mk_stalin", "udhayanidhi_stalin", "edappadi_palanisamy"}

ID_LEN = 11      # a YouTube id is always 11 characters


def find_json():
    for p in (Path(JSON_NAME), Path("data") / JSON_NAME):
        if p.exists():
            return p
    hit = next(Path(".").rglob(JSON_NAME), None)
    if hit is None:
        raise SystemExit(f"Could not find {JSON_NAME}. Put it in the project folder.")
    return hit


def load_locations(path):
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, dict):                       # tolerate {"videos": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), [])
    mapping = {}
    for item in data:
        vid, loc = item.get("id"), item.get("detected_location")
        if vid and loc and str(loc).strip().lower() not in ("", "null", "none", "nan"):
            mapping[str(vid).strip()] = re.sub(r"\s+", "_", str(loc).strip().lower())
    return mapping, len(data)


def split_video(video):
    """'DMK_mk_stalin_paramakudi_AUpKk_Tp-wc' -> (party, speaker, place, id) or None"""
    v = str(video).strip()
    if len(v) < ID_LEN + 4 or v[-(ID_LEN + 1)] != "_":
        return None
    vid, head = v[-ID_LEN:], v[:-(ID_LEN + 1)]
    toks = head.split("_")
    party, rest = toks[0], toks[1:]
    n = 1
    for k in (3, 2):
        if len(rest) > k and "_".join(rest[:k]).lower() in MULTIWORD_SPEAKERS:
            n = k
            break
    if len(rest) <= n:
        return None
    return party, "_".join(rest[:n]), "_".join(rest[n:]), vid


def main():
    json_path = find_json()
    locs, total = load_locations(json_path)
    print(f"{json_path}: {total} videos, {len(locs)} with a detected location\n")

    files = sorted(SUMMARY_DIR.rglob(FILE_PATTERN))
    if not files:
        raise SystemExit(f"No {FILE_PATTERN} files under {SUMMARY_DIR}")

    log, speakers = [], set()
    for f in files:
        df = pd.read_csv(f)
        if "video" not in df.columns:
            print(f"[skip] {f} has no 'video' column")
            continue

        new_names, changed, no_loc, unparsed = [], 0, 0, 0
        for old in df["video"]:
            parts = split_video(old)
            if parts is None:
                new_names.append(old); unparsed += 1
                log.append({"file": str(f), "old": old, "new": old, "status": "could_not_parse"})
                continue
            party, speaker, place, vid = parts
            speakers.add((party, speaker))
            if vid not in locs:
                new_names.append(old); no_loc += 1
                log.append({"file": str(f), "old": old, "new": old, "status": "no_location_in_json"})
                continue
            new = f"{party}_{speaker}_{locs[vid]}_{vid}"
            new_names.append(new)
            if new != old:
                changed += 1
            log.append({"file": str(f), "old": old, "new": new,
                        "status": "changed" if new != old else "already_correct"})

        # back up the original once, then overwrite
        backup = BACKUP_DIR / f.relative_to(SUMMARY_DIR)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, backup)
        df["video"] = new_names
        df.to_csv(f, index=False)
        print(f"{f}: {len(df)} videos | renamed {changed} | left as is (no location) {no_loc}"
              f" | could not parse {unparsed}")

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(LOG_CSV, index=False)
    print("\nSpeakers read from the names (check these look right):")
    for party, sp in sorted(speakers):
        print(f"  {party}: {sp}")
    print(f"\nLog of every change : {LOG_CSV}")
    print(f"Originals saved in  : {BACKUP_DIR}")


if __name__ == "__main__":
    main()