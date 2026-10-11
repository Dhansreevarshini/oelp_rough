from promise_level_classification import (
    PARTIES, SPEECH_CHUNKS_BASE, find_manifesto_chunks, video_of
)

print(f"{'Party':8} {'Manifesto chunks':>17} {'Speech chunks':>14} {'Speech videos':>14}")
for party in PARTIES:
    man = find_manifesto_chunks(party)
    folder = SPEECH_CHUNKS_BASE / party
    sp = sorted(folder.glob("*.txt")) if folder.is_dir() else []
    videos = {video_of(f.name) for f in sp}
    print(f"{party:8} {len(man):>17} {len(sp):>14} {len(videos):>14}")