
import os

base_folder = "chunks_speech"

video_ids = [
    "LI-Qjk5zAHA",
    "qBXEjDhB4ng",
    "01ZulxQ8Ffo",
    "ngRHXOc5lF4"
]

files_to_delete = []

for party in os.listdir(base_folder):
    party_folder = os.path.join(base_folder, party)

    if not os.path.isdir(party_folder):
        continue

    for filename in os.listdir(party_folder):
        for video_id in video_ids:
            if video_id in filename:
                files_to_delete.append(
                    os.path.join(party_folder, filename)
                )
                break

print("Files matching the selected video IDs:")

for filepath in files_to_delete:
    print(filepath)

print("\nTotal files found:", len(files_to_delete))


confirm = input(
    "\nType DELETE to permanently delete these files: "
)

if confirm == "DELETE":
    for filepath in files_to_delete:
        os.remove(filepath)

    print("Deleted", len(files_to_delete), "files.")
else:
    print("Deletion cancelled.")
