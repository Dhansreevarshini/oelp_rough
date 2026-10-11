
import json
import yt_dlp

input_file = "zone_analysis.json"
output_file = "zone_analysis_with_duration.json"

with open(input_file, "r", encoding="utf-8") as file:
    videos = json.load(file)

options = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "extract_flat": False
}

with yt_dlp.YoutubeDL(options) as ydl:
    for video in videos:
        try:
            print("Checking:", video["url"])

            info = ydl.extract_info(
                video["url"],
                download=False
            )

            video["duration"] = info.get("duration")
            video["title"] = info.get("title", "")
            video["id"] = info.get("id", "")

            print("Duration:", video["duration"], "seconds")

        except Exception as error:
            print("Could not retrieve:", video["url"])
            print(error)

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(videos, file, ensure_ascii=False, indent=2)

print("Finished. Updated JSON saved to:", output_file)