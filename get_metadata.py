import json
import re
import subprocess


INPUT_FILE = "cleaned_videos_metadata.json"
OUTPUT_FILE = "videos_with_locations.json"


# --------------------------------------------------
# Places in Tamil Nadu
# --------------------------------------------------

PLACES = {
    "chennai": [
        "chennai",
        "சென்னை"
    ],

    "chengalpattu": [
        "chengalpattu",
        "செங்கல்பட்டு"
    ],

    "kancheepuram": [
        "kancheepuram",
        "காஞ்சிபுரம்"
    ],

    "tiruvallur": [
        "tiruvallur",
        "திருவள்ளூர்"
    ],

    "vellore": [
        "vellore",
        "வேலூர்"
    ],

    "ranipet": [
        "ranipet",
        "ராணிப்பேட்டை"
    ],

    "tirupattur": [
        "tirupattur",
        "திருப்பத்தூர்"
    ],

    "tiruvannamalai": [
        "tiruvannamalai",
        "திருவண்ணாமலை"
    ],

    "villupuram": [
        "villupuram",
        "விழுப்புரம்"
    ],

    "cuddalore": [
        "cuddalore",
        "கடலூர்"
    ],

    "salem": [
        "salem",
        "சேலம்"
    ],

    "namakkal": [
        "namakkal",
        "நாமக்கல்"
    ],

    "erode": [
        "erode",
        "ஈரோடு"
    ],

    "tiruppur": [
        "tiruppur",
        "திருப்பூர்"
    ],

    "coimbatore": [
        "coimbatore",
        "கோயம்புத்தூர்"
    ],

    "karur": [
        "karur",
        "கரூர்"
    ],

    "trichy": [
        "trichy",
        "tiruchirappalli",
        "திருச்சி",
        "திருச்சிராப்பள்ளி"
    ],

    "perambalur": [
        "perambalur",
        "பெரம்பலூர்"
    ],

    "ariyalur": [
        "ariyalur",
        "அரியலூர்"
    ],

    "thanjavur": [
        "thanjavur",
        "தஞ்சாவூர்"
    ],

    "nagapattinam": [
        "nagapattinam",
        "நாகப்பட்டினம்"
    ],

    "mayiladuthurai": [
        "mayiladuthurai",
        "மயிலாடுதுறை"
    ],

    "tiruvarur": [
        "tiruvarur",
        "thiruvarur",
        "திருவாரூர்"
    ],

    "pudukkottai": [
        "pudukkottai",
        "புதுக்கோட்டை"
    ],

    "sivaganga": [
        "sivaganga",
        "sivagangai",
        "சிவகங்கை"
    ],

    "madurai": [
        "madurai",
        "மதுரை"
    ],

    "theni": [
        "theni",
        "தேனி"
    ],

    "dindigul": [
        "dindigul",
        "திண்டுக்கல்"
    ],

    "virudhunagar": [
        "virudhunagar",
        "விருதுநகர்"
    ],

    "ramanathapuram": [
        "ramanathapuram",
        "ramnad",
        "ராமநாதபுரம்"
    ],

    "thoothukudi": [
        "thoothukudi",
        "tuticorin",
        "தூத்துக்குடி"
    ],

    "tirunelveli": [
        "tirunelveli",
        "திருநெல்வேலி"
    ],

    "tenkasi": [
        "tenkasi",
        "தென்காசி"
    ],

    "kanniyakumari": [
        "kanyakumari",
        "kanniyakumari",
        "கன்னியாகுமரி"
    ]
}


# More specific towns/constituencies
# These are important because a rally may happen
# in a town rather than being described by the district.

TOWNS = {
    "kangeyam": [
        "kangeyam",
        "காங்கேயம்"
    ],

    "modakurichi": [
        "modakurichi",
        "மொடக்குறிச்சி"
    ],

    "tiruchendur": [
        "tiruchendur",
        "திருச்செந்தூர்"
    ],

    "manamadurai": [
        "manamadurai",
        "மானாமதுரை"
    ],

    "thiruppuvanam": [
        "thiruppuvanam",
        "திருப்புவனம்"
    ],

    "palladam": [
        "palladam",
        "பல்லடம்"
    ],

    "chengalpattu": [
        "chengalpattu",
        "செங்கல்பட்டு"
    ],

    "vandavasi": [
        "vandavasi",
        "வந்தவாசி"
    ],

    "tirupattur": [
        "tirupattur",
        "திருப்பத்தூர்"
    ],

    "tenkasi": [
        "tenkasi",
        "தென்காசி"
    ],

    "karur": [
        "karur",
        "கரூர்"
    ],

    "tiruppur": [
        "tiruppur",
        "திருப்பூர்"
    ],

    "tirunelveli": [
        "tirunelveli",
        "திருநெல்வேலி"
    ],

    "virudhunagar": [
        "virudhunagar",
        "விருதுநகர்"
    ],

    "cuddalore": [
        "cuddalore",
        "கடலூர்"
    ]
}


# --------------------------------------------------
# Get fresh YouTube metadata
# --------------------------------------------------

def get_youtube_metadata(url):

    command = [
        "yt-dlp",
        "--skip-download",
        "--dump-single-json",
        "--no-warnings",
        url
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        return data
    except:
        return None


# --------------------------------------------------
# Find location in text
# --------------------------------------------------

def find_location(text, place_dictionary):

    if text is None:
        return None

    text = text.lower()

    found = []

    for place, names in place_dictionary.items():

        for name in names:

            name = name.lower()

            if name in text:
                found.append(place)
                break

    return found


# --------------------------------------------------
# Process one video
# --------------------------------------------------

def process_video(video):

    url = video["url"]

    print()
    print("Processing:")
    print(url)

    metadata = get_youtube_metadata(url)

    if metadata is None:

        print("Could not get metadata")

        video["detected_location"] = None
        video["location_source"] = "metadata unavailable"
        video["location_candidates"] = []

        return video


    title = metadata.get("title", "")
    description = metadata.get("description", "")

    print("Title:", title)


    # ----------------------------------------------
    # Search title first
    # ----------------------------------------------

    title_towns = find_location(title, TOWNS)
    title_districts = find_location(title, PLACES)


    if len(title_towns) > 0:

        location = title_towns[0]

        video["detected_location"] = location
        video["location_source"] = "youtube_title_town"
        video["location_candidates"] = title_towns

        print("Location:", location)
        print("Source: title")

        return video


    if len(title_districts) > 0:

        location = title_districts[0]

        video["detected_location"] = location
        video["location_source"] = "youtube_title"
        video["location_candidates"] = title_districts

        print("Location:", location)
        print("Source: title")

        return video


    # ----------------------------------------------
    # Search description
    # ----------------------------------------------

    description_towns = find_location(
        description,
        TOWNS
    )

    description_districts = find_location(
        description,
        PLACES
    )


    if len(description_towns) > 0:

        location = description_towns[0]

        video["detected_location"] = location
        video["location_source"] = "youtube_description_town"
        video["location_candidates"] = description_towns

        print("Location:", location)
        print("Source: description")

        return video


    if len(description_districts) > 0:

        location = description_districts[0]

        video["detected_location"] = location
        video["location_source"] = "youtube_description"
        video["location_candidates"] = description_districts

        print("Location:", location)
        print("Source: description")

        return video


    # ----------------------------------------------
    # Nothing found
    # ----------------------------------------------

    video["detected_location"] = None
    video["location_source"] = "not_found"
    video["location_candidates"] = []

    print("Location: NOT FOUND")

    return video


# --------------------------------------------------
# Main
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as file:

    videos = json.load(file)


results = []


for video in videos:

    result = process_video(video)

    results.append(result)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=4
    )


print()
print("Finished.")
print("Output:", OUTPUT_FILE)