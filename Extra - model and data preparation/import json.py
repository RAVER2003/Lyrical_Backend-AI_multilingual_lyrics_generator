import json
import os

folders = {
    "devotional": r"e:\indic\New devotional",
    "happy":      r"e:\indic\New happy",
    "party":      r"e:\indic\New party",
    "romantic":   r"e:\indic\New romantic",
    "sad":        r"e:\indic\New sad",
}

all_songs = []

for genre, folder_path in folders.items():
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".txt"):
            continue

        song_id = filename.replace(".txt", "").strip().replace(" ", "_")
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into verses by blank lines
        raw_verses = content.strip().split("\n\n")
        verses = []
        for verse in raw_verses:
            lines = [line.strip() for line in verse.strip().split("\n") if line.strip()]
            if lines:
                verses.append(lines)

        if not verses:
            continue

        all_songs.append({
            "song_id": song_id,
            "genre": genre,
            "verses": verses
        })

output_path = r"e:\indic\all_songs.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(all_songs, f, ensure_ascii=False, indent=2)

print(f"Done. {len(all_songs)} songs written to {output_path}")