import json
import csv
import os

file_path = "metadata.json"
output_folder = "gender_trailersID"

os.makedirs(output_folder, exist_ok=True)

target_genres = ['action', 'adventure', 'comedy', 'crime', 'drama', 'fantasy', 'horror', 'romance']
genre_to_ids = {genre: set() for genre in target_genres}

with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

trailers = data['trailers12k']

for trailer_info in trailers.values():
    imdb_genres = trailer_info.get('imdb', {}).get('genres', [])
    youtube_trailers = trailer_info.get('youtube', {}).get('trailers', [])
    
    imdb_genres_lower = {genre.lower() for genre in imdb_genres}
    
    for yt_trailer in youtube_trailers:
        yt_id = yt_trailer.get('id')
        if not yt_id:
            continue
        for genre in target_genres:
            if genre in imdb_genres_lower and len(genre_to_ids[genre]) < 5:
                genre_to_ids[genre].add(yt_id)

# Convert sets to lists
for genre in genre_to_ids:
    genre_to_ids[genre] = list(genre_to_ids[genre])

# Save each genre IDs to separate CSV files inside the created folder
for genre, ids in genre_to_ids.items():
    filename = os.path.join(output_folder, f"{genre}_sampled.csv")
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([genre.capitalize()])  # Header
        for yt_id in ids:
            writer.writerow([yt_id])
    print(f"Saved {len(ids)} IDs to {filename}")

