import pandas as pd
import time
import requests
from tqdm import tqdm
import os

# === CONFIGURATION ===
INPUT_CSV = "output/geocoded_data.csv"
BACKUP_DIR = "output/backups"
SAVE_EVERY = 100  # Save every N updates

os.makedirs(BACKUP_DIR, exist_ok=True)

def get_location_data(search_query):
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={search_query}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    headers = {"Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vaW50ZXJuYWwtYWxiLW9tLXByZGV6aXQtaXQtbmV3LTE2MzM3OTk1NDIuYXAtc291dGhlYXN0LTEuZWxiLmFtYXpvbmF3cy5jb20vYXBpL3YyL3VzZXIvcGFzc3dvcmQiLCJpYXQiOjE3NDM2NjQ5MzgsImV4cCI6MTc0MzkyNDEzOCwibmJmIjoxNzQzNjY0OTM4LCJqdGkiOiJuTHU0ZHpIRk1CNGR4d0c5Iiwic3ViIjoiMmE1MjcyM2RiZGQxMzA1OWQ4NjYxNGUzMmJhNzcxZTkiLCJ1c2VyX2lkIjo2Njc3LCJmb3JldmVyIjpmYWxzZX0.FJNzVzrKIsYbS137XrOhdk74Qnxist_SKw2u93P0U6M"}

    while True:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 429:
                print("Rate limited. Sleeping for 5 seconds...")
                time.sleep(5)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}. Retrying in 5 seconds...")
            time.sleep(5)

def geocode_address(block, street, town):
    search_query = f"{block} {street} {town}"
    result = get_location_data(search_query)
    if result and result['results']:
        first_result = result['results'][0]
        return (
            first_result.get('LATITUDE'),
            first_result.get('LONGITUDE'),
            first_result.get('BUILDING'),
            first_result.get('ADDRESS')
        )
    return None, None, None, None

def main():
    df = pd.read_csv(INPUT_CSV)

    # Get rows that need geocoding
    missing_coords = df[df['latitude'].isna() | df['longitude'].isna()]
    print(f"Rows needing geocoding: {len(missing_coords)}")

    # Counter for when to save
    updated_count = 0

    # Loop with progress bar
    for idx in tqdm(missing_coords.index, desc="Geocoding"):
        row = df.loc[idx]
        blk, street, town = row['block'], row['street_name'], row['town']
        lat, lng, building, full_addr = geocode_address(blk, street, town)

        if lat and lng:
            df.at[idx, 'latitude'] = lat
            df.at[idx, 'longitude'] = lng
            if 'building' in df.columns:
                df.at[idx, 'building'] = building
            if 'address' in df.columns:
                df.at[idx, 'address'] = full_addr
        else:
            print(f"Could not geocode: {blk} {street} {town}")

        updated_count += 1
        if updated_count % SAVE_EVERY == 0:
            # Save intermediate backup
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(BACKUP_DIR, f"geocoded_backup_{timestamp}.csv")
            df.to_csv(backup_path, index=False)
            print(f"Saved backup: {backup_path}")

    df.to_csv(INPUT_CSV, index=False)
    print("Geocoding complete. Final CSV saved.")

main()