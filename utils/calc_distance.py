import pandas as pd
import math

def add_nearest_mrt(geocoded_df):
    mrt_df = pd.read_csv('data/mrt.csv')

    # Haversine distance function (in kilometers)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        
        a = math.sin(delta_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    # Pre-compute MRT coordinates
    mrt_coords = list(zip(mrt_df['lat'], mrt_df['lng']))

    # Compute nearest MRT distance per row
    def compute_nearest_mrt(lat, lon):
        return min(haversine(lat, lon, mrt_lat, mrt_lon) for mrt_lat, mrt_lon in mrt_coords)

    geocoded_df['distance_to_nearest_mrt'] = geocoded_df.apply(
        lambda row: compute_nearest_mrt(row['latitude'], row['longitude']),
        axis=1
    )

    return geocoded_df