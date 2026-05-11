import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.wkt import loads
import re

#Clean speed column
def clean_speed(df):
    df["maxspeed"] = df["maxspeed"].astype(str)

    df["maxspeed_clean"] = (
        df["maxspeed"]
        .str.lower()
        .str.replace("mph", "", regex=False)
        .str.strip()
    )

    df["maxspeed_clean"] = pd.to_numeric(df["maxspeed_clean"], errors="coerce")
    df["maxspeed_clean"] = df["maxspeed_clean"].fillna(60)
    df["maxspeed"] = df["maxspeed_clean"]
    df = df.drop(columns=["maxspeed_clean"])

    return df


#Clean tracks column
def fill_tracks(df):

    usage = df["usage"].fillna("").str.lower()
    
    def compute_tracks(row):
        if pd.notna(row["tracks"]) and str(row["tracks"]).strip() != "":
            return row["tracks"]
        
        if row["usage"] in ["branch", "industrial"]:
            return 1
        else:
            return 2
    
    df["tracks"] = df.apply(compute_tracks, axis=1)
    return df

#Clean electrification cilumn
def clean_elec(row):
    val = str(row['electrified']).lower()
    volt = str(row.get('voltage', '')).lower()
    
    if 'contact_line' in val or 'yes' in val or '25000' in volt:
        return 'Overhead'
    elif 'rail' in val and '4th' not in val:
        return '3rd_Rail'
    elif 'contact_line;rail' in val:
        return 'Dual_System'
    else:
        return 'No'
    
#Transform a dateframe into a geodataframe
def from_df_to_geodf(df, wkt_col='geometry_wkt', crs="EPSG:4326"):
    df = df.copy()
    df['geometry'] = df[wkt_col].apply(loads)
    return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)

#Keep the higher loading gauge and turn it into a number
#Example : some rows have a loading gauge like "W6, W8, W10" so the goal is to keep only W10
def max_gauge(val):
    if pd.isna(val) or val == '': return None
    numbers = re.findall(r'\d+', str(val))
    return max(map(int, numbers)) if numbers else None

#Find rail neighbours using location to apply the same loading gauge for missing values
def neighbour_gauge(gdf_rail, gdf_osm):

    #Clean to avoid errors
    cols_to_drop = ['index_right', 'index_left']
    gdf_rail = gdf_rail.drop(columns=[c for c in cols_to_drop if c in gdf_rail.columns])
    gauge_col = 'loading_gauge' if 'loading_gauge' in gdf_osm.columns else 'railway:loading_gauge'
    gdf_osm['loading_gauge_val'] = gdf_osm[gauge_col].apply(max_gauge)
    gdf_osm_clean = gdf_osm[['loading_gauge_val', 'geometry']].dropna(subset=['loading_gauge_val'])

    #Join the datasets
    joined = gpd.sjoin(gdf_rail, gdf_osm_clean, how='left', predicate='intersects')
    joined = joined[~joined.index.duplicated(keep='first')]

    #Propagation to the neighbours
    unknown = joined[joined['loading_gauge_val'].isna()].copy()
    known = joined[joined['loading_gauge_val'].notna()][['loading_gauge_val', 'geometry']]
    
    if not unknown.empty and not known.empty:
        unknown = unknown.drop(columns=[c for c in cols_to_drop if c in unknown.columns])
        
        #We create a buffer (a 1 meter zone) to ensure rail's connection
        unknown['geom_buffer'] = unknown.geometry.buffer(0.00001) 
        temp_unknown = gpd.GeoDataFrame(unknown, geometry='geom_buffer', crs=joined.crs)
        
        #We internaly join
        spatial_hit = gpd.sjoin(temp_unknown, known, how='left', predicate='intersects')
        
        #We take back the loading gauge value
        if 'loading_gauge_val_right' in spatial_hit.columns:
            propagated = spatial_hit.groupby(spatial_hit.index)['loading_gauge_val_right'].max()
            joined.loc[propagated.index, 'loading_gauge_val'] = joined.loc[propagated.index, 'loading_gauge_val'].fillna(propagated)
        
    return joined

#Decide the loading gauge based on voltage if no neighbours were found
def final_gauge(row):
    if pd.notna(row['loading_gauge_val']):
        return row['loading_gauge_val']
    
    volt = str(row.get('voltage', '')).lower()
    if '25000' in volt or '25kv' in volt: return 10
    if '750' in volt: return 7
    return 6

#Compute the distance between to points 
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

#Detect is components should be added or not based on multiple criteria
#Functions to be used in the final function important_component:


def component_bbox_miles(coords):
    lons = coords[:,0]
    lats = coords[:,1]
    min_lon, max_lon = lons.min(), lons.max()
    min_lat, max_lat = lats.min(), lats.max()

    # approx: 1° lat = 69 miles, 1° lon = 69 * cos(lat)
    lat_miles = (max_lat - min_lat) * 69
    lon_miles = (max_lon - min_lon) * 69 * np.cos(np.radians(coords[:,1].mean()))

    return lon_miles, lat_miles

#Minimum distance between two components
def min_distance_between_components(coords1, coords2, tree):

    min_dist = float('inf')

    for lon2, lat2 in coords2:
        #Closest point to the compoment
        dist, idx = tree.query([lon2, lat2], k=1)
        lon1, lat1 = coords1[idx]

        d = haversine_distance(lat1, lon1, lat2, lon2)

        if d < min_dist:
            min_dist = d

    return min_dist

#Checck if a compoment needs to be kept or not
def is_important_component(
        G, component, stations, main_coords, main_tree,
        min_line_length=2.0,
        max_small_size=0.3,
        near_main_threshold=0.1,
        reconnect_threshold=0.05
    ):
    
    #Extract coordinates
    coords = np.array([n for n in component if isinstance(n, tuple)])
    if len(coords) == 0:
        return False

    #Compute bounding boxes
    width_miles, height_miles = component_bbox_miles(coords)
    max_dim = max(width_miles, height_miles)

    #Long line = we keep it
    if max_dim >= min_line_length:
        return True

    #Contains 1 station at least = we keep it
    if any(n in stations for n in component):
        return True

    #Close to the main component = we keep it
    dist_to_main = min_distance_between_components(main_coords, coords, main_tree)
    if dist_to_main <= near_main_threshold:
        return True

    #Little and far component = we don't keep it
    if max_dim <= max_small_size and dist_to_main > near_main_threshold:
        return False

    #If small but really near to the main component = we keep it
    if dist_to_main <= reconnect_threshold:
        return True

    return False
