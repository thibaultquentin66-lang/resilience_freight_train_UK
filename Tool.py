import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.wkt import loads
import re
import random
import networkx as nx
import ast
from scipy.spatial import KDTree

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
        near_main_threshold=0.1
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

    return False

#Analyse resilience through a random and targeted nodes removal
def resilience_node(G, max_retrait=120, pas=5):

    N_total = len(G.nodes())
    closeness_dict = nx.closeness_centrality(G, distance='length')
    noeuds_cibles = [n for n, score in sorted(closeness_dict.items(), key=lambda x: x[1], reverse=True)]
    

    noeuds_aleatoires = list(G.nodes())
    random.shuffle(noeuds_aleatoires)
    
    lcc_random = [100.0]
    lcc_targeted = [100.0]
    X_noeuds = [0]
    
    #Random
    G_rand = G.copy()
    for i in range(0, max_retrait, pas):
        noeuds_a_retirer = noeuds_aleatoires[i:i+pas]
        G_rand.remove_nodes_from(noeuds_a_retirer)
        if len(G_rand) > 0:
            taille_lcc = len(max(nx.connected_components(G_rand), key=len)) / N_total * 100
            lcc_random.append(taille_lcc)
        else:
            lcc_random.append(0)
            
    #Targeted (closeness)
    G_target = G.copy()
    for i in range(0, max_retrait, pas):
        noeuds_a_retirer = noeuds_cibles[i:i+pas]
        G_target.remove_nodes_from(noeuds_a_retirer)
        if len(G_target) > 0:
            taille_lcc = len(max(nx.connected_components(G_target), key=len)) / N_total * 100
            lcc_targeted.append(taille_lcc)
        else:
            lcc_targeted.append(0)
            
        X_noeuds.append(i + pas)
        
    return X_noeuds, lcc_random, lcc_targeted

#Analyse resilience through a random and targeted nodes removal
def resilience_edge(G, max_retrait_edges=500, pas=20):

    N_total_nodes = len(G.nodes())
    closeness_dict = nx.closeness_centrality(G, distance='length')
    
    edges_scores = {}
    for u, v in G.edges():
        score_edge = closeness_dict.get(u, 0) + closeness_dict.get(v, 0)
        edges_scores[(u, v)] = score_edge
        
    edges_cibles = [e for e, score in sorted(edges_scores.items(), key=lambda x: x[1], reverse=True)]
    
    edges_aleatoires = list(G.edges())
    random.shuffle(edges_aleatoires)
    
    lcc_random = [100.0]
    lcc_targeted = [100.0]
    X_edges = [0]
    
    G_rand = G.copy()
    for i in range(0, max_retrait_edges, pas):
        edges_a_retirer = edges_aleatoires[i:i+pas]
        G_rand.remove_edges_from(edges_a_retirer)
        if len(G_rand) > 0:
            # On mesure toujours la taille de la composante géante de nœuds restants connectés
            taille_lcc = len(max(nx.connected_components(G_rand), key=len)) / N_total_nodes * 100
            lcc_random.append(taille_lcc)
        else:
            lcc_random.append(0)
            
    # Simulation Ciblée (Targeted Edge Removal)
    G_target = G.copy()
    for i in range(0, max_retrait_edges, pas):
        edges_a_retirer = edges_cibles[i:i+pas]
        G_target.remove_edges_from(edges_a_retirer)
        if len(G_target) > 0:
            taille_lcc = len(max(nx.connected_components(G_target), key=len)) / N_total_nodes * 100
            lcc_targeted.append(taille_lcc)
        else:
            lcc_targeted.append(0)
            
        X_edges.append(i + pas)
        
    return X_edges, lcc_random, lcc_targeted

#Fin the closest node in a graph
def closest_node(graphe, coord_cible):
    def extract_coord(n):
        if isinstance(n, (tuple, list)) and len(n) >= 2:
            return float(n[0]), float(n[1])
        elif isinstance(n, str) and n.startswith('(') and n.endswith(')'):
            try:
                t = ast.literal_eval(n)
                return float(t[0]), float(t[1])
            except:
                pass

        data = graphe.nodes[n]
        return float(data.get('lon', 0)), float(data.get('lat', 0))

    return min(
        graphe.nodes(), 
        key=lambda n: (extract_coord(n)[0] - coord_cible[0])**2 + (extract_coord(n)[1] - coord_cible[1])**2
    )


#Compute travel distance and time variations between baseline and alternative routes with loading gauge constraints
def calculate_travel_cost(graph, source, target, max_extra_stops=25):
    if not nx.has_path(graph, source, target):
        return None
        
    try:
        # 1. Itinéraire de référence (Baseline)
        main_path = nx.shortest_path(graph, source, target, weight='distance')
        main_stops = len(main_path) - 1
        
        gauges_on_path = []
        main_dist = 0
        main_time_min = 0
        
        # Parcours manuel de la baseline pour éviter nx.path_weight
        for u, v in zip(main_path[:-1], main_path[1:]):
            edge_data = graph[u][v]
            if isinstance(edge_data, dict) and 0 in edge_data: 
                edge_data = edge_data[0]
            elif hasattr(graph, 'is_multigraph') and graph.is_multigraph():
                edge_data = edge_data[list(edge_data.keys())[0]]
            
            g = edge_data.get('loading_gauge', 0)
            if g > 0: 
                gauges_on_path.append(g)
            
            dist = edge_data.get('distance', 0)
            speed = edge_data.get('maxspeed', 60.0)
            if speed <= 0: 
                speed = 60.0
                
            main_dist += dist
            main_time_min += (dist / speed) * 60
            
        required_gauge = min(gauges_on_path) if gauges_on_path else 0.0
        
        # 2. Construction du graphe compatible
        G_compatible = nx.MultiGraph() if graph.is_multigraph() else nx.Graph()
        G_compatible.add_nodes_from(graph.nodes(data=True))
        
        for u, v, k, d in graph.edges(keys=True, data=True) if graph.is_multigraph() else ((u,v,0,d) for u,v,d in graph.edges(data=True)):
            edge_gauge = d.get('loading_gauge', 0)
            if required_gauge == 0.0 or edge_gauge == 0 or edge_gauge >= required_gauge:
                if graph.is_multigraph():
                    G_compatible.add_edge(u, v, key=k, **d)
                else:
                    G_compatible.add_edge(u, v, **d)
                    
        # 3. Simulation de la coupure (section centrale)
        if len(main_path) > 2:
            mid = len(main_path) // 2
            u_cut, v_cut = main_path[mid], main_path[mid+1]
        else:
            u_cut, v_cut = main_path[0], main_path[1]
            
        if G_compatible.has_edge(u_cut, v_cut):
            if G_compatible.is_multigraph():
                keys = list(G_compatible[u_cut][v_cut].keys())
                for k in keys:
                    G_compatible.remove_edge(u_cut, v_cut, key=k)
            else:
                G_compatible.remove_edge(u_cut, v_cut)
                
        # 4. Calcul de l'itinéraire alternatif
        if nx.has_path(G_compatible, source, target):
            alt_path = nx.shortest_path(G_compatible, source, target, weight='distance')
            alt_stops = len(alt_path) - 1
            
            if alt_stops <= (main_stops + max_extra_stops):
                alt_dist = 0
                alt_time_min = 0
                
                # Parcours manuel de l'alternative
                for u, v in zip(alt_path[:-1], alt_path[1:]):
                    e_data = G_compatible[u][v]
                    if isinstance(e_data, dict) and 0 in e_data: 
                        e_data = e_data[0]
                    elif hasattr(G_compatible, 'is_multigraph') and G_compatible.is_multigraph():
                        e_data = e_data[list(e_data.keys())[0]]
                        
                    d_miles = e_data.get('distance', 0)
                    v_mph = e_data.get('maxspeed', 60.0)
                    if v_mph <= 0: 
                        v_mph = 60.0
                        
                    alt_dist += d_miles
                    alt_time_min += (d_miles / v_mph) * 60
                    
                extra_dist = max(0.0, alt_dist - main_dist)
                extra_time = max(0.0, alt_time_min - main_time_min)
                
                return {'Extra_Miles': extra_dist, 'Extra_Minutes': extra_time}
                
        return None
    except Exception as e:
        print(f"Erreur résiduelle bloquante : {e}")
        return None

#Generate a copy of a graph with a disruption applied
def generate_disrupted_graph(input_graph, mode="flow"):
    disrupted_graph = input_graph.copy()
    
    if mode == "flow":
        try:
            max_flow = -1
            selected_edge = None
            
            if input_graph.is_multigraph():
                for u, v, key, data in input_graph.edges(keys=True, data=True):
                    current_flow = data.get('traffic_flow', 0)
                    if current_flow > max_flow:
                        max_flow = current_flow
                        selected_edge = (u, v, key)
            else:
                for u, v, data in input_graph.edges(data=True):
                    current_flow = data.get('traffic_flow', 0)
                    if current_flow > max_flow:
                        max_flow = current_flow
                        selected_edge = (u, v, None)
                    
            if selected_edge:
                node_A, node_B, edge_key = selected_edge
                name_A = input_graph.nodes[node_A].get('name', f"STANOX: {input_graph.nodes[node_A].get('stanox')}")
                name_B = input_graph.nodes[node_B].get('name', f"STANOX: {input_graph.nodes[node_B].get('stanox')}")
                
                print(f"Critical link: {name_A} <--> {name_B}")
                print(f"Traffic Volume removed: {max_flow}")
                
                if edge_key is not None:
                    disrupted_graph.remove_edge(node_A, node_B, key=edge_key)
                else:
                    disrupted_graph.remove_edge(node_A, node_B)
            else:
                print("No flow detected")
                
        except Exception as e:
            print("Error during max flow edge removal:", e)
            
    elif mode == "random":
        try:
            toutes_les_aretes = list(input_graph.edges(keys=True if input_graph.is_multigraph() else False))
            
            if toutes_les_aretes:
                edge_aleatoire = random.choice(toutes_les_aretes)
                
                if len(edge_aleatoire) == 3:
                    u, v, key = edge_aleatoire
                    disrupted_graph.remove_edge(u, v, key=key)
                else:
                    u, v = edge_aleatoire
                    disrupted_graph.remove_edge(u, v)
                    
                name_u = input_graph.nodes[u].get('name', u)
                name_v = input_graph.nodes[v].get('name', v)
                print(f"Random link removed: {name_u} <--> {name_v}")
            else:
                print("No edges to remove")
                
        except Exception as e:
            print("Error during random edge removal:", e)

    return disrupted_graph

#Generate a copy of a graph with a disruption applied
def create_disrupted_graph(G, edges_to_remove):
    G_disrupted = G.copy()
    
    for u, v in edges_to_remove:
        if G_disrupted.has_edge(u, v):
            G_disrupted.remove_edge(u, v)
            
    return G_disrupted

#Tag nodes and associate STANOX code
def tag_graph_with_kdtree(G_target, stations_ref, max_distance_degrees=0.002):
    target_nodes = list(G_target.nodes())
    target_coords = []
    
    for node in target_nodes:
        data = G_target.nodes[node]
        if isinstance(node, (tuple, list)) and len(node) >= 2 and isinstance(node[0], (int, float)):
            lon, lat = node[0], node[1]
        elif 'lon' in data and 'lat' in data:
            lon, lat = float(data['lon']), float(data['lat'])
        else:
            lon, lat = 0.0, 0.0
        target_coords.append([lon, lat])
        
    tree = KDTree(target_coords)
    
    for node in G_target.nodes():
        G_target.nodes[node]['stanox'] = None
        
    tagged_count = 0
    for stat in stations_ref:
        ref_coord = [stat['lon'], stat['lat']]
        dist, idx = tree.query(ref_coord)
        
        if dist <= max_distance_degrees:
            target_node = target_nodes[idx]
            G_target.nodes[target_node]['stanox'] = stat['stanox']
            tagged_count += 1
            
    return tagged_count


#Remove edges with loading gauge less than 10
def apply_gauge_10_restriction(G_disrupted):
    """
    Takes a disrupted graph and removes all tracks where the loading_gauge is less than 10.
    """
    G_gauge_10 = G_disrupted.copy()
    edges_to_remove_gauge = []
    is_multigraph = G_gauge_10.is_multigraph()

    if is_multigraph:
        for u, v, key, edge_data in G_gauge_10.edges(keys=True, data=True):
            gauge_voie = edge_data.get('loading_gauge', None)
            # FIX: Using the correct variable name 'gauge_voie' here
            if gauge_voie is None or float(gauge_voie) < 10.0:
                edges_to_remove_gauge.append((u, v, key))
    else:
        for u, v, edge_data in G_gauge_10.edges(data=True):
            gauge_voie = edge_data.get('loading_gauge', None)
            if gauge_voie is None or float(gauge_voie) < 10.0:
                edges_to_remove_gauge.append((u, v))

    G_gauge_10.remove_edges_from(edges_to_remove_gauge)
    print(f"🚊 Gauge 10 Scenario: Removed tracks with gauge < 10.0.")
    return G_gauge_10

#Remove edges from a graph based on the bridge strikes results
def remove_bridge_edges(G, indices, disrupted_edges):
    

    G_disrupted = G.copy()


    for idx in indices:


        if idx >= len(disrupted_edges):
            continue


        edge = disrupted_edges[idx]


        try:


            if G_disrupted.is_multigraph() and len(edge) >= 3:


                G_disrupted.remove_edge(
                    edge[0],
                    edge[1],
                    key=edge[2]
                )


            else:


                G_disrupted.remove_edge(
                    edge[0],
                    edge[1]
                )


        except (
            nx.NetworkXError,
            KeyError
        ):

            pass


    return G_disrupted

#Say if a node can be considerate coastal with arbitrary values
def is_coastal_node(node):
    
    if not (isinstance(node, tuple) and len(node)==2):
        return False

    lon, lat = node

    if abs(lon)<0.001 and abs(lat)<0.001:
        return False

    return (
        lon > 1.0
        or lon < -3.0
        or lat < 51.0
        or lat > 55.0
    )

    is_far_east = lon > 1.3
    is_far_west = lon < -3.2
    is_far_south = lat < 50.9
    is_far_north = lat > 55.8
    
    return (is_far_east or is_far_west or is_far_south or is_far_north)

#Remove edges based and execute the flow scenario
def run_flow_disruption(scenario_name, edges_to_remove, G_final, G_gauge_6_initial, G_gauge_10_initial, trajets_trust_uniques):
    
    G_6_disc = G_final.copy()

    for edge in edges_to_remove:
        try:
            G_6_disc.remove_edge(edge[0], edge[1])
        except:
            pass

    G_10_disc = G_gauge_10_initial.copy()

    for edge in edges_to_remove:
        try:
            G_10_disc.remove_edge(edge[0], edge[1])
        except:
            pass

    metrics = {
        "Gauge 6": {
            "ok": 0,
            "reroutes": 0,
            "blocked": 0,
            "dist": 0,
            "time": 0
        },
        "Gauge 10": {
            "ok": 0,
            "reroutes": 0,
            "blocked": 0,
            "dist": 0,
            "time": 0
        }
    }

    for travel_id, (origin, destination) in trajets_trust_uniques.items():

        #Loading gauge 6
        try:
            d0_6 = nx.shortest_path_length(G_gauge_6_initial, origin, destination, weight="distance")
            t0_6 = nx.shortest_path_length(G_gauge_6_initial, origin, destination, weight="temps_minutes")
        except:
            continue

        try:
            d = nx.shortest_path_length(G_6_disc, origin, destination, weight="distance")
            t = nx.shortest_path_length(G_6_disc, origin, destination, weight="temps_minutes")
            if d - d0_6 <= 0.001:
                metrics["Gauge 6"]["ok"] += 1
            else:
                metrics["Gauge 6"]["reroutes"] += 1
                metrics["Gauge 6"]["dist"] += (d - d0_6)
                metrics["Gauge 6"]["time"] += (t - t0_6)
        except:
            metrics["Gauge 6"]["blocked"] += 1

        #Loading gauge 10
        try:
            d0_10 = nx.shortest_path_length(G_gauge_10_initial, origin, destination, weight="distance")
            t0_10 = nx.shortest_path_length(G_gauge_10_initial, origin, destination, weight="temps_minutes")
        except:
            continue

        try:
            d = nx.shortest_path_length(G_10_disc, origin, destination, weight="distance")
            t = nx.shortest_path_length(G_10_disc, origin, destination, weight="temps_minutes")
            if d - d0_10 <= 0.001:
                metrics["Gauge 10"]["ok"] += 1
            else:
                metrics["Gauge 10"]["reroutes"] += 1
                metrics["Gauge 10"]["dist"] += (d - d0_10)
                metrics["Gauge 10"]["time"] += (t - t0_10)
        except:
            metrics["Gauge 10"]["blocked"] += 1

    #Save
    rows = []

    for gauge in ["Gauge 6", "Gauge 10"]:
        total = (
            metrics[gauge]["ok"]
            + metrics[gauge]["reroutes"]
            + metrics[gauge]["blocked"]
        )

        reroute_number = max(1, metrics[gauge]["reroutes"])

        rows.append({
            "Scenario": scenario_name,
            "Gauge": gauge,
            "Intact Trips": f'{metrics[gauge]["ok"]} ({metrics[gauge]["ok"]/total*100:.1f}%)',
            "Rerouted Trips": f'{metrics[gauge]["reroutes"]} ({metrics[gauge]["reroutes"]/total*100:.1f}%)',
            "Blocked Trips": f'{metrics[gauge]["blocked"]} ({metrics[gauge]["blocked"]/total*100:.1f}%)',
            "Avg Detour (mi)": f'{metrics[gauge]["dist"]/reroute_number:.2f}',
            "Avg Delay (min)": f'{metrics[gauge]["time"]/reroute_number:.1f}'
        })

    return rows