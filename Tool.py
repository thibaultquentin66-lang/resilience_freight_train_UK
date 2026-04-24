import pandas as pd
import numpy as np



#Clean column speed
def clean_speed(df):
    # Convertir en string pour éviter les erreurs
    df["maxspeed"] = df["maxspeed"].astype(str)

    # Nettoyer : enlever "mph", espaces, etc.
    df["maxspeed_clean"] = (
        df["maxspeed"]
        .str.lower()
        .str.replace("mph", "", regex=False)
        .str.strip()
    )

    # Convertir en numérique (les valeurs impossibles deviennent NaN)
    df["maxspeed_clean"] = pd.to_numeric(df["maxspeed_clean"], errors="coerce")

    # Remplir les NaN avec 60 mph
    df["maxspeed_clean"] = df["maxspeed_clean"].fillna(60)

    # Remplacer l’ancienne colonne
    df["maxspeed"] = df["maxspeed_clean"]

    # Supprimer la colonne temporaire
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