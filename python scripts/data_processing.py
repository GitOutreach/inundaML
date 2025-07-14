# enhanced_soil_join.py
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
BASE_DIR = r"C:\Users\CoreG\Downloads\wss_aoi_2025-07-11_15-39-39"
SPATIAL_FILE = os.path.join(BASE_DIR, "spatial", "soilmu_a_aoi.shp")
TABULAR_DIR = os.path.join(BASE_DIR, "tabular")
OUTPUT_GEOJSON = "soil_with_attributes.geojson"
OUTPUT_CSV = "soil_with_attributes.csv"
OUTPUT_ML_CSV = "soil_ml_features.csv"

def validate_paths():
    """Validate that all required files exist."""
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")
    
    if not os.path.exists(SPATIAL_FILE):
        raise FileNotFoundError(f"Spatial file not found: {SPATIAL_FILE}")
    
    if not os.path.exists(TABULAR_DIR):
        raise FileNotFoundError(f"Tabular directory not found: {TABULAR_DIR}")
    
    required_files = ["mapunit.txt", "muaggatt.txt", "comp.txt"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(TABULAR_DIR, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Missing files: {missing_files}")
        # List available files
        available = [f for f in os.listdir(TABULAR_DIR) if f.endswith('.txt')]
        print(f"Available files: {available}")
    
    return missing_files

def read_ssurgo_safe(filename, usecols, col_names, required=True):
    """Safely read SSURGO file with error handling."""
    filepath = os.path.join(TABULAR_DIR, filename)
    
    if not os.path.exists(filepath):
        if required:
            raise FileNotFoundError(f"Required file not found: {filepath}")
        else:
            print(f"Optional file not found: {filepath}")
            return None
    
    try:
        # First, peek at the file to understand its structure
        with open(filepath, 'r', encoding='latin1') as f:
            first_line = f.readline()
            num_cols = len(first_line.split('|'))
            print(f"{filename}: {num_cols} columns detected")
        
        # Read the file
        df = pd.read_csv(filepath, sep="|", header=None, dtype=str, encoding="latin1")
        
        # Validate column indices
        max_col = max(usecols) if usecols else 0
        if max_col >= len(df.columns):
            print(f"Warning: Column index {max_col} out of range for {filename} (max: {len(df.columns)-1})")
            # Adjust usecols to available columns
            usecols = [col for col in usecols if col < len(df.columns)]
            col_names = col_names[:len(usecols)]
        
        if usecols:
            df = df.iloc[:, usecols]
        
        df.columns = col_names
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        print(f"Successfully read {filename}: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        if required:
            raise
        return None

def encode_categorical_features(df):
    """Convert categorical soil features to numerical values for ML."""
    
    # Drainage class encoding (better to worse drainage)
    drainage_map = {
        'excessively': 1,
        'somewhat excessively': 2,
        'well': 3,
        'moderately well': 4,
        'somewhat poorly': 5,
        'poorly': 6,
        'very poorly': 7
    }
    
    # Hydrologic group encoding (A=best drainage, D=worst)
    hydgrp_map = {
        'A': 1,
        'B': 2,
        'C': 3,
        'D': 4,
        'A/D': 2.5,  # dual groups get intermediate values
        'B/D': 3.5,
        'C/D': 3.5
    }
    
    # Flood frequency encoding
    flood_map = {
        'none': 0,
        'very rare': 1,
        'rare': 2,
        'occasional': 3,
        'frequent': 4,
        'very frequent': 5
    }
    
    # Runoff encoding
    runoff_map = {
        'negligible': 1,
        'very low': 2,
        'low': 3,
        'medium': 4,
        'high': 5,
        'very high': 6
    }
    
    # Apply encodings
    if 'drainagecl' in df.columns:
        df['drainage_encoded'] = df['drainagecl'].str.lower().map(drainage_map)
        df['drainage_encoded'] = df['drainage_encoded'].fillna(4)  # default to moderate
    
    if 'hydgrp' in df.columns:
        df['hydgrp_encoded'] = df['hydgrp'].str.upper().map(hydgrp_map)
        df['hydgrp_encoded'] = df['hydgrp_encoded'].fillna(2.5)  # default to B/C
    
    if 'flodfreqdcd' in df.columns:
        df['flood_freq_encoded'] = df['flodfreqdcd'].str.lower().map(flood_map)
        df['flood_freq_encoded'] = df['flood_freq_encoded'].fillna(0)  # default to none
    
    if 'runoff' in df.columns:
        df['runoff_encoded'] = df['runoff'].str.lower().map(runoff_map)
        df['runoff_encoded'] = df['runoff_encoded'].fillna(3)  # default to medium
    
    return df

def calculate_flood_risk_score(df):
    """Calculate a composite flood risk score from soil attributes."""
    
    # Initialize risk score
    df['flood_risk_score'] = 0
    
    # Add components (higher score = higher risk)
    if 'drainage_encoded' in df.columns:
        df['flood_risk_score'] += df['drainage_encoded'] * 0.3
    
    if 'hydgrp_encoded' in df.columns:
        df['flood_risk_score'] += df['hydgrp_encoded'] * 0.25
    
    if 'flood_freq_encoded' in df.columns:
        df['flood_risk_score'] += df['flood_freq_encoded'] * 0.3
    
    if 'runoff_encoded' in df.columns:
        df['flood_risk_score'] += df['runoff_encoded'] * 0.15
    
    # Normalize to 0-10 scale
    if df['flood_risk_score'].max() > 0:
        df['flood_risk_score'] = (df['flood_risk_score'] / df['flood_risk_score'].max()) * 10
    
    return df

def main():
    """Main processing function."""
    
    print("=== Enhanced Soil Data Processing ===")
    
    # Validate paths
    missing_files = validate_paths()
    
    # ------------------------------------------------------------------
    # 1. Load spatial data
    # ------------------------------------------------------------------
    print("\n1. Loading spatial data...")
    gdf = gpd.read_file(SPATIAL_FILE)
    print(f"Loaded {len(gdf)} soil polygons.")
    
    # Convert MUKEY to string and find the key column
    if 'MUKEY' in gdf.columns:
        key_col = 'MUKEY'
    elif 'mukey' in gdf.columns:
        key_col = 'mukey'
    else:
        # Find key column case-insensitively
        key_col = next((c for c in gdf.columns if c.upper() == "MUKEY"), None)
        if not key_col:
            raise ValueError("MUKEY column not found in spatial data")
    
    gdf[key_col] = gdf[key_col].astype(str)
    print(f"Using spatial key column: {key_col}")
    
    # ------------------------------------------------------------------
    # 2. Load tabular data
    # ------------------------------------------------------------------
    print("\n2. Loading tabular data...")
    
    # mapunit.txt - basic unit information
    mu = read_ssurgo_safe("mapunit.txt",
                          usecols=[0, 1, 2],
                          col_names=["mukey", "musym", "muname"])
    
    # muaggatt.txt - aggregated attributes
    agg = read_ssurgo_safe("muaggatt.txt",
                           usecols=[39, 35, 36, 34],
                           col_names=["mukey", "drainagecl", "hydgrp", "flodfreqdcd"])
    
    # comp.txt - component information
    comp = read_ssurgo_safe("comp.txt",
                            usecols=[23, 4, 1, 14],
                            col_names=["mukey", "compname", "comppct_r", "runoff"])
    
    # ------------------------------------------------------------------
    # 3. Process and merge tabular data
    # ------------------------------------------------------------------
    print("\n3. Processing tabular data...")
    
    # Start with mapunit data
    soil = mu.copy()
    
    # Merge additional tables
    if agg is not None:
        soil = pd.merge(soil, agg, on="mukey", how="left")
    
    if comp is not None:
        # For components, we might have multiple per mapunit
        # Take the dominant component (highest percentage)
        if 'comppct_r' in comp.columns:
            comp['comppct_r'] = pd.to_numeric(comp['comppct_r'], errors='coerce')
            comp = comp.sort_values(['mukey', 'comppct_r'], ascending=[True, False])
            comp = comp.groupby('mukey').first().reset_index()
        
        soil = pd.merge(soil, comp, on="mukey", how="left")
    
    # ------------------------------------------------------------------
    # 4. Encode categorical features for ML
    # ------------------------------------------------------------------
    print("\n4. Encoding features for ML...")
    
    soil = encode_categorical_features(soil)
    soil = calculate_flood_risk_score(soil)
    
    # ------------------------------------------------------------------
    # 5. Merge with spatial data
    # ------------------------------------------------------------------
    print("\n5. Merging with spatial data...")
    
    soil_gdf = gdf.merge(soil, left_on=key_col, right_on="mukey", how="left")
    
    # ------------------------------------------------------------------
    # 6. Prepare ML features
    # ------------------------------------------------------------------
    print("\n6. Preparing ML features...")
    
    # Select numerical features for ML
    ml_features = ['mukey']
    
    # Add encoded features
    encoded_cols = [col for col in soil_gdf.columns if col.endswith('_encoded')]
    ml_features.extend(encoded_cols)
    
    # Add flood risk score
    if 'flood_risk_score' in soil_gdf.columns:
        ml_features.append('flood_risk_score')
    
    # Add component percentage if available
    if 'comppct_r' in soil_gdf.columns:
        soil_gdf['comppct_r'] = pd.to_numeric(soil_gdf['comppct_r'], errors='coerce')
        ml_features.append('comppct_r')
    
    # Create ML dataset
    ml_df = soil_gdf[ml_features].copy()
    
    # Fill NaN values with appropriate defaults
    for col in ml_df.columns:
        if col != 'mukey':
            ml_df[col] = ml_df[col].fillna(ml_df[col].median() if ml_df[col].dtype in ['float64', 'int64'] else 0)
    
    # ------------------------------------------------------------------
    # 7. Export results
    # ------------------------------------------------------------------
    print("\n7. Exporting results...")
    
    # Export full GeoJSON
    soil_gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    print(f"GeoJSON saved: {OUTPUT_GEOJSON}")
    
    # Export CSV (without geometry)
    soil_gdf.drop(columns=["geometry"]).to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved: {OUTPUT_CSV}")
    
    # Export ML features
    ml_df.to_csv(OUTPUT_ML_CSV, index=False)
    print(f"ML features saved: {OUTPUT_ML_CSV}")
    
    # ------------------------------------------------------------------
    # 8. Summary statistics
    # ------------------------------------------------------------------
    print("\n8. Summary Statistics:")
    print(f"Total soil polygons: {len(soil_gdf)}")
    print(f"Unique soil types: {soil_gdf['musym'].nunique() if 'musym' in soil_gdf.columns else 'N/A'}")
    
    if 'flood_risk_score' in soil_gdf.columns:
        print(f"Average flood risk score: {soil_gdf['flood_risk_score'].mean():.2f}")
        print(f"Max flood risk score: {soil_gdf['flood_risk_score'].max():.2f}")
    
    print("\nML Features available:")
    for col in ml_features:
        if col != 'mukey':
            print(f"  - {col}")
    
    print("\nData processing complete!")
    
    return soil_gdf, ml_df

if __name__ == "__main__":
    try:
        soil_gdf, ml_df = main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
