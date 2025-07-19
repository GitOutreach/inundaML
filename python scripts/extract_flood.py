import sqlite3
import pandas as pd
import geopandas as gpd
import os

# -------- CONFIG --------
BASE_DIR = r"C:\NewSSURGODatabase_gpkg"  # Update this path
GPKG_FILE = os.path.join(BASE_DIR, "NewSSURGODatabase.gpkg")
OUTPUT_CSV = "soil_flooding_data.csv"
OUTPUT_SHP = "soil_flooding_polygons.shp"

# -------- 1. Connect to SSURGO Database --------
print(f"Connecting to SSURGO database: {GPKG_FILE}")
print(f"Database exists: {os.path.exists(GPKG_FILE)}")

# Connect to the GeoPackage database
conn = sqlite3.connect(GPKG_FILE)

# -------- 2. Explore Database Structure --------
print("\n=== Database Tables ===")
tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(tables_query, conn)
print("Available tables:")
for table in tables['name']:
    print(f"  - {table}")

# -------- 3. Load Flooding Rating Table --------
print("\n=== Loading Flooding Rating Table ===")
try:
    flooding_query = "SELECT * FROM rating_FloodFCls_DCD_jan_dec LIMIT 10;"
    flooding_sample = pd.read_sql_query(flooding_query, conn)
    print("Flooding rating table columns:")
    print(flooding_sample.columns.tolist())
    print("\nSample flooding data:")
    print(flooding_sample.head())
    
    # Load full flooding data
    flooding_full_query = "SELECT * FROM rating_FloodFCls_DCD_jan_dec;"
    flooding_data = pd.read_sql_query(flooding_full_query, conn)
    print(f"\nTotal flooding records: {len(flooding_data)}")
    
except Exception as e:
    print(f"Error loading flooding table: {e}")
    flooding_data = None

# -------- 4. Load Spatial Data (mupolygon) --------
print("\n=== Loading Spatial Data ===")
try:
    # Load the mupolygon table as a GeoDataFrame
    spatial_gdf = gpd.read_file(GPKG_FILE, layer='mupolygon')
    print(f"Spatial data loaded: {len(spatial_gdf)} polygons")
    print("Spatial columns:")
    print(spatial_gdf.columns.tolist())
    print("\nSample spatial data:")
    print(spatial_gdf[['mukey', 'areasymbol', 'musym']].head())
    
except Exception as e:
    print(f"Error loading spatial data: {e}")
    spatial_gdf = None

# -------- 5. Join Flooding Data with Spatial Data --------
if flooding_data is not None and spatial_gdf is not None:
    print("\n=== Joining Flooding and Spatial Data ===")
    
    # Check mukey column types and convert if needed
    print(f"Flooding mukey dtype: {flooding_data['mukey'].dtype}")
    print(f"Spatial mukey dtype: {spatial_gdf['mukey'].dtype}")
    
    # Ensure mukey columns are the same type
    flooding_data['mukey'] = flooding_data['mukey'].astype(str)
    spatial_gdf['mukey'] = spatial_gdf['mukey'].astype(str)
    
    # Check for common mukeys
    flooding_keys = set(flooding_data['mukey'])
    spatial_keys = set(spatial_gdf['mukey'])
    common_keys = flooding_keys & spatial_keys
    print(f"Common mukeys: {len(common_keys)}")
    print(f"Flooding mukeys: {len(flooding_keys)}")
    print(f"Spatial mukeys: {len(spatial_keys)}")
    
    # Join the data
    merged_gdf = spatial_gdf.merge(flooding_data, on='mukey', how='left')
    print(f"Merged data shape: {merged_gdf.shape}")
    
    # Check join success
    if 'rating_FloodFCls_DCD_jan_dec' in merged_gdf.columns or any('flood' in col.lower() for col in merged_gdf.columns):
        print("Flooding data successfully joined!")
    else:
        print("Warning: Flooding columns not found in merged data")
    
    print("\nMerged data columns:")
    print(merged_gdf.columns.tolist())
    
    # -------- 6. Export Results --------
    print("\n=== Exporting Results ===")
    
    # Export to CSV (without geometry)
    csv_data = merged_gdf.drop(columns=['geom'] if 'geom' in merged_gdf.columns else [])
    csv_data.to_csv(OUTPUT_CSV, index=False)
    print(f"CSV saved: {OUTPUT_CSV}")
    
    # Export to Shapefile (with geometry)
    merged_gdf.to_file(OUTPUT_SHP)
    print(f"Shapefile saved: {OUTPUT_SHP}")
    
    # Show sample of final data
    print("\n=== Sample Final Data ===")
    display_cols = ['mukey', 'areasymbol', 'musym']
    # Add any flooding-related columns
    flood_cols = [col for col in merged_gdf.columns if 'flood' in col.lower() or 'rating' in col.lower()]
    display_cols.extend(flood_cols)
    
    print("Sample of merged data:")
    print(merged_gdf[display_cols].head())
    
    # -------- 7. Data Summary --------
    print("\n=== Data Summary ===")
    print(f"Total polygons: {len(merged_gdf)}")
    
    # Analyze flooding data if available
    for col in flood_cols:
        if col in merged_gdf.columns:
            print(f"\n{col} value counts:")
            print(merged_gdf[col].value_counts())
    
else:
    print("Could not complete data processing due to missing tables")

# -------- 8. Close Database Connection --------
conn.close()
print(f"Processing complete!")