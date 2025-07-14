import trimesh
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd
from sklearn.cluster import KMeans
from scipy.interpolate import griddata
from scipy.ndimage import sobel

# Load 3D GLB terrain model
mesh = trimesh.load(r'C:\Users\CoreG\Downloads\hanselfull.glb')

if isinstance(mesh, trimesh.Scene):
    all_vertices = [geom.vertices for geom in mesh.geometry.values()]
    vertices = np.vstack(all_vertices)
else:
    vertices = mesh.vertices

print(f"Total number of vertices: {len(vertices)}")

x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

# Create elevation grid
resolution = 1000
grid_x, grid_y = np.meshgrid(
    np.linspace(x.min(), x.max(), resolution),
    np.linspace(y.min(), y.max(), resolution)
)
grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

# Calculate slope using Sobel filters
dz_dx = sobel(grid_z, axis=1, mode='constant')
dz_dy = sobel(grid_z, axis=0, mode='constant')
slope = np.sqrt(dz_dx**2 + dz_dy**2)

# NOAA CDO API configuration
NOAA_TOKEN = 'sfwgSyWldoXgQfPAIbYJCGimFmXiLQCE'
headers = {'token': NOAA_TOKEN}

# Location and FIPS for Bucks County, PA
fips_code = 'FIPS:42017'  # Bucks County FIPS code
latitude = 40.350143
longitude = -75.0845715

# Find stations by FIPS location (Bucks County, PA)
station_params = {
    'datasetid': 'GHCND',
    'locationid': fips_code,
    'datatypeid': 'PRCP',
    'limit': 5
}

resp = requests.get(
    'https://www.ncei.noaa.gov/cdo-web/api/v2/stations',
    headers=headers,
    params=station_params
)

print("Status code:", resp.status_code)
print("Response preview:", resp.text[:500])

if resp.status_code != 200:
    raise Exception("Failed to retrieve station data.")

stations = resp.json().get('results', [])
if not stations:
    raise Exception("No weather stations found in the specified FIPS area.")

# NOAA API only allows max 1 year date range per request
start_year = 2019
end_year = 2023

precip_data = []
used_station = None

for station in stations:
    station_id = station['id']
    print(f"Trying station: {station_id} - {station['name']}")

    station_data = []

    # Loop through each year chunk (max 1 year)
    for year in range(start_year, end_year + 1):
        startdate = f"{year}-01-01"
        enddate = f"{year}-12-31"

        data_params = {
            'datasetid': 'GHCND',
            'datatypeid': 'PRCP',
            'stationid': station_id,
            'startdate': startdate,
            'enddate': enddate,
            'limit': 1000,
            'units': 'metric'
        }

        data_resp = requests.get(
            'https://www.ncei.noaa.gov/cdo-web/api/v2/data',
            headers=headers,
            params=data_params
        )

        print(f"Year {year} request status: {data_resp.status_code}")

        if data_resp.status_code == 200:
            yearly_data = data_resp.json().get('results', [])
            if yearly_data:
                station_data.extend(yearly_data)
            else:
                print(f"No precipitation data for {year} at station {station_id}")
        else:
            print(f"Failed to get data for {year} at station {station_id}: {data_resp.text[:200]}")

    if station_data:
        precip_data = station_data
        used_station = station
        print(f"Using station: {station_id} - {station['name']} with {len(precip_data)} total data points")
        break
    else:
        print(f"No usable data for station {station_id}")

if not precip_data:
    raise Exception("Failed to retrieve precipitation data from all stations.")

# Process precipitation data into DataFrame
df = pd.DataFrame(precip_data)
df['date'] = pd.to_datetime(df['date'])
df['precip_mm'] = df['value'] * 0.1  # Convert tenths of mm to mm

# Calculate average annual precipitation
annual = df.groupby(df['date'].dt.year)['precip_mm'].sum()
avg_rainfall_mm = annual.mean()
print(f"Average annual rainfall: {avg_rainfall_mm:.2f} mm")

# Create rainfall grid matching elevation grid shape with uniform value (simplification)
rainfall_grid = np.full(grid_z.shape, avg_rainfall_mm)

# Prepare data for clustering (elevation, slope, rainfall)
elevation_flat = grid_z.flatten()
slope_flat = slope.flatten()
rain_flat = rainfall_grid.flatten()

features = np.column_stack((elevation_flat, slope_flat, rain_flat))

# Remove rows with NaN values (due to interpolation at edges)
valid_idx = ~np.isnan(features).any(axis=1)
features_clean = features[valid_idx]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42)
labels = np.full(elevation_flat.shape, np.nan)
labels[valid_idx] = kmeans.fit_predict(features_clean)
label_grid = labels.reshape(grid_z.shape)

# Plot the resulting clusters
plt.figure(figsize=(8, 6))
plt.imshow(label_grid.T, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='tab10')
plt.title('Terrain Clustering (Elevation + Slope + Rainfall)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.colorbar(label='Cluster ID')
plt.show()
