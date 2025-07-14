import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors

# === STEP 1: Load the GLB terrain model ===
model_path = r'C:/Users/CoreG/Downloads/hanselfull.glb'
mesh = trimesh.load(model_path)

# === STEP 2: Extract mesh vertices ===
if isinstance(mesh, trimesh.Scene):
    all_vertices = []
    for geom in mesh.geometry.values():
        all_vertices.append(geom.vertices)
    vertices = np.vstack(all_vertices)
else:
    vertices = mesh.vertices

# === STEP 3: Extract coordinate arrays ===
x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

# === STEP 4: Create 2D heightmap grid ===
grid_resolution = 1000
grid_x, grid_y = np.mgrid[
    x.min():x.max():complex(grid_resolution),
    y.min():y.max():complex(grid_resolution)
]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')

# === STEP 5: Elevation Heatmap ===
plt.figure(figsize=(10, 8))
plt.imshow(grid_z.T, extent=(x.min(), x.max(), y.min(), y.max()),
           origin='lower', cmap='terrain')
plt.colorbar(label='Elevation (Z)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Terrain Elevation Heatmap')
plt.show()
plt.close()

# === STEP 6: Slope Map ===
dz_dx, dz_dy = np.gradient(grid_z)
slope = np.sqrt(dz_dx**2 + dz_dy**2)

plt.figure(figsize=(10, 8))
plt.imshow(slope.T, extent=(x.min(), x.max(), y.min(), y.max()),
           origin='lower', cmap='viridis')
plt.colorbar(label='Slope Magnitude')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Slope Heatmap')
plt.show()
plt.close()

# === STEP 7: Prepare Features ===
elevation_flat = grid_z.flatten()
slope_flat = slope.flatten()

# Optional: curvature feature (commented)
# curvature = np.gradient(slope)[0]
# curvature_flat = curvature.flatten()

# Combine features
features = np.column_stack((elevation_flat, slope_flat))
# To include curvature: uncomment line below and comment the one above
# features = np.column_stack((elevation_flat, slope_flat, curvature_flat))

valid_idx = ~np.isnan(features).any(axis=1)
features_clean = features[valid_idx]

print(f"Prepared {features_clean.shape[0]} terrain samples for clustering.")

# === STEP 8: KMeans Clustering (8 clusters) ===
k = 8
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(features_clean)

# Create cluster map
elevation_flat_full = grid_z.flatten()
cluster_map = np.full(elevation_flat_full.shape, -1)
cluster_map[valid_idx] = kmeans.labels_
cluster_map = cluster_map.reshape(grid_z.shape)

# === STEP 9: Plot Cluster Map with Discrete Colors ===
cmap = plt.cm.get_cmap('Accent', k)

plt.figure(figsize=(10, 8))
im = plt.imshow(cluster_map.T, extent=(x.min(), x.max(), y.min(), y.max()),
                origin='lower', cmap=cmap)

# Create a clean, discrete colorbar
cbar = plt.colorbar(im, ticks=range(k))
cbar.set_label('Cluster ID')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Terrain Clustering (KMeans with {k} Clusters)')
plt.show()
plt.close()
