import matplotlib.pyplot as plt
import rasterio
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

# Path to the GeoTIFF file
tif_path = r"E:\Education\Python\GeoProject\IMG-HH-ALOS2014410740-140829-UBSL3.1GUA.tif"

# Open the GeoTIFF file
with rasterio.open(tif_path) as dataset:
    # Read the first band with reduced resolution (downsample by factor of 4)
    band1 = dataset.read(
        1,
        out_shape=(
            dataset.height // 4,
            dataset.width // 4
        )
    )
    
    # Improve water detection
    # Apply Gaussian filter to reduce speckle noise
    band1_smooth = gaussian_filter(band1, sigma=2)
    band1_log = np.log10(band1_smooth + 1e-10)
    
    # Use Otsu's method for automatic thresholding
    water_threshold = threshold_otsu(band1_log)
    water_mask = band1_log < water_threshold

    # Create a custom colormap for water bodies
    colors = ['darkblue', 'lightblue', 'white']
    n_bins = 3
    water_cmap = ListedColormap(colors)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Original SAR data with better contrast
im1 = ax1.imshow(band1_log, 
                 cmap='gray',
                 extent=(dataset.bounds.left, dataset.bounds.right, 
                        dataset.bounds.bottom, dataset.bounds.top))
ax1.set_title("SAR Intensity (Log-transformed)")
plt.colorbar(im1, ax=ax1, label="Log(Backscatter)")

# Plot 2: Water detection with custom colormap
im2 = ax2.imshow(water_mask,
                 cmap=water_cmap,
                 extent=(dataset.bounds.left, dataset.bounds.right, 
                        dataset.bounds.bottom, dataset.bounds.top))
ax2.set_title("Water Bodies Detection")
plt.colorbar(im2, ax=ax2, label="Water (Blue) / Land (White)")

# Add grid lines and improve layout
for ax in [ax1, ax2]:
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

plt.tight_layout()
plt.show()

# Optional: Save the water mask
with rasterio.open('water_mask.tif', 'w',
                  driver='GTiff',
                  height=water_mask.shape[0],
                  width=water_mask.shape[1],
                  count=1,
                  dtype=water_mask.dtype,
                  crs=dataset.crs,
                  transform=dataset.transform) as dst:
    dst.write(water_mask.astype(np.uint8), 1)
