import matplotlib.pyplot as plt
import rasterio
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from datetime import datetime
import pandas as pd

class RicePaddyMonitor:
    def __init__(self, tif_path):
        self.tif_path = tif_path
        self.water_mask = None
        self.water_level = None
        self.date = datetime.strptime(tif_path.split('-')[-2], '%y%m%d')
    
    def process_image(self):
        with rasterio.open(self.tif_path) as dataset:
            # Read and downsample the image
            band1 = dataset.read(
                1,
                out_shape=(dataset.height // 4, dataset.width // 4)
            )
            
            # Improve water detection
            band1_smooth = gaussian_filter(band1, sigma=2)
            band1_log = np.log10(band1_smooth + 1e-10)
            
            # Water detection with Otsu's method
            water_threshold = threshold_otsu(band1_log)
            self.water_mask = band1_log < water_threshold
            
            # Estimate water level (simplified version)
            self.water_level = np.mean(self.water_mask)
            
            return self.create_visualization(band1_log, dataset)
    
    def estimate_methane_emissions(self):
        """
        Simplified methane emission estimation based on water coverage
        Returns estimated methane emissions in kg/ha/day
        """
        # Simplified model: more water = more methane
        # These coefficients should be calibrated with actual field data
        base_emission = 0.1  # kg CH4/ha/day
        water_factor = 2.0  # multiplication factor for water coverage
        
        return base_emission * (1 + water_factor * self.water_level)
    
    def create_visualization(self, band1_log, dataset):
        # Create custom colormaps
        water_cmap = ListedColormap(['darkblue', 'lightblue', 'white'])
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # Plot 1: Original SAR data
        im1 = ax1.imshow(band1_log, cmap='gray',
                        extent=(dataset.bounds.left, dataset.bounds.right,
                               dataset.bounds.bottom, dataset.bounds.top))
        ax1.set_title("SAR Intensity")
        plt.colorbar(im1, ax=ax1, label="Log(Backscatter)")
        
        # Plot 2: Water detection
        im2 = ax2.imshow(self.water_mask, cmap=water_cmap,
                        extent=(dataset.bounds.left, dataset.bounds.right,
                               dataset.bounds.bottom, dataset.bounds.top))
        ax2.set_title(f"Water Bodies Detection\nWater Coverage: {self.water_level:.2%}")
        plt.colorbar(im2, ax=ax2, label="Water (Blue) / Land (White)")
        
        # Plot 3: Estimated Methane Emissions
        methane = self.estimate_methane_emissions()
        ax3.text(0.5, 0.5, f"Estimated Methane Emissions:\n{methane:.2f} kg/ha/day",
                ha='center', va='center', fontsize=12)
        ax3.set_title("Emissions Estimate")
        ax3.axis('off')
        
        # Add grid lines and improve layout
        for ax in [ax1, ax2]:
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")
        
        plt.suptitle(f"Rice Paddy Monitoring - {self.date.strftime('%Y-%m-%d')}")
        plt.tight_layout()
        
        return fig

    def save_results(self):
        """Save results to CSV for historical tracking"""
        results = {
            'date': [self.date],
            'water_coverage': [self.water_level],
            'methane_emissions': [self.estimate_methane_emissions()]
        }
        
        # Create new DataFrame from results
        new_data = pd.DataFrame(results)
        
        # Append to existing CSV or create new one
        try:
            existing_df = pd.read_csv('monitoring_history.csv')
            # Use concat instead of append
            df = pd.concat([existing_df, new_data], ignore_index=True)
        except FileNotFoundError:
            df = new_data
        
        df.to_csv('monitoring_history.csv', index=False)

# Usage example
if __name__ == "__main__":
    tif_path = r"E:\Education\Python\GeoProject\IMG-HH-ALOS2014410740-140829-UBSL3.1GUA.tif"
    
    # Initialize and process
    monitor = RicePaddyMonitor(tif_path)
    fig = monitor.process_image()
    
    # Save results
    monitor.save_results()
    
    # Save figure instead of showing it
    output_path = 'monitoring_report.png'
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory
    
    print(f"Results saved to: {output_path}") 