# AgriWater-Monitoring-Water-Levels-with-Satellite-Data

## Overview
This project is focused on monitoring water levels in rice paddies using Synthetic Aperture Radar (SAR) data from the ALOS-2 satellite. The project aims to:

1. Detect water bodies using SAR intensity data.
2. Estimate water coverage in rice paddies.
3. Provide a simplified estimate of methane emissions based on water levels.
4. Enable temporal analysis of water coverage and methane emissions trends.

## Features
- **SAR Data Processing:**
  - Downsamples and smoothens SAR data using Gaussian filters to reduce noise.
  - Applies Otsu's method for automatic thresholding to detect water bodies.
  - Visualizes SAR intensity and water body detection results.
- **Methane Emission Estimation:**
  - Simplified methane emission estimation based on detected water levels.
- **Machine Learning Integration:**
  - (Optional) Uses Random Forest models to improve water detection if scikit-learn is available.
- **Temporal Analysis:**
  - Tracks changes in water coverage and methane emissions over time.
- **Parallel Processing:**
  - Processes multiple SAR images simultaneously using multithreading.

## Requirements

### Python Libraries
- `rasterio`
- `numpy`
- `matplotlib`
- `scipy`
- `skimage`
- `pandas`
- `concurrent.futures`
- `scikit-learn` (optional, for machine learning features)

### External Data
- ALOS-2 SAR GeoTIFF files.

### Files and Directories
- `raw`: Directory for unprocessed SAR data.
- `processed`: Directory for processed SAR data.
- `results`: Directory for analysis results.
- `validation`: Directory for validation data.

## Code Overview

### 1. SAR Image Processing
The `RicePaddyMonitor` class handles processing of individual SAR images. It performs the following steps:
- Reads and downsamples the SAR image.
- Applies Gaussian filtering to reduce speckle noise.
- Uses Otsu's method for automatic thresholding to detect water bodies.
- Estimates water coverage and methane emissions.
- Generates visualizations and saves results.

### 2. Workflow Management
The `RicePaddyWorkflow` class manages the overall workflow, including:
- Organizing directories.
- Extracting SAR data from zip files.
- Training machine learning models (optional).
- Processing all available SAR images using parallel execution.
- Validating results against ground truth data.
- Performing temporal analysis of water coverage and methane emissions.

## Example Usage

### Single Image Processing
```python
from rice_paddy_monitoring import RicePaddyMonitor

# Path to SAR GeoTIFF file
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
```

### Workflow Execution
```python
from rice_paddy_monitoring import RicePaddyWorkflow

# Initialize workflow
workflow = RicePaddyWorkflow(
    data_dir="E:/Education/Python/GeoProject/",
    output_dir="E:/Education/Python/GeoProject/output"
)

# Set up directories
workflow.setup_directories()

# Process all images
workflow.process_all_images()

# Perform temporal analysis
stats = workflow.generate_temporal_analysis()
print(stats)
```

## Output

### Visualizations
1. SAR Intensity (Log-transformed):
   - Visualizes backscatter intensity from the SAR data.
2. Water Detection:
   - Highlights water bodies in blue and land in white.
3. Methane Emissions Estimate:
   - Displays a simplified methane emissions estimate.

### Data Files
1. `monitoring_history.csv`:
   - Tracks historical water coverage and methane emissions data.
2. `water_mask.tif`:
   - GeoTIFF file of detected water bodies.
3. `monitoring_report.png`:
   - Visual report summarizing results.

## Future Work
- Improve methane emission modeling using field-calibrated coefficients.
- Incorporate additional environmental factors (e.g., temperature, soil type) into methane emission estimates.
- Develop a web-based dashboard for real-time monitoring and visualization.
- Enhance machine learning models for water detection using larger training datasets.


