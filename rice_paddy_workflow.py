import os
import glob
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from rice_paddy_monitoring import RicePaddyMonitor


try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not found. Machine learning features will be disabled.")
    print("To enable ML features, install scikit-learn using:")
    print("pip install scikit-learn")
    SKLEARN_AVAILABLE = False

class RicePaddyWorkflow:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.ml_model = None
        self.validation_data = None
        self.use_ml = SKLEARN_AVAILABLE
        
    def setup_directories(self):
        """Create necessary directories for the workflow"""
        dirs = ['raw', 'processed', 'results', 'validation']
        for dir_name in [os.path.join(self.output_dir, d) for d in dirs]:
            os.makedirs(dir_name, exist_ok=True)
    
    def process_zip_files(self):
        """Extract and organize ALOS-2 zip files"""
        zip_files = glob.glob(os.path.join(self.data_dir, "*.zip"))
        
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                date_str = self._extract_date_from_filename(zip_file)
                extract_dir = os.path.join(self.output_dir, 'raw', date_str)
                os.makedirs(extract_dir, exist_ok=True)
                zip_ref.extractall(extract_dir)
    
    def train_ml_model(self, training_data_path):
        """Train Random Forest model for water detection"""
        if not self.use_ml:
            print("ML features are disabled. Install scikit-learn to enable them.")
            return
            
        if os.path.exists(training_data_path):
            training_data = pd.read_csv(training_data_path)
            X = training_data[['HH', 'HV']]  
            y = training_data['is_water']
            
            self.ml_model = RandomForestClassifier(n_estimators=100)
            self.ml_model.fit(X, y)
    
    def process_all_images(self):
        """Process all available SAR images with parallel execution"""
        tif_files = glob.glob(os.path.join(self.output_dir, 'raw', '**', '*.tif'))
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._process_single_image, tif) for tif in tif_files]
            results = [f.result() for f in futures]
        
        return self._compile_results(results)
    
    def _process_single_image(self, tif_path):
        """Process individual SAR image"""
        monitor = RicePaddyMonitor(tif_path)
        
        if self.ml_model:
            monitor.use_ml_model(self.ml_model)
        
        results = monitor.process_image()
        monitor.save_results()
        return results
    
    def validate_results(self, validation_data_path):
        """Validate results against ground truth data"""
        if os.path.exists(validation_data_path):
            self.validation_data = pd.read_csv(validation_data_path)
            validation_results = self._calculate_accuracy()
            self._save_validation_report(validation_results)
    
    def generate_temporal_analysis(self):
        """Analyze temporal patterns in water coverage and emissions"""
        history_df = pd.read_csv(os.path.join(self.output_dir, 'monitoring_history.csv'))
        history_df['date'] = pd.to_datetime(history_df['date'])
        
        # Temporal analysis
        monthly_stats = history_df.set_index('date').resample('M').agg({
            'water_coverage': ['mean', 'std'],
            'methane_emissions': ['mean', 'std']
        })
        
        return monthly_stats
    
    def _extract_date_from_filename(self, filename):
        """Extract date from ALOS-2 filename"""
        pass
    
    def _compile_results(self, results):
        """Compile results from parallel processing"""
        pass
    
    def _calculate_accuracy(self):
        """Calculate accuracy metrics"""
        pass
    
    def _save_validation_report(self, validation_results):
        """Save validation report"""
        pass

# Example usage
if __name__ == "__main__":
    # Use the direct path to your TIF file
    tif_file = "E:/Education/Python/GeoProject/IMG-HH-ALOS2014410740-140829-UBSL3.1GUA.tif"
    
    # Set up directories
    workflow = RicePaddyWorkflow(
        data_dir=os.path.dirname(tif_file),  # Get the directory containing the TIF
        output_dir="E:/Education/Python/GeoProject/output"
    )
    
    try:
        # Setup directories
        workflow.setup_directories()
        
        # Check if the TIF file exists
        if not os.path.exists(tif_file):
            print(f"TIF file not found at: {tif_file}")
            exit(1)
            
        # Process single file
        monitor = RicePaddyMonitor(tif_file)
        results = monitor.process_image()
        monitor.save_results()
        
        print(f"Successfully processed: {os.path.basename(tif_file)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your input file path") 