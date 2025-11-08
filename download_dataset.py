"""
Script to download the PlantVillage dataset
This script provides multiple methods to download the dataset without Git
"""

import os
import urllib.request
import zipfile
from pathlib import Path

def download_from_kaggle():
    """
    Download dataset from Kaggle using kaggle API
    """
    print("Downloading PlantVillage dataset from Kaggle...")
    print("\nPrerequisites:")
    print("1. Install kaggle: pip install kaggle")
    print("2. Set up Kaggle API credentials:")
    print("   - Go to https://www.kaggle.com/settings")
    print("   - Click 'Create New API Token'")
    print("   - Place kaggle.json in: C:\\Users\\<username>\\.kaggle\\")
    print("\nThen run:")
    print("kaggle datasets download -d emmarex/plantdisease")
    print("This will download plantdisease.zip to the current directory")
    
def download_from_github_zip():
    """
    Download dataset as ZIP from GitHub
    """
    print("\nDownloading PlantVillage dataset from GitHub...")
    
    # GitHub repository ZIP URL
    url = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"
    output_file = "PlantVillage-Dataset.zip"
    
    try:
        print(f"Downloading from: {url}")
        print("This may take several minutes depending on your internet speed...")
        
        # Download with progress
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='')
        
        urllib.request.urlretrieve(url, output_file, reporthook)
        print("\n✓ Download complete!")
        
        # Extract the ZIP file
        print("\nExtracting files...")
        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        print("✓ Extraction complete!")
        print(f"\nDataset downloaded to: PlantVillage-Dataset-master/")
        
        # Clean up
        os.remove(output_file)
        print("✓ Cleaned up temporary files")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        return False

def organize_dataset():
    """
    Organize the downloaded dataset into the required structure
    """
    print("\n" + "="*60)
    print("Organizing Dataset")
    print("="*60)
    
    # Check for downloaded dataset
    source_dirs = [
        "PlantVillage-Dataset-master/raw/color",
        "PlantVillage-Dataset-master/raw/grayscale",
        "PlantVillage-Dataset-master/raw/segmented"
    ]
    
    # Use color images by default
    source_dir = None
    for dir_path in source_dirs:
        if os.path.exists(dir_path):
            source_dir = dir_path
            break
    
    if not source_dir:
        print("✗ Could not find downloaded dataset")
        print("Please download manually from:")
        print("https://github.com/spMohanty/PlantVillage-Dataset")
        return False
    
    # Create target directory
    target_dir = "data/plantvillage"
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\nCopying images from: {source_dir}")
    print(f"To: {target_dir}")
    
    # Copy or move files
    import shutil
    
    try:
        # Get all subdirectories (disease classes)
        for class_dir in os.listdir(source_dir):
            source_class_path = os.path.join(source_dir, class_dir)
            target_class_path = os.path.join(target_dir, class_dir)
            
            if os.path.isdir(source_class_path):
                print(f"Processing: {class_dir}")
                
                # Copy directory
                if os.path.exists(target_class_path):
                    shutil.rmtree(target_class_path)
                shutil.copytree(source_class_path, target_class_path)
        
        print("\n✓ Dataset organized successfully!")
        print(f"Dataset location: {os.path.abspath(target_dir)}")
        
        # Count images
        total_images = 0
        for class_dir in os.listdir(target_dir):
            class_path = os.path.join(target_dir, class_dir)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                total_images += num_images
                print(f"  {class_dir}: {num_images} images")
        
        print(f"\nTotal images: {total_images}")
        return True
        
    except Exception as e:
        print(f"✗ Error organizing dataset: {e}")
        return False

def manual_download_instructions():
    """
    Print manual download instructions
    """
    print("\n" + "="*60)
    print("Manual Download Instructions")
    print("="*60)
    print("\nOption 1: Download from Kaggle (Recommended)")
    print("1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("2. Click 'Download' button (requires Kaggle account)")
    print("3. Extract the ZIP file")
    print("4. Move the extracted folders to: data/plantvillage/")
    
    print("\nOption 2: Download from GitHub")
    print("1. Go to: https://github.com/spMohanty/PlantVillage-Dataset")
    print("2. Click the green 'Code' button")
    print("3. Click 'Download ZIP'")
    print("4. Extract the ZIP file")
    print("5. Navigate to: PlantVillage-Dataset-master/raw/color/")
    print("6. Copy all folders to: data/plantvillage/")
    
    print("\nOption 3: Install Git and clone")
    print("1. Download Git from: https://git-scm.com/download/win")
    print("2. Install Git")
    print("3. Run: git clone https://github.com/spMohanty/PlantVillage-Dataset")
    print("4. Organize as described above")

if __name__ == "__main__":
    print("="*60)
    print("PlantVillage Dataset Download Tool")
    print("="*60)
    
    print("\nChoose download method:")
    print("1. Download from GitHub (automatic)")
    print("2. Download from Kaggle (requires setup)")
    print("3. Show manual download instructions")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        success = download_from_github_zip()
        if success:
            organize_dataset()
    elif choice == "2":
        download_from_kaggle()
    elif choice == "3":
        manual_download_instructions()
    else:
        print("Invalid choice")
    
    print("\n" + "="*60)
    print("Once dataset is ready, run: python train_model.py")
    print("="*60)
