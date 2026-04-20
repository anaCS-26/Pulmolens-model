import os
from glob import glob
from PIL import Image
from tqdm import tqdm
import subprocess

def verify_and_repair(data_dir, zip_path):
    print("Scanning extracted images for corruption...")
    images = glob(os.path.join(data_dir, '**', '*.png'), recursive=True)
    
    corrupted_files = []
    
    # 1. Quickly verify each file without fully loading into memory
    for img_path in tqdm(images, desc="Verifying Images"):
        try:
            with Image.open(img_path) as img:
                img.verify() # Only reads headers - extremely fast
        except Exception:
            corrupted_files.append(img_path)
            
    if not corrupted_files:
        print("All image files are perfectly intact! No repair needed.")
        return

    print(f"\nFound {len(corrupted_files)} corrupted files.")
    
    # 2. Delete the corrupted files
    for bad_file in corrupted_files:
        print(f"Deleting broken file: {bad_file}")
        os.remove(bad_file)
        
    print("\nExtracting ONLY the missing/repaired files from the archive...")
    # 3. Use unzip -n (never overwrite) to skip intact files and ONLY replace the missing (deleted) ones
    subprocess.run(["unzip", "-q", "-n", zip_path, "-d", data_dir])
    
    print("\nTargeted repair complete!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(current_dir, 'data')
    zip_file = os.path.join(current_dir, 'data', 'data.zip')
    
    if not os.path.exists(data_folder) or not os.path.exists(zip_file):
        print("Data folder or data.zip not found! Cannot run repair.")
    else:
        verify_and_repair(data_folder, zip_file)
