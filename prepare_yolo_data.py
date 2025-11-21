import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataset import NIHChestXrayDataset # Reuse for path finding

def prepare_yolo_data(csv_path, output_dir, image_root_dir):
    print("Reading CSV...")
    df = pd.read_csv(csv_path)
    
    # Map classes to IDs
    classes = sorted(df['Finding Label'].unique())
    class_to_id = {cls: i for i, cls in enumerate(classes)}
    print(f"Classes: {class_to_id}")
    
    # Create directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    # Group by image
    grouped = df.groupby('Image Index')
    images = list(grouped.groups.keys())
    
    # Split
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Helper to find image path (using our existing logic)
    # We'll just scan the directory once to build a map
    print("Scanning image directories...")
    image_path_map = {}
    for root, dirs, files in os.walk(image_root_dir):
        for file in files:
            if file.endswith('.png'):
                image_path_map[file] = os.path.join(root, file)
                
    def process_split(image_list, split_name):
        print(f"Processing {split_name} split...")
        for img_name in tqdm(image_list):
            if img_name not in image_path_map:
                print(f"Warning: Image {img_name} not found.")
                continue
                
            src_path = image_path_map[img_name]
            dst_path = os.path.join(output_dir, 'images', split_name, img_name)
            
            # Copy image
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
            
            # Create label file
            label_path = os.path.join(output_dir, 'labels', split_name, img_name.replace('.png', '.txt'))
            
            # Get image dimensions (assuming 1024x1024 for NIH, but let's check one or assume from CSV if needed)
            # The BBox CSV has absolute coordinates. We need to normalize.
            # NIH images are typically 1024x1024. Let's verify with PIL for each image to be safe.
            from PIL import Image
            with Image.open(src_path) as img:
                img_w, img_h = img.size
                
            with open(label_path, 'w') as f:
                group = grouped.get_group(img_name)
                for _, row in group.iterrows():
                    cls_id = class_to_id[row['Finding Label']]
                    # Columns are: 'Bbox [x', 'y', 'w', 'h]'
                    x = row['Bbox [x']
                    y = row['y']
                    w = row['w']
                    h = row['h]']
                    
                    # Wait, pandas might have parsed it weirdly if the header is "Bbox [x,y,w,h]"
                    # Let's inspect the dataframe structure in the main block first or handle it robustly.
                    # Based on previous `head` output:
                    # Image Index,Finding Label,Bbox [x,y,w,h],,,
                    # 00013118_008.png,Atelectasis,225.084745762712,547.019216763771,86.7796610169491,79.1864406779661
                    # So pandas likely put x in 'Bbox [x,y,w,h]', y in 'Unnamed: 3', w in 'Unnamed: 4', h in 'Unnamed: 5'
                    # Actually, let's look at the head output again.
                    # 00013118_008.png,Atelectasis,225...,547...,86...,79...
                    # Col 0: Image Index
                    # Col 1: Finding Label
                    # Col 2: Bbox [x,y,w,h] (Header name) -> Value is x
                    # Col 3: Unnamed: 3 -> Value is y
                    # Col 4: Unnamed: 4 -> Value is w
                    # Col 5: Unnamed: 5 -> Value is h
                    
                    
                    # Convert to YOLO format: x_center, y_center, w, h (normalized)
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    process_split(train_imgs, 'train')
    process_split(val_imgs, 'val')
    
    # Create data.yaml
    yaml_content = f"""
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
names:
"""
    for i, cls in enumerate(classes):
        yaml_content += f"  {i}: {cls}\n"
        
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
        
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_yolo_data(
        csv_path='data/BBox_List_2017.csv',
        output_dir='datasets/lung_disease',
        image_root_dir='data'
    )
