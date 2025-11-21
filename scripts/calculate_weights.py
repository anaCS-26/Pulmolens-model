import pandas as pd
import numpy as np

def calculate_weights():
    df = pd.read_csv('data/Data_Entry_2017.csv')
    all_labels = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    N = len(df)
    weights = []
    print(f"Total images: {N}")
    
    for label in all_labels:
        # Use str.contains because labels are pipe-separated (e.g., "Cardiomegaly|Emphysema")
        pos = df['Finding Labels'].str.contains(label).sum()
        neg = N - pos
        if pos > 0:
            weight = neg / pos
        else:
            weight = 1.0
        weights.append(weight)
        print(f"{label}: Pos={pos}, Weight={weight:.2f}")
        
    print(f"\nWEIGHTS_LIST = {weights}")

if __name__ == "__main__":
    calculate_weights()
