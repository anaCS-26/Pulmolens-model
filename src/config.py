import os

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
IMAGE_SIZE = 512
NUM_EPOCHS = 20
NUM_WORKERS = 4

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
IMAGES_DIR = DATA_DIR  # Images are in subfolders like images_001, images_002 etc inside data
CSV_FILE = os.path.join(DATA_DIR, 'Data_Entry_2017.csv')
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# Class Names
CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# os.makedirs(LOGS_DIR, exist_ok=True) # Logs not currently used
