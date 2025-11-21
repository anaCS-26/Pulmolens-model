from ultralytics import YOLO

def train_yolo():
    # Load a model
    model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="datasets/lung_disease/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="pulmolens_yolo",
        name="yolo11m_run",
        device=0, # Use GPU 0
        plots=True
    )
    
    print("Training complete.")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    train_yolo()
