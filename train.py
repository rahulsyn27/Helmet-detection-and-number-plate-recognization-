# #run 1
# from ultralytics import YOLO

# def main():
#     # Load the base YOLOv8 Nano model (fastest for real-time video later)
#     model = YOLO("yolov8n.pt") 

#     print("Starting training on Apple Silicon (MPS)...")
    
#     # Train the model
#     results = model.train(
#         data="dataset.yaml",      # Ensure this matches your YAML filename
#         epochs=100,               # Set high, relying on patience to stop it at the perfect time
#         patience=10,              # Early stopping: halts if validation accuracy doesn't improve for 10 epochs
#         imgsz=640,                # Standard input size
#         batch=16,                 # Safe batch size for Mac Unified Memory (drop to 8 if it crashes)
#         device="mps",             # Force Apple Silicon GPU usage
#         workers=4,                # Number of CPU threads to load images (keep at 4 or 2 for Mac Air)
#         project="helmet_project", # Master folder for your training runs
#         name="run_1"              # Subfolder for this specific attempt
#     )

#     print("Training complete! Your new model weights are saved at: helmet_project/run_1/weights/best.pt")

# if __name__ == '__main__':
#     main()


#run 2
# from ultralytics import YOLO

# def main():
#     # UPGRADE 1: Swapping Nano (n) for Small (s). 
#     # It has more neural layers to understand complex, tiny features.
#     model = YOLO("yolov8s.pt") 

#     print("Starting Run 2: Heavyweight Training on Apple Silicon (MPS)...")
    
#     results = model.train(
#         data="dataset.yaml",      
#         epochs=50,                # 50 full epochs
#         patience=0,               # UPGRADE 2: Disabled early stopping. Force it to learn through the flatline!
#         imgsz=800,                # UPGRADE 3: Increased resolution from 640 to 800. Plates won't be blurry now.
#         batch=8,                  # SAFETY: Dropped from 16 to 8. The larger model/image size needs more Mac memory.
#         device="mps",             
#         workers=2,                # SAFETY: Dropped to 2 threads to keep your Mac Air from freezing.
#         project="helmet_project", 
#         name="run_2"              # Saving this in a new folder so we don't overwrite run_1
#     )

#     print("Run 2 complete! Check helmet_project/run_2/weights/best.pt")

# if __name__ == '__main__':
#     main()


from ultralytics import YOLO

# 1. Load the last checkpoint using your exact path
# model = YOLO('/Users/rahulr/projects/untitled folder/Automatic-Helmet-and-Number-Plate-Detection/runs/detect/helmet_project/run_2/weights/last.pt')

# 2. Resume training
# model.train(resume=True)
# Results saved to /Users/rahulr/projects/untitled folder/Automatic-Helmet-and-Number-Plate-Detection/runs/detect/helmet_project/run_2
print(model.val())