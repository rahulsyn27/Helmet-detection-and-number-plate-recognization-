import os
import cv2
import glob

# 1. Define your folder paths (Update these to match your dataset structure)
IMAGES_DIR = "images/train"  # Path to your images
LABELS_DIR = "label/train"  # Path to your .txt files
OUTPUT_DIR = "visualized_output"        # Where the drawn images will be saved

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Class mapping and colors
class_names = {
    0: "Number Plate",
    1: "Without Helmet",
    2: "With Helmet",
    3: "Rider", 
    
}

colors = {
    0: (255, 0, 255),   # Magenta
    1: (0, 0, 255),     # Red
    2: (0, 255, 0),     # Green
    3: (255, 0, 0),     # Blue
}

# 3. Find all images in the images directory
# Adjust the extension if you are using .png or something else
image_paths = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))

if not image_paths:
    print(f"No images found in {IMAGES_DIR}. Check the path and file extensions.")

for img_path in image_paths:
    # Get the base filename (e.g., "image_0fda78") without the extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Construct the expected path for the corresponding .txt file
    label_path = os.path.join(LABELS_DIR, f"{base_name}.txt")
    
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        continue
        
    img_h, img_w, _ = img.shape
    
    # 4. Check if the label file exists
    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Convert YOLO to OpenCV coordinates
                x1 = int((x_center - (width / 2)) * img_w)
                y1 = int((y_center - (height / 2)) * img_h)
                x2 = int((x_center + (width / 2)) * img_w)
                y2 = int((y_center + (height / 2)) * img_h)
                
                label = class_names.get(class_id, f"Class {class_id}")
                color = colors.get(class_id, (255, 255, 255))
                
                # Draw box and text
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        print(f"Warning: No label file found for {base_name}.jpg")

    # 5. Save the visualized image
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_visualized.jpg")
    cv2.imwrite(output_path, img)

print(f"Done! Processed {len(image_paths)} images. Check the '{OUTPUT_DIR}' folder.")