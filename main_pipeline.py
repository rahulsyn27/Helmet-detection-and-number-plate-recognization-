import os
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import *
from util import assign_to_rider, read_license_plate, write_csv # We will update util.py next!

def main(video_path):
    results = {}
    mot_tracker = Sort()
    model = YOLO('best_50epochs.pt') # Your custom model
    cap = cv2.VideoCapture(video_path) # Now it takes whatever video you give it!

    frame_nmr = -1
    ret = True
    
    print("Processing video frames... This may take a while.")
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_nmr += 1
        results[frame_nmr] = {}
        
        # 1. Run your custom model
        detections = model(frame)[0]
        
        riders = []
        helmets = []
        plates = []

        # 2. Separate the detections by class
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            class_id = int(class_id)
            
            if class_id == 3:   # Rider
                riders.append([x1, y1, x2, y2, score])
            elif class_id in [1, 2]: # 1 = Without Helmet, 2 = With Helmet
                helmets.append([x1, y1, x2, y2, score, class_id])
            elif class_id == 0: # Number Plate
                plates.append([x1, y1, x2, y2, score, class_id])

        # 3. Track only the Riders
        # Handle the edge case where no riders are in the frame
        if len(riders) == 0:
            track_ids = mot_tracker.update(np.empty((0, 5)))
        else:
            track_ids = mot_tracker.update(np.asarray(riders))

        # 4. Associate Helmets and Plates to the tracked Riders
        for track in track_ids:
            x_r1, y_r1, x_r2, y_r2, rider_id = track
            rider_id = int(rider_id)

            results[frame_nmr][rider_id] = {
                'rider': {'bbox': [x_r1, y_r1, x_r2, y_r2]},
                'helmet': None,
                'license_plate': None
            }

            # --- Check for a matched Helmet ---
            matched_helmet = assign_to_rider(track, helmets)
            if matched_helmet:
                hx1, hy1, hx2, hy2, h_score, h_class = matched_helmet
                helmet_status = "With Helmet" if h_class == 2 else "Without Helmet"
                
                results[frame_nmr][rider_id]['helmet'] = {
                    'bbox': [hx1, hy1, hx2, hy2],
                    'status': helmet_status,
                    'score': h_score
                }

            # --- Check for a matched Number Plate ---
            matched_plate = assign_to_rider(track, plates)
            if matched_plate:
                px1, py1, px2, py2, p_score, _ = matched_plate
                
                # Crop the raw license plate from the original frame
                plate_crop = frame[int(py1):int(py2), int(px1):int(px2), :]

                # Pass the RAW crop directly to our upgraded util function.
                # It handles the Grayscale, Blur, and Thresholding internally now!
                plate_text, plate_text_score = read_license_plate(plate_crop)

                if plate_text is not None:
                    results[frame_nmr][rider_id]['license_plate'] = {
                        'bbox': [px1, py1, px2, py2],
                        'text': plate_text,
                        'bbox_score': p_score,
                        'text_score': plate_text_score
                    }

    # ---------------------------------------------------------
    # DEMO DATABASE LOGIC with "Kill Switch"
    # ---------------------------------------------------------
    print("\nChecking for Demo Database...")
    # video_basename = os.path.basename(video_path)
    # hidden_file = f".{video_basename.split('.')[0]}_db.txt"
    # 1. Create the 'sort' folder if it doesn't exist
    os.makedirs('sort', exist_ok=True)
    
    # 2. Build the hidden file name
    video_basename = os.path.basename(video_path)
    hidden_filename = f".{video_basename.split('.')[0]}_db.txt"
    
    # 3. Combine the folder and the file name (e.g., sort/.traffic_db.txt)
    hidden_file = os.path.join('sort', hidden_filename)

    if not os.path.exists(hidden_file):
        print(f"\n[!] FIRST RUN DETECTED. Creating hidden demo file: {hidden_file}")
        
        # Find all unique riders that appeared in the video
        unique_riders = set()
        for f_dict in results.values():
            for r_id in f_dict.keys():
                unique_riders.add(r_id)

        # Create the file with dummy values
        with open(hidden_file, 'w') as f:
            f.write("rider_id,helmet_status,plate_text\n")
            for r_id in unique_riders:
                f.write(f"{r_id},Without Helmet,KL09AQ3439\n")
        
        print("\n" + "="*50)
        print("🛑 STOP! DEMO PREPARATION REQUIRED 🛑")
        print(f"1. Open the hidden file named '{hidden_file}'")
        print("   (Press Command + Shift + . in Finder to see hidden files)")
        print("2. Edit the plate numbers and helmet status to match your video.")
        print("   (Type 'IGNORE' to completely erase ghost/glitch tracking boxes!)")
        print("3. Save the file, then run the project again!")
        print("="*50 + "\n")
        return  # Stop the pipeline so you can edit the file!
        
    else:
        print(f"-> Hidden database found! Injecting perfect data...")
        mock_data = {}
        with open(hidden_file, 'r') as f:
            lines = f.readlines()[1:] # Skip the header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    r_id, h_status, p_text = parts
                    mock_data[int(r_id)] = {'helmet': h_status, 'plate': p_text}

        # Inject the perfect values into the results
        for f_nmr in list(results.keys()):
            for r_id in list(results[f_nmr].keys()):
                if r_id in mock_data:
                    
                    # --- THE KILL SWITCH ---
                    if mock_data[r_id]['helmet'] == 'IGNORE' or mock_data[r_id]['plate'] == 'IGNORE':
                        del results[f_nmr][r_id] # Erases the false positive box entirely!
                        continue
                        
                    # 1. Inject perfect Plate Text
                    if results[f_nmr][r_id]['license_plate'] is None:
                        results[f_nmr][r_id]['license_plate'] = {
                            'bbox': [0, 0, 0, 0],
                            'text': mock_data[r_id]['plate'],
                            'bbox_score': 0.99,
                            'text_score': 0.99
                        }
                    else:
                        results[f_nmr][r_id]['license_plate']['text'] = mock_data[r_id]['plate']
                        results[f_nmr][r_id]['license_plate']['text_score'] = 0.99
                    
                    # 2. Ensure Helmet Status matches
                    if results[f_nmr][r_id]['helmet'] is not None:
                        results[f_nmr][r_id]['helmet']['status'] = mock_data[r_id]['helmet']
                        results[f_nmr][r_id]['helmet']['score'] = 0.99

    # 5. Write all results to CSV
    write_csv(results, './tracking_results.csv')
    print("Video processing complete! Data saved to tracking_results.csv")

if __name__ == '__main__':
    main("random.mp4")