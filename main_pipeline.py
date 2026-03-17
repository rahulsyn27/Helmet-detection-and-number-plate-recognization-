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
        # This generates unique IDs so we can follow them across frames
        # track_ids = mot_tracker.update(np.asarray(riders))

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
                
                # Crop and process license plate for OCR
                plate_crop = frame[int(py1):int(py2), int(px1):int(px2), :]
                plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, plate_crop_thresh = cv2.threshold(plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read text using EasyOCR
                plate_text, plate_text_score = read_license_plate(plate_crop_thresh)

                if plate_text is not None:
                    results[frame_nmr][rider_id]['license_plate'] = {
                        'bbox': [px1, py1, px2, py2],
                        'text': plate_text,
                        'bbox_score': p_score,
                        'text_score': plate_text_score
                    }

    # 5. Write all results to CSV
    write_csv(results, './tracking_results.csv')
    print("Video processing complete! Data saved to tracking_results.csv")

if __name__ == '__main__':
    main()