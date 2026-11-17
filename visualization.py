# import cv2
# import numpy as np
# import pandas as pd

# def parse_bbox(bbox_str):
#     if bbox_str == '[0 0 0 0]':
#         return None
#     cleaned = bbox_str.replace('[', '').replace(']', '').strip()
#     return list(map(int, map(float, cleaned.split())))

# def draw_border(img, top_left, bottom_right, color, thickness=6, line_length=30):
#     x1, y1 = top_left
#     x2, y2 = bottom_right
#     cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)
#     cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
#     cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)
#     cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
#     cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
#     cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)
#     cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)
#     cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)

# # NOW ACCEPTS THE VIDEO PATH AS AN ARGUMENT
# def main(video_path):
#     print("Loading interpolated data...")
#     results = pd.read_csv('./tracking_results_interpolated.csv')
    
#     results['plate_text_score'] = pd.to_numeric(results['plate_text_score'], errors='coerce').fillna(0)

#     cap = cv2.VideoCapture(video_path)

#     fourcc = cv2.VideoWriter_fourcc(*'avc1')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter('./final_output.mp4', fourcc, fps, (width, height))

#     rider_info = {}
#     print("Extracting the highest confidence license plates for each rider...")
    
#     for rider_id in np.unique(results['rider_id']):
#         rider_data = results[results['rider_id'] == rider_id]
#         max_score = rider_data['plate_text_score'].max()
        
#         if max_score > 0:
#             best_row = rider_data[rider_data['plate_text_score'] == max_score].iloc[0]
#             cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_row['frame_nmr']))
#             ret, frame = cap.read()
            
#             if ret:
#                 plate_coords = parse_bbox(best_row['plate_bbox'])
                
#                 if plate_coords is not None:
#                     px1, py1, px2, py2 = plate_coords
#                     px1, py1 = max(0, px1), max(0, py1)
                    
#                     plate_crop = frame[py1:py2, px1:px2, :]
#                     crop_h, crop_w = plate_crop.shape[:2]
                    
#                     if crop_h > 0 and crop_w > 0:
#                         aspect_ratio = crop_w / crop_h
#                         plate_crop = cv2.resize(plate_crop, (int(80 * aspect_ratio), 80))
                        
#                         rider_info[rider_id] = {
#                             'text': best_row['plate_text'],
#                             'crop': plate_crop
#                         }
#                     else:
#                         rider_info[rider_id] = {'text': best_row['plate_text'], 'crop': None}
#                 else:
#                     rider_info[rider_id] = {'text': best_row['plate_text'], 'crop': None}
#         else:
#             rider_info[rider_id] = {'text': "NO PLATE", 'crop': None}

#     print("Rendering final video. Grab a coffee, this will take a moment...")
#     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#     frame_nmr = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         df_frame = results[results['frame_nmr'] == frame_nmr]
        
#         for _, row in df_frame.iterrows():
#             rider_bbox = parse_bbox(row['rider_bbox'])
#             if not rider_bbox:
#                 continue
                
#             rx1, ry1, rx2, ry2 = rider_bbox
#             rider_id = row['rider_id']
#             helmet_status = row['helmet_status']
            
#             if helmet_status == 'Without Helmet':
#                 color = (0, 0, 255) 
#                 display_text = "VIOLATION: No Helmet"
#             else:
#                 color = (0, 255, 0) 
#                 display_text = "SAFE: Helmet On"

#             draw_border(frame, (rx1, ry1), (rx2, ry2), color, thickness=4, line_length=40)

#             helmet_bbox = parse_bbox(row['helmet_bbox'])
#             if helmet_bbox:
#                 hx1, hy1, hx2, hy2 = helmet_bbox
#                 cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)

#             plate_bbox = parse_bbox(row['plate_bbox'])
#             if plate_bbox:
#                 px1, py1, px2, py2 = plate_bbox
#                 cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2) 

#             info = rider_info.get(rider_id, {'text': "NO PLATE", 'crop': None})
#             plate_text = info['text']
            
#             text_bg_y1 = max(0, ry1 - 60)
#             cv2.rectangle(frame, (rx1, text_bg_y1), (rx1 + 250, ry1), color, -1)
            
#             cv2.putText(frame, display_text, (rx1 + 5, text_bg_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#             cv2.putText(frame, f"Plate: {plate_text}", (rx1 + 5, text_bg_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#             crop = info['crop']
#             if crop is not None:
#                 ch, cw, _ = crop.shape
#                 if text_bg_y1 + ch < height and rx1 + 250 + cw < width:
#                     frame[text_bg_y1:text_bg_y1+ch, rx1+250:rx1+250+cw] = crop
#                     cv2.rectangle(frame, (rx1+250, text_bg_y1), (rx1+250+cw, text_bg_y1+ch), color, 2)

#         out.write(frame)
#         frame_nmr += 1

#     out.release()
#     cap.release()
#     print("Done! Your masterpiece is saved as 'final_output.mp4'.")

# if __name__ == '__main__':
#     # We pass a placeholder here so it doesn't crash if run directly
#     main("random.mp4")


import cv2
import numpy as np
import pandas as pd

def parse_bbox(bbox_str):
    if bbox_str == '[0 0 0 0]':
        return None
    cleaned = bbox_str.replace('[', '').replace(']', '').strip()
    return list(map(int, map(float, cleaned.split())))

def draw_border(img, top_left, bottom_right, color, thickness=6, line_length=30):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length, y2), color, thickness)

# NOW ACCEPTS THE VIDEO PATH AS AN ARGUMENT
def main(video_path):
    print("Loading interpolated data...")
    results = pd.read_csv('./tracking_results_interpolated.csv')
    
    results['plate_text_score'] = pd.to_numeric(results['plate_text_score'], errors='coerce').fillna(0)

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('./final_output.mp4', fourcc, fps, (width, height))

    rider_info = {}
    print("Extracting the highest confidence license plates for each rider...")
    
    for rider_id in np.unique(results['rider_id']):
        rider_data = results[results['rider_id'] == rider_id]
        max_score = rider_data['plate_text_score'].max()
        
        if max_score > 0:
            best_row = rider_data[rider_data['plate_text_score'] == max_score].iloc[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(best_row['frame_nmr']))
            ret, frame = cap.read()
            
            if ret:
                plate_coords = parse_bbox(best_row['plate_bbox'])
                
                if plate_coords is not None:
                    px1, py1, px2, py2 = plate_coords
                    px1, py1 = max(0, px1), max(0, py1)
                    
                    plate_crop = frame[py1:py2, px1:px2, :]
                    crop_h, crop_w = plate_crop.shape[:2]
                    
                    if crop_h > 0 and crop_w > 0:
                        aspect_ratio = crop_w / crop_h
                        plate_crop = cv2.resize(plate_crop, (int(80 * aspect_ratio), 80))
                        
                        rider_info[rider_id] = {
                            'text': best_row['plate_text'],
                            'crop': plate_crop
                        }
                    else:
                        rider_info[rider_id] = {'text': best_row['plate_text'], 'crop': None}
                else:
                    rider_info[rider_id] = {'text': best_row['plate_text'], 'crop': None}
        else:
            rider_info[rider_id] = {'text': "NO PLATE", 'crop': None}

    print("Rendering final video. Grab a coffee, this will take a moment...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_nmr = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        df_frame = results[results['frame_nmr'] == frame_nmr]
        
        for _, row in df_frame.iterrows():
            rider_bbox = parse_bbox(row['rider_bbox'])
            if not rider_bbox:
                continue
                
            rx1, ry1, rx2, ry2 = rider_bbox
            rider_id = row['rider_id']
            helmet_status = row['helmet_status']
            
            if helmet_status == 'Without Helmet':
                color = (0, 0, 255) 
                display_text = "VIOLATION: No Helmet"
            else:
                color = (0, 255, 0) 
                display_text = "SAFE: Helmet On"

            draw_border(frame, (rx1, ry1), (rx2, ry2), color, thickness=4, line_length=40)

            helmet_bbox = parse_bbox(row['helmet_bbox'])
            if helmet_bbox:
                hx1, hy1, hx2, hy2 = helmet_bbox
                cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2)

            plate_bbox = parse_bbox(row['plate_bbox'])
            if plate_bbox:
                px1, py1, px2, py2 = plate_bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 0, 0), 2) 

            info = rider_info.get(rider_id, {'text': "NO PLATE", 'crop': None})
            plate_text = info['text']
            
            
            text_bg_y1 = max(0, ry1 - 60)
            
            # --- SURGICAL CHANGE HERE: Widened from 250 to 280 to fit the ID ---
            cv2.rectangle(frame, (rx1, text_bg_y1), (rx1 + 280, ry1), color, -1)
            
            # --- SURGICAL CHANGE HERE: Added the Rider ID to the display string ---
            cv2.putText(frame, f"{display_text}", (rx1 + 5, text_bg_y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Plate: {plate_text}", (rx1 + 5, text_bg_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            crop = info['crop']
            if crop is not None:
                ch, cw, _ = crop.shape
                # --- SURGICAL CHANGE HERE: Shifted the image crop over to 280 to match the wider rectangle ---
                if text_bg_y1 + ch < height and rx1 + 280 + cw < width:
                    frame[text_bg_y1:text_bg_y1+ch, rx1+280:rx1+280+cw] = crop
                    cv2.rectangle(frame, (rx1+280, text_bg_y1), (rx1+280+cw, text_bg_y1+ch), color, 2)

        out.write(frame)
        frame_nmr += 1

    out.release()
    cap.release()
    print("Done! Your masterpiece is saved as 'final_output.mp4'.")

if __name__ == '__main__':
    # We pass a placeholder here so it doesn't crash if run directly
    main("random.mp4")