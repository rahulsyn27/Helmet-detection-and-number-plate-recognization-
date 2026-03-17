import easyocr

# Initialize the OCR reader (Turn gpu=True if you eventually run this on a CUDA machine)
reader = easyocr.Reader(['en'], gpu=False)

def write_csv(results, output_path):
    """
    Writes our nested dictionary of riders, helmets, and plates into a clean CSV.
    """
    with open(output_path, 'w') as f:
        # Our new headers for the expanded pipeline
        f.write('frame_nmr,rider_id,rider_bbox,helmet_status,helmet_bbox,helmet_score,plate_bbox,plate_bbox_score,plate_text,plate_text_score\n')

        for frame_nmr in results.keys():
            for rider_id in results[frame_nmr].keys():
                data = results[frame_nmr][rider_id]
                
                # Format Rider Box
                r_bbox = '[{} {} {} {}]'.format(*data['rider']['bbox']) if data.get('rider') else '[0 0 0 0]'
                
                # Format Helmet Data
                h_status = data['helmet']['status'] if data.get('helmet') else 'Missing'
                h_bbox = '[{} {} {} {}]'.format(*data['helmet']['bbox']) if data.get('helmet') else '[0 0 0 0]'
                h_score = data['helmet']['score'] if data.get('helmet') else '0'
                
                # Format Plate Data
                p_bbox = '[{} {} {} {}]'.format(*data['license_plate']['bbox']) if data.get('license_plate') else '[0 0 0 0]'
                p_score = data['license_plate']['bbox_score'] if data.get('license_plate') else '0'
                p_text = data['license_plate']['text'] if data.get('license_plate') else '0'
                p_text_score = data['license_plate']['text_score'] if data.get('license_plate') else '0'
                
                f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
                    frame_nmr, rider_id, r_bbox, h_status, h_bbox, h_score, p_bbox, p_score, p_text, p_text_score
                ))

def read_license_plate(license_plate_crop):
    """
    Reads text from the cropped plate. We removed the strict 7-character limit 
    so it correctly captures varying plate formats.
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        # Clean up the OCR text (remove spaces and weird characters)
        text = text.upper().replace(' ', '').replace('_', '').replace('-', '')
        
        # As long as the OCR found at least 4 alphanumeric characters, accept it
        if len(text) >= 4:
            return text, score

    return None, None

def assign_to_rider(tracked_rider, items_list):
    """
    Calculates if a sub-object (helmet or plate) is physically located inside 
    the bounding box of a tracked rider using Area of Intersection.
    """
    rx1, ry1, rx2, ry2, _ = tracked_rider
    
    best_item = None
    max_overlap_area = 0
    
    for item in items_list:
        # item can be a helmet or a plate
        ix1, iy1, ix2, iy2, score, class_id = item
        
        # Calculate intersection coordinates
        x_left = max(rx1, ix1)
        y_top = max(ry1, iy1)
        x_right = min(rx2, ix2)
        y_bottom = min(ry2, iy2)
        
        # Check if there is an actual intersection
        if x_right > x_left and y_bottom > y_top:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            item_area = (ix2 - ix1) * (iy2 - iy1)
            
            # If more than 50% of the item is inside the rider's box, it belongs to them
            if intersection_area / item_area > 0.5:
                if intersection_area > max_overlap_area:
                    max_overlap_area = intersection_area
                    best_item = item
                    
    return best_item