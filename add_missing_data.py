import csv
import numpy as np
from scipy.interpolate import interp1d
from collections import Counter

def interpolate_bounding_boxes(data):
    interpolated_data = []

    # Get all unique rider IDs
    rider_ids = np.unique([int(row['rider_id']) for row in data])

    for rider_id in rider_ids:
        # 1. Group all frames belonging to this specific rider
        rider_rows = [row for row in data if int(row['rider_id']) == rider_id]
        rider_rows = sorted(rider_rows, key=lambda x: int(x['frame_nmr']))

        # 2. MAJORITY VOTE: Stop the helmet flickering!
        # Find the most frequent helmet prediction for this rider across the whole video
        # valid_statuses = [row['helmet_status'] for row in rider_rows if row['helmet_status'] != 'Missing']
        # if valid_statuses:
        #     majority_helmet_status = Counter(valid_statuses).most_common(1)[0][0]
        # else:
        #     majority_helmet_status = 'Missing'

        # 2. WEIGHTED VOTE: Stop the helmet flickering using Confidence Scores!
        # Sum the confidence scores for each prediction to find the strongest overall status
        status_weights = {'With Helmet': 0.0, 'Without Helmet': 0.0}
        
        for row in rider_rows:
            status = row['helmet_status']
            if status in status_weights:
                try:
                    # Add the confidence score to the total weight for this status
                    score = float(row['helmet_score'])
                    status_weights[status] += score
                except ValueError:
                    pass # Ignore corrupted score strings
        
        # Determine the winner based on the highest total confidence score
        if status_weights['With Helmet'] == 0.0 and status_weights['Without Helmet'] == 0.0:
            majority_helmet_status = 'Missing'
        else:
            majority_helmet_status = max(status_weights, key=status_weights.get)

        # 3. Extract valid frames to build our interpolation paths
        frames = []
        rider_bboxes, helmet_frames, helmet_bboxes, plate_frames, plate_bboxes = [], [], [], [], []

        for row in rider_rows:
            f_num = int(row['frame_nmr'])
            frames.append(f_num)
            
            # Clean the string arrays back into Python lists
            rider_bboxes.append(list(map(float, row['rider_bbox'].strip('[]').split())))

            if row['helmet_bbox'] != '[0 0 0 0]':
                helmet_frames.append(f_num)
                helmet_bboxes.append(list(map(float, row['helmet_bbox'].strip('[]').split())))

            if row['plate_bbox'] != '[0 0 0 0]':
                plate_frames.append(f_num)
                plate_bboxes.append(list(map(float, row['plate_bbox'].strip('[]').split())))

        # 4. Create the interpolation mathematical functions
        interp_rider = interp1d(frames, rider_bboxes, axis=0, kind='linear') if len(frames) > 1 else None
        interp_helmet = interp1d(helmet_frames, helmet_bboxes, axis=0, kind='linear') if len(helmet_frames) > 1 else None
        interp_plate = interp1d(plate_frames, plate_bboxes, axis=0, kind='linear') if len(plate_frames) > 1 else None

        first_frame = frames[0]
        last_frame = frames[-1]

        # 5. Rebuild the timeline, filling in every single missing frame perfectly
        for f in range(first_frame, last_frame + 1):
            row = {
                'frame_nmr': str(f),
                'rider_id': str(rider_id),
                'helmet_status': majority_helmet_status, # Lock in our anti-flicker vote
                'helmet_score': '0',
                'plate_bbox_score': '0',
                'plate_text': '0',
                'plate_text_score': '0'
            }

            # Smoothly animate the boxes
            if interp_rider:
                row['rider_bbox'] = '[{:.1f} {:.1f} {:.1f} {:.1f}]'.format(*interp_rider(f))
            else:
                row['rider_bbox'] = '[{:.1f} {:.1f} {:.1f} {:.1f}]'.format(*rider_bboxes[0])

            if interp_helmet and helmet_frames[0] <= f <= helmet_frames[-1]:
                row['helmet_bbox'] = '[{:.1f} {:.1f} {:.1f} {:.1f}]'.format(*interp_helmet(f))
            else:
                row['helmet_bbox'] = '[0 0 0 0]'

            if interp_plate and plate_frames[0] <= f <= plate_frames[-1]:
                row['plate_bbox'] = '[{:.1f} {:.1f} {:.1f} {:.1f}]'.format(*interp_plate(f))
            else:
                row['plate_bbox'] = '[0 0 0 0]'

            # If this frame actually existed in the raw data, restore the real OCR scores
            original_row = next((r for r in rider_rows if int(r['frame_nmr']) == f), None)
            if original_row:
                row['helmet_score'] = original_row['helmet_score']
                row['plate_bbox_score'] = original_row['plate_bbox_score']
                row['plate_text'] = original_row['plate_text']
                row['plate_text_score'] = original_row['plate_text_score']

            interpolated_data.append(row)

    return interpolated_data

def main():
    print("Loading raw tracking data...")
    with open('tracking_results.csv', 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    print("Interpolating missing boxes and neutralizing flicker...")
    interpolated_data = interpolate_bounding_boxes(data)

    print("Saving clean data...")
    header = ['frame_nmr', 'rider_id', 'rider_bbox', 'helmet_status', 'helmet_bbox', 'helmet_score', 'plate_bbox', 'plate_bbox_score', 'plate_text', 'plate_text_score']
    with open('tracking_results_interpolated.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)
        
    print("Success! tracking_results_interpolated.csv is ready.")

if __name__ == '__main__':
    main()