# Helmet Detection & Number Plate Recognition

A real-time pipeline that finds motorcycle riders without helmets in traffic video and reads their number plates.

**How it works:** a custom YOLOv8 model detects 4 classes. SORT tracks each rider across frames. EasyOCR reads each plate, which is then checked against the Indian number-plate format. Results go to a CSV and an annotated output video.

## Pipeline

1. **Detect** (`main_pipeline.py`): YOLOv8 (`best_50epochs.pt`) runs on every frame for `number plate`, `without helmet`, `with helmet` and `rider`.
2. **Track**: SORT (`sort/`) gives each rider a stable ID across frames.
3. **Associate** (`util.assign_to_rider`): links helmet and plate detections to the rider box that contains them.
4. **OCR** (`util.read_license_plate`): preprocesses the plate crop, runs EasyOCR, and checks the text with `is_valid_indian_plate`.
5. **Smooth** (`normalization.py`): fills gaps and picks the best plate reading per rider (`add_missing_data.py` interpolates missing frames).
6. **Visualize** (`visualization.py`): draws boxes, helmet status and plate text into the output video.

`run_project.py` runs all steps in order and asks for the input video name.

## Files

| File | Purpose |
|---|---|
| `run_project.py` | One-command runner for the full pipeline |
| `main_pipeline.py` | Detection, tracking and OCR, writes results CSV |
| `add_missing_data.py`, `normalization.py` | Interpolate and clean per-frame results |
| `visualization.py`, `visualization_testing.py` | Render annotated video / test images |
| `util.py` | OCR, plate validation, rider association, CSV writer |
| `train.py` | YOLOv8 training runs (run 1: `yolov8n`, run 2: `yolov8s`) |
| `dataset.yaml` | Dataset config. Update `path:` to your local dataset folder. |

## Dataset

YOLO-format dataset with 4 classes (`number plate`, `without helmet`, `with helmet`, `rider`), with `images/train`, `images/val` and matching `labels/`. The images are Roboflow exports of Indian traffic scenes (about 940 images). `visualized_output/` holds the labels drawn onto the training images for checking.

## Model weights & videos

Not tracked in git:

- `best_50epochs.pt`: the custom YOLOv8 model trained for 50 epochs (about 22 MB)
- `yolov8n.pt` / `yolov8s.pt`: Ultralytics base weights, which download automatically
- `test_videos/*.mp4`: traffic clips used for testing

Put them in the project root before running.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install ultralytics opencv-python easyocr filterpy scikit-image pandas scipy
python run_project.py
```
