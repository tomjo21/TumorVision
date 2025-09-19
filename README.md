# TumorVision

Brain tumor MRI classification with TensorFlow/Keras.

## Structure
- `src/dataset.py` — dataset loader (Training/Validation split + Testing).
- `src/model.py` — simple CNN model (128x128) used for ~0.95 test accuracy on the provided dataset.
- `src/train.py` — trains the model, saves `brain_tumor_model.keras` and `class_names.json`.
- `predict.py` — single image prediction (uses `IMG_SIZE` from `dataset.py`).
- `src/evaluate.py` — prints accuracy and classification report; saves `confusion_matrix.png`.

## Quick start
1. Place data under `data/dataset/Training` and `data/dataset/Testing` with class subfolders: `glioma`, `meningioma`, `notumor`, `pituitary`.
2. Create/activate a virtual environment and install TensorFlow (and sklearn, matplotlib, seaborn):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install tensorflow scikit-learn matplotlib seaborn
```

3. Train:
```powershell
python src\train.py
```

4. Predict (uses a default sample if no arg is provided):
```powershell
python predict.py
# or
python predict.py data\dataset\Testing\glioma\Te-gl_0010.jpg
```

5. Evaluate:
```powershell
python src\evaluate.py
```

## Patient-wise splits (prevent leakage)
Create splits that keep all images from a patient in a single set:

```powershell
# Option A: regex to extract patient id from filename (use a capturing group)
python src\build_patient_splits.py --regex "^(Te-[a-z]+)_"

# Option B: provide a CSV mapping (image_path,patient_id) relative to data/dataset
python src\build_patient_splits.py --csv_map data\patient_map.csv
```

This writes `data/splits/train.txt`, `val.txt`, `test.txt`. When these exist, `src/dataset.py` will load from them automatically.

## Clean duplicates
Detect near-duplicate images to reduce label noise:

```powershell
python src\clean_duplicates.py data\dataset --threshold 5 --quarantine data\dupe_quarantine
```

Review the quarantine before deleting. Adjust `--threshold` (lower = stricter).

## Notes
- Data directory is ignored by Git; add your own images locally.
- Best model checkpoint is `brain_tumor_model.keras`.
