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

## Notes
- Data directory is ignored by Git; add your own images locally.
- Best model checkpoint is `brain_tumor_model.keras`.
