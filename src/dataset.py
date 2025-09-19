import tensorflow as tf
import os
from pathlib import Path
from typing import List, Tuple

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


def _load_filelist(paths_file: str) -> List[str]:
    with open(paths_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.strip().startswith("#")]
    return lines


def _infer_class_from_path(p: Path) -> str:
    # Expect .../<class_name>/<filename>
    # Works for both Training and Testing trees
    return p.parent.name


def _make_tf_dataset(image_paths: List[str], labels: List[int], shuffle: bool) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    def _load_and_preprocess(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, IMG_SIZE, antialias=True)
        img = tf.cast(img, tf.float32)  # model will handle normalization/standardization
        return img, label

    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000, seed=123)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def _load_from_filelists(base_dir: str, splits_dir: str = "data/splits") -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    train_list = os.path.join(splits_dir, "train.txt")
    val_list = os.path.join(splits_dir, "val.txt")
    test_list = os.path.join(splits_dir, "test.txt")
    if not (os.path.exists(train_list) and os.path.exists(val_list) and os.path.exists(test_list)):
        raise FileNotFoundError("Expected train.txt, val.txt, and test.txt in data/splits/ for filelist loading.")

    # Read relative or absolute paths
    def resolve_paths(lines: List[str]) -> List[str]:
        resolved = []
        for ln in lines:
            p = Path(ln)
            if not p.is_absolute():
                p = Path(base_dir) / ln
            resolved.append(str(p))
        return resolved

    train_lines = resolve_paths(_load_filelist(train_list))
    val_lines = resolve_paths(_load_filelist(val_list))
    test_lines = resolve_paths(_load_filelist(test_list))

    # Infer class names from the file paths and build mapping
    def classes_from_paths(lines: List[str]) -> List[str]:
        cls = sorted(list({_infer_class_from_path(Path(p)) for p in lines}))
        return cls

    all_classes = sorted(list({*_classes for _classes in [set(classes_from_paths(train_lines)), set(classes_from_paths(val_lines)), set(classes_from_paths(test_lines))]}))
    class_to_index = {c: i for i, c in enumerate(all_classes)}

    def to_labels(lines: List[str]) -> List[int]:
        return [class_to_index[_infer_class_from_path(Path(p))] for p in lines]

    train_labels = to_labels(train_lines)
    val_labels = to_labels(val_lines)
    test_labels = to_labels(test_lines)

    train_ds = _make_tf_dataset(train_lines, train_labels, shuffle=True)
    val_ds = _make_tf_dataset(val_lines, val_labels, shuffle=False)
    test_ds = _make_tf_dataset(test_lines, test_labels, shuffle=False)

    print("Class names:", all_classes)
    return train_ds, val_ds, test_ds, all_classes


def load_datasets(base_dir: str = "data/dataset"):
    """
    Load datasets either from directory structure (default) or from file lists in data/splits.
    Provide patient-wise splits by creating:
      data/splits/train.txt, val.txt, test.txt
    each containing one image path per line (relative to base_dir or absolute). Labels are inferred from the
    parent directory name (glioma/meningioma/notumor/pituitary).
    """
    splits_dir = "data/splits"
    if os.path.exists(os.path.join(splits_dir, "train.txt")):
        return _load_from_filelists(base_dir=base_dir, splits_dir=splits_dir)

    # Fallback: directory-based loader with validation split
    train_dir = f"{base_dir}/Training"
    test_dir = f"{base_dir}/Testing"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
