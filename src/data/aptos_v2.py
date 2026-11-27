from __future__ import annotations

import os

import numpy as np
import pandas as pd

IMAGES_FILE = "aptos_v2_u8.npy"
META_FILE = "aptos_v2_meta.csv"
KAGGLE_ROOTS = ("/kaggle/input", "/kaggle/working")
BACKGROUND_MAX = 12.0


def data_roots():
    env = os.environ.get("RG_DATA_ROOTS")
    return tuple(p for p in env.split(os.pathsep) if p) if env else KAGGLE_ROOTS


def find_file(name: str, roots=None, required: bool = True):
    roots = roots or data_roots()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    if required:
        raise FileNotFoundError(f"{name} not found under {roots}")
    return None


def load(data_dir: str | None = None):
    img_path = os.path.join(data_dir, IMAGES_FILE) if data_dir else find_file(IMAGES_FILE)
    meta_path = os.path.join(os.path.dirname(img_path), META_FILE)
    X = np.load(img_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    assert len(meta) == len(X), f"meta rows {len(meta)} != images {len(X)}"
    assert X.dtype == np.uint8, X.dtype
    return X, meta


def check_splits(meta: pd.DataFrame) -> None:
    used = meta[meta.split != "dropped"]
    per_group = used.groupby("dup_group").split.nunique()
    assert (per_group <= 1).all(), f"dup groups crossing splits: {per_group[per_group > 1].index[:10].tolist()}"
    tr = used[used.split == "train"]
    per_group_fold = tr.groupby("dup_group").fold.nunique()
    assert (per_group_fold <= 1).all(), "dup groups crossing folds"
    assert meta.id_code.is_unique


def fold_indices(meta: pd.DataFrame, fold: int):
    tr = meta.index[(meta.split == "train") & (meta.fold != fold)].to_numpy()
    va = meta.index[(meta.split == "train") & (meta.fold == fold)].to_numpy()
    return tr, va


def test_indices(meta: pd.DataFrame):
    return meta.index[meta.split == "test"].to_numpy()


def augment_batch(imgs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.empty_like(imgs)
    for j in range(len(imgs)):
        im = imgs[j]
        background = im.max(axis=-1, keepdims=True) <= BACKGROUND_MAX
        if rng.random() < 0.5:
            im = im[:, ::-1]
            background = background[:, ::-1]
        if rng.random() < 0.5:
            im = im[::-1, :]
            background = background[::-1, :]
        k = int(rng.integers(4))
        im, background = np.rot90(im, k), np.rot90(background, k)
        im = im * rng.uniform(0.9, 1.1) + rng.uniform(-10, 10)
        out[j] = np.where(background, 0.0, np.clip(im, 0, 255))
    return out


def tta_views(imgs: np.ndarray):
    return [imgs, imgs[:, :, ::-1], imgs[:, ::-1, :], imgs[:, ::-1, ::-1]]


class StackedArrays:

    def __init__(self, *arrays):
        assert len({a.shape[1:] for a in arrays}) == 1, "all arrays need the same image size"
        self.arrays = arrays
        self.offsets = np.cumsum([0] + [len(a) for a in arrays])
        self.shape = (int(self.offsets[-1]),) + tuple(arrays[0].shape[1:])
        self.dtype = arrays[0].dtype

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        if np.isscalar(idx):
            k = int(np.searchsorted(self.offsets, idx, side="right")) - 1
            return self.arrays[k][int(idx) - self.offsets[k]]
        idx = np.asarray(idx, dtype=np.int64)
        out = np.empty((len(idx),) + self.shape[1:], self.dtype)
        src = np.searchsorted(self.offsets, idx, side="right") - 1
        for k, a in enumerate(self.arrays):
            m = src == k
            if m.any():
                out[m] = a[idx[m] - self.offsets[k]]
        return out


def load_eyepacs(part: str, data_dir: str | None = None):
    img_path = os.path.join(data_dir, f"eyepacs_{part}_u8.npy") if data_dir else find_file(f"eyepacs_{part}_u8.npy")
    X = np.load(img_path, mmap_mode="r")
    meta = pd.read_csv(os.path.join(os.path.dirname(img_path), f"eyepacs_{part}_meta.csv"), dtype={"patient": str})
    assert len(meta) == len(X)
    return X, meta


def make_dataset(X, indices, labels=None, sample_weights=None, batch_size: int = 32,
                 training: bool = False, augment: bool = False, seed: int = 42):
    import tensorflow as tf

    indices = np.asarray(indices, dtype=np.int64)
    S = X.shape[1]
    aug_rng = np.random.default_rng(seed)

    tensors = [indices]
    if labels is not None:
        tensors.append(np.asarray(labels))
    if sample_weights is not None:
        tensors.append(np.asarray(sample_weights, dtype=np.float32))
    ds = tf.data.Dataset.from_tensor_slices(tuple(tensors))
    if training:
        ds = ds.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)

    def _load(idx):
        order = np.argsort(idx)
        imgs = np.empty((len(idx), S, S, 3), np.float32)
        imgs[order] = X[np.sort(idx)]
        return augment_batch(imgs, aug_rng) if augment else imgs

    def load_batch(idx, *rest):
        images = tf.numpy_function(_load, [idx], tf.float32)
        images.set_shape([None, S, S, 3])
        return (images, *rest) if rest else images

    ds = ds.batch(batch_size).map(load_batch, num_parallel_calls=1)
    return ds.prefetch(2)


def predict_memmap(model, X, indices, batch_size: int = 64, tta: bool = False) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    outs = []
    for i in range(0, len(indices), batch_size):
        b = np.asarray(X[np.sort(indices[i:i + batch_size])], dtype=np.float32)
        b = b[np.argsort(np.argsort(indices[i:i + batch_size]))]
        n = len(b)
        if n < batch_size:
            b = np.concatenate([b, np.zeros((batch_size - n,) + b.shape[1:], b.dtype)])
        views = tta_views(b) if tta else [b]
        outs.append(np.mean([np.asarray(model.predict_on_batch(np.ascontiguousarray(v))) for v in views], axis=0)[:n])
    return np.concatenate(outs, axis=0)
