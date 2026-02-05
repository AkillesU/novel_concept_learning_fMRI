#!/usr/bin/env python3
"""
Generate trial-wise "ground-truth" ROI patterns from CORnet-RT (IT) features,
with within-class linear convergence toward class exemplars.

Implements your spec:
- CORnet-RT, use IT output tensor
- ImageNet preprocessing (resize shorter side 256, center crop 224, normalize)
- Multiple convergence images per class: average IT features within each class folder
- Dimensionality reduction to 200 "voxel" dimensions using PCA
- Variant 2: interpolate in the 200D space
- Within-class scheduled linear interpolation, max lambda = 0.5 (50% toward convergence)

Design CSV must contain:
  - ObjectSpace (class label)
  - image_file (filename for the trial's image)
Optionally contains run_id/trial_id (recommended).

Directory layout expected:
  --img_dir
      <ObjectSpace>/<image_file>   (preferred)
    or --img_dir/<image_file>      (flat)
  --conv_parent
      <ObjectSpace>/*.png|*.jpg|...

Outputs:
  - NPZ with:
      patterns_enc_200 : (n_trials, 200) final encoding patterns after convergence
      patterns_task_200: (n_trials, 200) trial task-image patterns (pre-convergence)
      patterns_conv_200: (n_classes, 200) class convergence target patterns
      class_names       : (n_classes,)
      lambdas           : (n_trials,) lambda used per trial
      trial_meta        : structured arrays for run_id, trial_id, ObjectSpace, image_path

Usage:
  python cornet_it_patterns.py \
    --design_csv path/to/design.csv \
    --img_dir path/to/task_images \
    --conv_parent path/to/convergence_parent \
    --out_npz patterns_from_cornet.npz \
    --n_components 200 \
    --max_lambda 0.5 \
    --schedule_scope session
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# torch/torchvision are required
import torch
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ----------------------------
# Image path resolution
# ----------------------------
def resolve_image_path(img_dir: Path, objectspace: str, image_file: str) -> Path:
    """Match the Psychopy runner's robust logic."""
    img_dir = Path(img_dir)
    # 1) subfolder layout
    p1 = img_dir / str(objectspace) / str(image_file)
    if p1.exists():
        return p1
    # 2) flat layout
    p2 = img_dir / str(image_file)
    if p2.exists():
        return p2
    # 3) recursive search by basename
    target = str(image_file)
    for root, _, files in os.walk(img_dir):
        if target in files:
            return Path(root) / target
    raise FileNotFoundError(f"Could not locate image_file='{image_file}' for ObjectSpace='{objectspace}' under {img_dir}")


def list_images_in_folder(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    out = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


# ----------------------------
# CORnet-RT loading + IT extraction
# ----------------------------
def load_cornet_rt(device: str = "cpu"):
    """
    Tries a few common ways to load CORnet-RT.
    You may need to install the CORnet repo (dicarlolab/CORnet) locally.

    Priority:
      1) import cornet and use cornet.cornet_rt()
      2) torch.hub.load('dicarlolab/CORnet', 'cornet_rt')  (requires internet/cache)
    """
    device = torch.device(device)
    model = None

    # 1) Local import
    try:
        from cornet import cornet_rt
        model = cornet_rt(pretrained=True)            
    except ImportError:
        model = None

    # 2) Torch hub fallback
    if model is None:
        try:
            model = torch.hub.load("dicarlolab/CORnet", "cornet_rt", pretrained=True)
        except Exception as e:
            raise RuntimeError(
                "Could not load CORnet-RT. Install dicarlolab/CORnet locally (recommended), "
                "or ensure torch.hub can fetch it.\n"
                f"Original error: {e}"
            )

    model = model.to(device)
    model.eval()
    return model


class ITExtractor:
    """
    Forward-hook extractor for CORnet-RT IT features.
    Handles DataParallel/DistributedDataParallel (model.module).
    Can hook either the whole IT block or IT.output.
    """
    def __init__(self, model, hook_output_node: bool = True):
        self.model = model
        self._buf = None

        # Unwrap DataParallel / DDP
        base = model
        if hasattr(model, "module"):
            base = model.module

        if not hasattr(base, "IT"):
            raise AttributeError(
                "Base model has no attribute 'IT'. "
                "If you're using a different CORnet implementation, inspect model structure."
            )

        # Choose hook target:
        # - base.IT.output is an Identity() in your printout
        # - base.IT is the full CORblock_RT module
        target = base.IT.output if (hook_output_node and hasattr(base.IT, "output")) else base.IT

        self.hook = target.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        # DataParallel can return output as Tensor, tuple, or dict depending on model;
        # we try to select the tensor-like part.
        if isinstance(output, (tuple, list)):
            # pick first tensor-like element
            for o in output:
                if torch.is_tensor(o):
                    self._buf = o
                    return
            self._buf = output[0]
        elif isinstance(output, dict):
            # common key patterns; otherwise take first tensor value
            for k in ("out", "output", "x", "feat", "features"):
                if k in output and torch.is_tensor(output[k]):
                    self._buf = output[k]
                    return
            for v in output.values():
                if torch.is_tensor(v):
                    self._buf = v
                    return
            self._buf = next(iter(output.values()))
        else:
            self._buf = output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self._buf = None
        _ = self.model(x)
        if self._buf is None:
            raise RuntimeError("IT hook did not capture output.")
        return self._buf

    def close(self):
        try:
            self.hook.remove()
        except Exception:
            pass



def imagenet_preprocess():
    # CORnet uses standard ImageNet preprocessing
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def it_vector_for_image(path: Path, extractor: ITExtractor, tfm, device: str) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    x = tfm(im).unsqueeze(0).to(device)  # (1,3,224,224)
    it = extractor(x)  # e.g. (1,C,H,W) or (1,C)
    it = it.squeeze(0)

    # Pool spatial dims if present
    if it.ndim == 3:
        it = it.mean(dim=(1, 2))  # (C,)
    elif it.ndim == 1:
        pass
    else:
        # handle (C,H,W) already squeezed? or unexpected
        it = it.reshape(it.shape[0], -1).mean(dim=1)

    v = it.detach().cpu().float().numpy()
    return v


# ----------------------------
# PCA to 200D (sklearn if available; else SVD)
# ----------------------------
def pca_fit_transform(X: np.ndarray, n_components: int, whiten: bool = False) -> Tuple[np.ndarray, Dict]:
    X = np.asarray(X, float)
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    # before trying sklearn PCA, validate the input and clamp n_components
    X = np.asarray(X, float)
    if not np.isfinite(X).all():
        raise ValueError("Input to PCA contains NaN or infinite values. Inspect task/conv vectors.")
    n_samples, n_features = X.shape
    max_possible = min(n_samples, n_features)
    if max_possible <= 0:
        raise ValueError("Empty data matrix for PCA.")
    k = min(n_components, max_possible)
    if k < n_components:
        print(f"[WARN] Requested n_components={n_components} reduced to {k} (min(n_samples,n_features)={max_possible}).")
    try:
        from sklearn.decomposition import PCA  # type: ignore
        pca = PCA(n_components=k, whiten=whiten, svd_solver="auto", random_state=0)
        Z = pca.fit_transform(X - X.mean(axis=0, keepdims=True))
        meta = {
            "mean": X_mean.squeeze(0),
            "components": pca.components_,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "whiten": whiten,
        }
        return Z, meta
    except Exception:
        # SVD fallback (no whitening)
        # Xc = U S Vt, projection onto first k PCs = U[:, :k] * S[:k]
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(n_components, Vt.shape[0])
        Z = U[:, :k] * S[:k]
        meta = {
            "mean": X_mean.squeeze(0),
            "components": Vt[:k, :],
            "singular_values": S[:k],
            "whiten": False,
            "note": "sklearn not available; used numpy SVD fallback",
        }
        return Z, meta


def pca_transform(X: np.ndarray, meta: Dict) -> np.ndarray:
    X = np.asarray(X, float)
    Xc = X - meta["mean"][None, :]
    comps = meta["components"]
    Z = Xc @ comps.T
    if meta.get("whiten", False):
        # Z already whitened by sklearn; here we approximate
        ev = meta.get("explained_variance", None)
        if ev is not None:
            Z = Z / np.sqrt(ev[None, :] + 1e-12)
    return Z



# ----------------------------
# RDM utilities (Representational Dissimilarity Matrices)
# ----------------------------
def _safe_rowwise_zscore(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-score each row; rows with ~0 variance become zeros."""
    X = np.asarray(X, float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def rdm_correlation_distance(patterns: np.ndarray, zscore_rows: bool = False) -> np.ndarray:
    """
    Correlation-distance RDM: D[i,j] = 1 - corr(pattern_i, pattern_j)
    patterns: (n_items, n_features)
    Returns: (n_items, n_items)
    """
    X = np.asarray(patterns, float)
    if zscore_rows:
        X = _safe_rowwise_zscore(X)

    C = np.corrcoef(X)  # correlation between rows/items
    C = np.clip(C, -1.0, 1.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return D


def rdm_upper_triangle_vector(D: np.ndarray) -> np.ndarray:
    """Vectorize upper triangle (excluding diagonal) for RDM comparisons."""
    D = np.asarray(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("RDM must be square (n x n).")
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def pearsonr_np(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Pearson correlation without scipy."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())) + eps
    return float((x * y).sum() / denom)


def plot_rdms(rdm_raw: np.ndarray, rdm_pca: np.ndarray, out_png: Path | None = None, show: bool = False) -> None:
    """
    Plot RAW and post-PCA RDMs with matplotlib.
    If out_png is provided, saves the figure. If show=True, shows interactively.
    """
    rdm_raw = np.asarray(rdm_raw)
    rdm_pca = np.asarray(rdm_pca)

    # Use common scale for easier visual comparison
    vmax = float(max(np.nanmax(rdm_raw), np.nanmax(rdm_pca)))
    vmin = float(min(np.nanmin(rdm_raw), np.nanmin(rdm_pca)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(rdm_raw, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("RDM (RAW pre-PCA)\ncorrelation distance")
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("trial")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(rdm_pca, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title("RDM (post-PCA)\ncorrelation distance")
    axes[1].set_xlabel("trial")
    axes[1].set_ylabel("trial")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)

    if show:
        plt.show()

    plt.close(fig)



# ----------------------------
# Convergence schedule + interpolation
# ----------------------------
def linear_lambda_within_class(df: pd.DataFrame, max_lambda: float, scope: str = "session") -> np.ndarray:
    """
    Compute lambda per trial based on occurrence index within each ObjectSpace.
    scope:
      - 'session': occurrences counted across the whole CSV
      - 'run': occurrences reset per run_id
    """
    df = df.copy()
    if scope not in ("session", "run"):
        raise ValueError("scope must be 'session' or 'run'")

    lambdas = np.zeros(len(df), float)

    if scope == "session":
        for os_name, g in df.groupby("ObjectSpace"):
            idxs = g.index.to_list()
            n = len(idxs)
            if n <= 1:
                lambdas[idxs] = max_lambda
            else:
                for k, ix in enumerate(idxs):
                    lambdas[ix] = max_lambda * (k / (n - 1))
    else:
        if "run_id" not in df.columns:
            raise ValueError("scope='run' requires run_id in CSV")
        for (rid, os_name), g in df.groupby(["run_id", "ObjectSpace"]):
            idxs = g.index.to_list()
            n = len(idxs)
            if n <= 1:
                lambdas[idxs] = max_lambda
            else:
                for k, ix in enumerate(idxs):
                    lambdas[ix] = max_lambda * (k / (n - 1))

    return lambdas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design_csv", required=True)
    ap.add_argument("--img_dir", required=True, help="Parent directory containing task images.")
    ap.add_argument("--conv_parent", required=True, help="Parent directory containing class subfolders of convergence images.")
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--plot_rdms", action="store_true", help="Plot RAW and post-PCA RDMs (saves PNG; use --show_plots to display).")
    ap.add_argument("--show_plots", action="store_true", help="If set, display matplotlib windows (in addition to saving).")
    ap.add_argument("--rdm_png", type=str, default="", help="Optional output PNG path for RDM plots. Default: <out_npz>_rdms.png")

    ap.add_argument("--n_components", type=int, default=200)
    ap.add_argument("--max_lambda", type=float, default=0.5)
    ap.add_argument("--schedule_scope", choices=["session", "run"], default="session")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32, help="(Not used yet; placeholder for future batching).")
    args = ap.parse_args()

    design_csv = Path(args.design_csv)
    img_dir = Path(args.img_dir)
    conv_parent = Path(args.conv_parent)
    out_npz = Path(args.out_npz)

    df = pd.read_csv(design_csv)
    if "ObjectSpace" not in df.columns or "image_file" not in df.columns:
        raise ValueError("design_csv must include columns: ObjectSpace, image_file")

    # stable order (trial order) as in your GLM designs
    if "img_onset" in df.columns:
        df = df.sort_values(["run_id", "img_onset"]) if "run_id" in df.columns else df.sort_values("img_onset")
    else:
        df = df.reset_index(drop=True)

    # Load model + extractor
    model = load_cornet_rt(device=args.device)
    extractor = ITExtractor(model, hook_output_node=True)  # hooks model.module.IT.output if available
    tfm = imagenet_preprocess()

    # 1) Task image IT vectors per trial
    task_paths = []
    task_vecs = []
    for _, row in df.iterrows():
        os_name = str(row["ObjectSpace"])
        image_file = str(row["image_file"])
        p = resolve_image_path(img_dir, os_name, image_file)
        task_paths.append(str(p))
        task_vecs.append(it_vector_for_image(p, extractor, tfm, args.device))
    task_vecs = np.vstack(task_vecs)  # (n_trials, d)

    # --- RDM from RAW (pre-PCA) trial IT vectors ---
    rdm_raw = rdm_correlation_distance(task_vecs, zscore_rows=False)

    # 2) Convergence vectors per class: average over images in each class folder
    class_names = sorted({str(x) for x in df["ObjectSpace"].unique().tolist()})
    conv_vecs = []
    conv_counts = []
    for os_name in class_names:
        folder = conv_parent / os_name
        imgs = list_images_in_folder(folder)
        if len(imgs) == 0:
            raise FileNotFoundError(f"No convergence images found for class '{os_name}' in {folder}")
        vecs = np.vstack([it_vector_for_image(p, extractor, tfm, args.device) for p in imgs])
        conv_vecs.append(vecs.mean(axis=0))
        conv_counts.append(len(imgs))
    conv_vecs = np.vstack(conv_vecs)  # (n_classes, d)

    extractor.close()

    # 3) PCA basis fit on union of task+conv vectors
    X_all = np.vstack([task_vecs, conv_vecs])
    Z_all, pca_meta = pca_fit_transform(X_all, n_components=args.n_components, whiten=False)

    # debug: inspect data before PCA
    print("X_all.shape:", X_all.shape)                 # (n_samples, n_features)
    print("requested n_components:", args.n_components)
    print("min(n_samples, n_features):", min(X_all.shape[0], X_all.shape[1]))

    Z_task = Z_all[: task_vecs.shape[0], :]
    Z_conv = Z_all[task_vecs.shape[0] :, :]

    # --- RDM from PCA (post-PCA) trial vectors ---
    rdm_pca = rdm_correlation_distance(Z_task, zscore_rows=False)
    v_raw = rdm_upper_triangle_vector(rdm_raw)
    v_pca = rdm_upper_triangle_vector(rdm_pca)
    rdm_raw_vs_pca_r = pearsonr_np(v_raw, v_pca)

    np.set_printoptions(precision=4, suppress=True, threshold=200, edgeitems=3, linewidth=140)
    print("\n--- RDM (RAW pre-PCA; correlation distance) ---")
    print(f"rdm_raw shape = {rdm_raw.shape}")
    print(rdm_raw)
    print("\n--- RDM (PCA post-PCA; correlation distance) ---")
    print(f"rdm_pca shape = {rdm_pca.shape}")
    print(rdm_pca)
    print(f"\nRDM similarity (Pearson r, upper triangle): {rdm_raw_vs_pca_r:.6f}\n")

    if args.plot_rdms:
        default_png = Path(str(args.out_npz)).with_suffix("")
        default_png = default_png.parent / (default_png.name + "_rdms.png")
        out_png = Path(args.rdm_png) if args.rdm_png else default_png
        plot_rdms(rdm_raw, rdm_pca, out_png=out_png, show=args.show_plots)
        print(f"Saved RDM plot: {out_png}")

    # 4) Build per-trial convergence target in 200D by class lookup
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    conv_target = np.zeros_like(Z_task)
    for i in range(len(df)):
        conv_target[i] = Z_conv[class_to_idx[str(df.iloc[i]["ObjectSpace"])], :]

    # 5) Within-class linear schedule with max_lambda
    lambdas = linear_lambda_within_class(df, max_lambda=args.max_lambda, scope=args.schedule_scope)

    # 6) Interpolate in 200D (variant 2)
    patterns_task_200 = Z_task
    patterns_enc_200 = (1.0 - lambdas[:, None]) * Z_task + lambdas[:, None] * conv_target

    # Optional: normalize vectors (keeps scale comparable across trials)
    def rownorm(X):
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    patterns_task_200 = rownorm(patterns_task_200)
    patterns_enc_200 = rownorm(patterns_enc_200)
    patterns_conv_200 = rownorm(Z_conv)

    # Trial metadata
    run_id = df["run_id"].to_numpy() if "run_id" in df.columns else np.full(len(df), -1)
    trial_id = df["trial_id"].to_numpy() if "trial_id" in df.columns else np.arange(len(df))
    obj = df["ObjectSpace"].astype(str).to_numpy()

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        patterns_enc_200=patterns_enc_200.astype(np.float32),
        patterns_task_200=patterns_task_200.astype(np.float32),
        patterns_conv_200=patterns_conv_200.astype(np.float32),
        class_names=np.array(class_names),
        conv_image_counts=np.array(conv_counts, dtype=int),
        lambdas=lambdas.astype(np.float32),
        run_id=run_id,
        trial_id=trial_id,
        ObjectSpace=obj,
        task_image_paths=np.array(task_paths),
        pca_mean=pca_meta["mean"].astype(np.float32),
        pca_components=pca_meta["components"].astype(np.float32),
        pca_meta_str=str({k: v for k, v in pca_meta.items() if k not in ("mean", "components")}),
        rdm_raw=rdm_raw.astype(np.float32),
        rdm_pca=rdm_pca.astype(np.float32),
        rdm_raw_vs_pca_r=np.array([rdm_raw_vs_pca_r], dtype=np.float32),
    )
    print(f"Wrote: {out_npz}")
    print(f"Trials: {len(df)} | Classes: {len(class_names)} | IT dim: {task_vecs.shape[1]} -> PCA {args.n_components}")
    print(f"Schedule scope: {args.schedule_scope} | max_lambda: {args.max_lambda}")
    print("Convergence images per class:", dict(zip(class_names, conv_counts)))


if __name__ == "__main__":
    main()
