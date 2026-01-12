#!/usr/bin/env python3
"""
infer_complete.py

The Definitive Pipeline.
1. Metrics:
   - Inter-Model Agreement (Do models agree on the same image?)
   - Intra-Model Consistency (Is a single model stable across the group?)
   - Ensemble Consistency (Is the group average stable?)
   - Outlier Detection (Which model is the 'black sheep'?)
2. Slices: Top-20, Mid-20 (Mass), Bottom-20.
3. Visualization: Images + Transparent Gaussian + Metrics Sidebar.
"""

import argparse
import csv
import sys
import os
import math
import itertools
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision
from torchvision import models
import numpy as np

# Visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

# ==========================================
# 1. METRICS LOGIC
# ==========================================

def get_slice_indices(probs: torch.Tensor, slice_type: str, k: int) -> Set[int]:
    """Returns set of indices for Top/Mid/Bot slices."""
    numel = probs.numel(); k = min(k, numel)
    sorted_vals, sorted_indices = torch.sort(probs, descending=True)
    
    if slice_type == "top":
        return set(sorted_indices[:k].tolist())
    elif slice_type == "bot":
        return set(sorted_indices[-k:].tolist())
    elif slice_type == "mid":
        # Mass-based Median
        cumsum = torch.cumsum(sorted_vals, dim=0)
        mass_indices = (cumsum > 0.5).nonzero(as_tuple=True)[0]
        center_idx = mass_indices[0].item() if len(mass_indices) > 0 else numel // 2
        start = max(0, center_idx - (k // 2))
        end = min(numel, start + k)
        if end - start < k: start = max(0, end - k)
        return set(sorted_indices[start:end].tolist())
    return set()

def calculate_set_similarity(sets: List[Set[int]]) -> float:
    """Average Pairwise Jaccard Similarity."""
    if len(sets) < 2: return 1.0
    scores = []
    for s1, s2 in itertools.combinations(sets, 2):
        intersection = len(s1.intersection(s2))
        union = len(s1.union(s2))
        scores.append(intersection / union if union > 0 else 0.0)
    return sum(scores) / len(scores)

def detect_divergent_model(prob_list: List[torch.Tensor], model_names: List[str]) -> str:
    """Identifies which model is furthest from the group consensus (Euclidean distance)."""
    if len(prob_list) < 2: return "None"
    stack = torch.stack(prob_list, dim=0) # [Models, Classes]
    consensus = stack.mean(dim=0)
    dists = torch.norm(stack - consensus, p=2, dim=1) # Distance per model
    max_dist, max_idx = torch.max(dists, dim=0)
    return f"{model_names[max_idx.item()]} ({max_dist.item():.2f})"

# ==========================================
# 2. HELPERS
# ==========================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

def discover_images(root: Path, recursive: bool) -> List[Path]:
    if recursive: return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")

def parse_filename_features(path: Path) -> Tuple[str, Optional[float], Optional[float]]:
    stem = path.stem
    parts = stem.split('_')
    if len(parts) >= 3:
        try: return parts[0], float(parts[1]), float(parts[2])
        except: return parts[0], None, None
    if len(parts) >= 1: return parts[0], None, None
    return "Unknown", None, None

def compute_weight(f1, f2, tf1, tf2, sigma, uniform):
    if uniform: return 1.0 if (f1 is not None) else 0.0
    if f1 is None: return 0.0
    dist_sq = (f1 - tf1)**2 + (f2 - tf2)**2
    if sigma <= 0: return 1.0 if dist_sq == 0 else 0.0
    return math.exp(-dist_sq / (2 * sigma**2))

def format_k(values, indices, labels):
    return [f"{labels[i] if 0<=i<len(labels) else str(i)} ({v:.4f})" for v, i in zip(values.tolist(), indices.tolist())]

# ==========================================
# 3. MODEL LOADING
# ==========================================

def resolve_model_and_weights(model_name: str):
    if hasattr(models, "get_model"):
        try:
            w = models.get_model_weights(model_name).DEFAULT
            return models.get_model(model_name, weights=w), w.transforms(), w.meta.get("categories")
        except: pass
    ctor = getattr(models, model_name)
    if hasattr(ctor, "Weights"):
        w = ctor.Weights.DEFAULT
        return ctor(weights=w), w.transforms(), w.meta.get("categories")
    return ctor(pretrained=True), torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), None

# ==========================================
# 4. PLOTTING & STATS
# ==========================================

def get_stats_slices_mass_display(scores: torch.Tensor, k: int):
    # For display text only
    numel = scores.numel(); k = min(k, numel)
    sorted_vals, sorted_indices = torch.sort(scores, descending=True)
    top_vals, top_idx = sorted_vals[:k], sorted_indices[:k]
    bot_vals, bot_idx = sorted_vals[-k:], sorted_indices[-k:]
    cumsum = torch.cumsum(sorted_vals, dim=0)
    mass_indices = (cumsum > 0.5).nonzero(as_tuple=True)[0]
    center_idx = mass_indices[0].item() if len(mass_indices) > 0 else numel // 2
    start = max(0, center_idx - (k // 2))
    end = min(numel, start + k)
    if end - start < k: start = max(0, end - k)
    med_vals, med_idx = sorted_vals[start:end], sorted_indices[start:end]
    return (top_vals, top_idx), (med_vals, med_idx), (bot_vals, bot_idx)

def generate_group_plot(group_id, items, target_f1, target_f2, sigma, uniform,
                        top_res, med_res, bot_res, metrics_str, outlier_str, output_dir):
    if not VIZ_AVAILABLE: return
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
    
    t_str = "UNIFORM" if uniform else f"Tgt:({target_f1},{target_f2}) Sig:{sigma}"
    ax.set_title(f"Group {group_id} | {t_str}\nOutlier Model: {outlier_str}", fontsize=14)
    
    for item in items:
        if item['f1'] is None: continue
        try:
            img_obj = Image.open(item['path']).convert("RGB")
            img_obj.thumbnail((128, 128)) 
            im_box = OffsetImage(np.array(img_obj), zoom=0.6)
            ab = AnnotationBbox(im_box, (item['f1'], item['f2']), frameon=True, pad=0.2, bboxprops=dict(edgecolor='gray', alpha=0.5))
            ax.add_artist(ab)
            ax.scatter(item['f1'], item['f2'], c='red', s=20, zorder=10)
        except: pass

    if not uniform:
        x = np.linspace(-0.1, 1.1, 100); y = np.linspace(-0.1, 1.1, 100)
        X, Y = np.meshgrid(x, y)
        dist_sq = (X - target_f1)**2 + (Y - target_f2)**2
        Z = np.exp(-dist_sq / (2 * sigma**2))
        cmap = plt.get_cmap('viridis')
        rgba_img = cmap(Z); rgba_img[..., 3] = Z * 0.6 
        im = ax.imshow(rgba_img, extent=[-0.1, 1.1, -0.1, 1.1], origin='lower', aspect='auto', zorder=20)

    def clean(l): return "\n".join(l[:5])
    txt = (f"METRICS (Jaccard 0-1):\n{metrics_str}\n\n{'='*30}\n\n"
           f"TOP 5:\n{clean(top_res)}\n\n{'-'*20}\n\n"
           f"MID 5:\n{clean(med_res)}\n\n{'-'*20}\n\n"
           f"BOT 5:\n{clean(bot_res)}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc')
    ax.text(1.15, 1.0, txt, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)
    plt.subplots_adjust(right=0.70) 
    plt.savefig(os.path.join(output_dir, f"plot_group_{group_id}.png"), dpi=100)
    plt.close(fig)

# ==========================================
# 5. AGGREGATION LOGIC
# ==========================================

def aggregate_group_metrics(items, model_count, top_k):
    """Calculates the 3x3 Metrics Matrix."""
    
    # 1. Weighted Ensemble Probability
    tensors = [x['ensemble_probs'] for x in items]
    weights = [x['weight'] for x in items]
    
    stack = torch.stack(tensors, dim=0).to(tensors[0].device)
    w_tensor = torch.tensor(weights, device=stack.device, dtype=torch.float32).view(-1, 1)

    sum_w = w_tensor.sum()
    sum_sq_w = (w_tensor ** 2).sum()
    eff = (sum_w ** 2) / sum_sq_w if sum_sq_w > 0 else 0.0
    
    if sum_w == 0: agg_prob = torch.ones(stack.shape[1], device=stack.device)/stack.shape[1]
    else: agg_prob = (stack * (w_tensor / sum_w)).sum(dim=0)
    agg_prob = agg_prob / agg_prob.sum()

    # 2. Metrics
    metrics = {}
    for sl in ['top', 'mid', 'bot']:
        # A. Inter-Model Agreement (Avg across images)
        img_agrees = []
        for item in items:
            sets = [get_slice_indices(p, sl, top_k) for p in item['individual_probs']]
            img_agrees.append(calculate_set_similarity(sets))
        metrics[f"{sl}_model_agree"] = np.average(img_agrees, weights=weights) if sum_w > 0 else 0.0

        # B. Intra-Model Consistency (Avg across models)
        mod_consts = []
        for m_idx in range(model_count):
            m_probs = [item['individual_probs'][m_idx] for item in items]
            m_sets = [get_slice_indices(p, sl, top_k) for p in m_probs]
            mod_consts.append(calculate_set_similarity(m_sets))
        metrics[f"{sl}_model_const"] = sum(mod_consts) / len(mod_consts)

        # C. Ensemble Consistency
        ens_probs = [item['ensemble_probs'] for item in items]
        ens_sets = [get_slice_indices(p, sl, top_k) for p in ens_probs]
        metrics[f"{sl}_ens_const"] = calculate_set_similarity(ens_sets)

    # 3. Outlier Model Detection (Most common outlier in group)
    outlier_counts = {}
    for item in items:
        name = item['outlier_info'].split(' ')[0]
        outlier_counts[name] = outlier_counts.get(name, 0) + 1
    most_common_outlier = max(outlier_counts, key=outlier_counts.get) if outlier_counts else "None"

    return agg_prob, eff, metrics, most_common_outlier

# ==========================================
# 6. MAIN
# ==========================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=str)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--target-f1", type=float, default=0.5)
    ap.add_argument("--target-f2", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--uniform", action="store_true")
    ap.add_argument("--plot-dir", type=str, default=None)
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--recursive", action="store_true")
    
    args = ap.parse_args()

    if args.plot_dir and VIZ_AVAILABLE and not os.path.exists(args.plot_dir): os.makedirs(args.plot_dir)
    img_paths = discover_images(Path(args.folder), args.recursive)
    if not img_paths: sys.exit("[ERROR] No images.")

    loaded_models = []
    for m in args.models:
        print(f"[INFO] Loading {m}...")
        mod, pre, cats = resolve_model_and_weights(m)
        mod.to(args.device).eval()
        loaded_models.append({'name': m, 'model': mod, 'pre': pre, 'cats': cats})

    labels = loaded_models[0]['cats'] if loaded_models[0]['cats'] else [f"Class_{i}" for i in range(1000)]
    model_names = [m['name'] for m in loaded_models]

    # --- INFERENCE ---
    results_db = {}
    with torch.no_grad():
        for m_idx, rec in enumerate(loaded_models):
            batch_t, batch_p = [], []
            def flush():
                if not batch_t: return
                t = torch.stack(batch_t).to(args.device)
                probs = F.softmax(rec['model'](t), dim=1).cpu()
                for i, p in enumerate(probs):
                    k = str(batch_p[i])
                    if k not in results_db:
                        g, f1, f2 = parse_filename_features(batch_p[i])
                        w = compute_weight(f1, f2, args.target_f1, args.target_f2, args.sigma, args.uniform)
                        results_db[k] = {'group': g, 'f1': f1, 'f2': f2, 'weight': w, 'path': batch_p[i], 
                                         'individual_probs': [None]*len(loaded_models)}
                    results_db[k]['individual_probs'][m_idx] = p
                batch_t[:], batch_p[:] = [], []
            for p in img_paths:
                batch_t.append(rec['pre'](load_image(p))); batch_p.append(p)
                if len(batch_t) >= args.batch_size: flush()
            flush()

    # Pre-calc Ensemble & Outlier per Image
    group_data = {}
    for key, data in results_db.items():
        if any(x is None for x in data['individual_probs']): continue
        
        # Ensemble Probs
        stack = torch.stack(data['individual_probs'])
        ensemble_p = stack.mean(dim=0)
        data['ensemble_probs'] = ensemble_p / ensemble_p.sum()
        
        # Outlier Detection for this image
        data['outlier_info'] = detect_divergent_model(data['individual_probs'], model_names)
        
        gid = data['group']
        if gid not in group_data: group_data[gid] = []
        group_data[gid].append(data)

    sorted_groups = sorted(group_data.keys(), key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    if args.csv:
        f_csv = open(args.csv, "w", newline="", encoding="utf-8")
        writer = csv.writer(f_csv)
        header = ["group", "count", "eff_samples", "outlier_model"]
        for s in ['top', 'mid', 'bot']:
            header.extend([f"{s}_model_agree", f"{s}_model_const", f"{s}_ens_const"])
        header.extend(["top_labels", "mid_labels", "bot_labels"])
        writer.writerow(header)

    print(f"\n{'='*30}\n FINAL ANALYSIS \n{'='*30}")

    for gid in sorted_groups:
        items = group_data[gid]
        agg_prob, eff, metrics, outlier = aggregate_group_metrics(items, len(loaded_models), args.topk, args.target_f1, args.target_f2, args.sigma, args.uniform)
        
        (tv, ti), (mv, mi), (bv, bi) = get_stats_slices_mass_display(agg_prob, args.topk)
        top_str, med_str, bot_str = format_k(tv, ti, labels), format_k(mv, mi, labels), format_k(bv, bi, labels)

        print(f"\nGroup: {gid} | Eff: {eff:.1f} | Outlier: {outlier}")
        print("  --- Top ---")
        print(f"  Model Agree: {metrics['top_model_agree']:.2f} | Indiv Const: {metrics['top_model_const']:.2f} | Ens Const: {metrics['top_ens_const']:.2f}")
        print("  --- Mid ---")
        print(f"  Model Agree: {metrics['mid_model_agree']:.2f} | Indiv Const: {metrics['mid_model_const']:.2f} | Ens Const: {metrics['mid_ens_const']:.2f}")

        plot_metrics_str = (
            f"TOP: Agr={metrics['top_model_agree']:.2f}, I-Const={metrics['top_model_const']:.2f}, E-Const={metrics['top_ens_const']:.2f}\n"
            f"MID: Agr={metrics['mid_model_agree']:.2f}, I-Const={metrics['mid_model_const']:.2f}, E-Const={metrics['mid_ens_const']:.2f}\n"
            f"BOT: Agr={metrics['bot_model_agree']:.2f}, I-Const={metrics['bot_model_const']:.2f}, E-Const={metrics['bot_ens_const']:.2f}"
        )

        if args.plot_dir:
            generate_group_plot(gid, items, args.target_f1, args.target_f2, args.sigma, args.uniform,
                                top_str, med_str, bot_str, plot_metrics_str, outlier, args.plot_dir)
        
        if args.csv:
            row = [gid, len(items), f"{eff:.2f}", outlier]
            for s in ['top', 'mid', 'bot']:
                row.extend([f"{metrics[f'{s}_model_agree']:.4f}", f"{metrics[f'{s}_model_const']:.4f}", f"{metrics[f'{s}_ens_const']:.4f}"])
            row.extend(["; ".join(top_str), "; ".join(med_str), "; ".join(bot_str)])
            writer.writerow(row)

    if args.csv: f_csv.close()

if __name__ == "__main__":
    main()