"""Recompute domain AUC + frac-sparse for the paper diagnostics table (tab:diag).

Reuses cached GAP feats (dl/cache) + the exact pipeline protocol:
StandardScaler -> PCA(50, seed 42) on train+val+test stacked, then
utils.embedding_metrics.train_test_shift (seed 42, cap 30k, k=20).
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/acz25/repos/eo_fm/src")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import utils.embedding_metrics as em

CACHE = "/maps/acz25/phd-thesis-data/output/lcz-classification/dl/cache"
out = {}
for emb in ["GeoTessera_v1.1_global", "AlphaEarthCoop", "EmbeddedSeamless"]:
    Xs = [np.load(f"{CACHE}/global_{emb}_gap_{s}_feats.npy") for s in ("train", "val", "test")]
    splits = np.concatenate([np.full(len(x), s) for x, s in zip(Xs, ("train", "val", "test"))])
    X = np.vstack(Xs).astype(np.float32)
    meta = pd.DataFrame({"split": splits, "lcz_name": np.concatenate(
        [np.load(f"{CACHE}/global_{emb}_gap_{s}_labels.npy") for s in ("train", "val", "test")]).astype(str)})
    Xp = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=min(50, Xp.shape[1]), random_state=42).fit_transform(Xp)
    shift = em.train_test_shift(Xp, meta, k=20, cap=30_000, seed=42)
    out[emb] = dict(domain_auc=round(shift["domain_auc"], 4),
                    frac_sparse=round(shift["frac_test_in_sparse_regions"], 4))
    print(emb, out[emb], flush=True)

json.dump(out, open("/tmp/claude-1016/-home-acz25-repos-eo-fm/7273584e-46e8-434b-a736-eec73d645d3c/scratchpad/domain_auc_recompute.json", "w"), indent=1)
print("DONE")
