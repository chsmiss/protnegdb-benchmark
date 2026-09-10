#!/usr/bin/env python3
"""Pure helpers for family-disjoint layerwise pair-CLS probes.

No training of PLM-interact. Linear probes are L2-regularized least squares
on frozen CLS. Distance for rho is cosine distance on L2-normalized CLS.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = l2_normalize(np.asarray(left, dtype=np.float64))
    b = l2_normalize(np.asarray(right, dtype=np.float64))
    if a.ndim == 1:
        return np.dot(a, b)
    return (a * b).sum(axis=-1)


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 1.0 - cosine(left, right)


def rho_triplet(ad: np.ndarray, ac: np.ndarray, ar: np.ndarray) -> float:
    """d(AD,AC) / (0.5 * (d(AD,AR) + d(AC,AR))) with cosine distance."""
    d_dc = float(cosine_distance(ad, ac))
    d_dr = float(cosine_distance(ad, ar))
    d_cr = float(cosine_distance(ac, ar))
    den = 0.5 * (d_dr + d_cr)
    if den <= 0.0:
        return float("nan")
    return d_dc / den


def decompose_relu_cls(cls: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z = ReLU(cls); return (z_parallel, z_orthogonal) along classifier w."""
    z = relu(np.asarray(cls, dtype=np.float64).reshape(-1))
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(w))
    if norm <= 0.0:
        raise ValueError("classifier weight has zero norm")
    u = w / norm
    z_par = float(np.dot(z, u)) * u
    z_orth = z - z_par
    return z_par, z_orth


def family_set(rows: list[dict[str, str]], keys: tuple[str, ...] = ("fam_a", "fam_c", "fam_d")) -> set[str]:
    out: set[str] = set()
    for row in rows:
        for key in keys:
            value = row.get(key, "")
            if value:
                out.add(value)
    return out


def families_disjoint(train: list[dict[str, str]], test: list[dict[str, str]]) -> bool:
    return family_set(train).isdisjoint(family_set(test))


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC. Ties contribute 0.5. Returns NaN if a class is empty."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # vectorized P(pos > neg) + 0.5 P(eq)
    greater = np.sum(pos[:, None] > neg[None, :])
    equal = np.sum(pos[:, None] == neg[None, :])
    return float((greater + 0.5 * equal) / (pos.size * neg.size))


def fit_ridge_scores(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    lam: float = 1.0,
) -> np.ndarray:
    """Standardized ridge probe. Returns test scores, not probabilities."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64) * 2.0 - 1.0
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    xt = (x_train - mu) / sd
    xe = (x_test - mu) / sd
    xt = np.concatenate([xt, np.ones((xt.shape[0], 1))], axis=1)
    xe = np.concatenate([xe, np.ones((xe.shape[0], 1))], axis=1)
    gram = xt.T @ xt
    n = gram.shape[0]
    gram.flat[:: n + 1] += lam
    weight = np.linalg.solve(gram, xt.T @ y)
    return xe @ weight


def bootstrap_auroc(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 200,
    seed: int = 20260818,
) -> dict[str, float]:
    """Cluster bootstrap over `groups` (typically triplet_id)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray(groups)
    point = auroc(y, s)
    uniq, inverse = np.unique(g, return_inverse=True)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        chosen = rng.integers(0, uniq.size, size=uniq.size)
        # resample clusters with replacement; repeat a group as often as drawn.
        parts_y = []
        parts_s = []
        for idx in chosen:
            sel = inverse == idx
            parts_y.append(y[sel])
            parts_s.append(s[sel])
        stats.append(auroc(np.concatenate(parts_y), np.concatenate(parts_s)))
    arr = np.asarray(stats, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"auroc": point, "n": int(y.size), "n_groups": int(uniq.size),
                "ci95_lo": float("nan"), "ci95_hi": float("nan")}
    return {
        "auroc": point,
        "n": int(y.size),
        "n_groups": int(uniq.size),
        "ci95_lo": float(np.quantile(arr, 0.025)),
        "ci95_hi": float(np.quantile(arr, 0.975)),
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if v == v], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def matched_random_from_rows(
    rows: list[dict[str, str]],
    known_contacts: set[frozenset[str]],
    taxon_key: str = "tax_c",
) -> dict[str, str]:
    """Deterministic taxon-wise C permutation. Keys are triplet_id -> c_random."""

    def legal(row: dict[str, str], cand: str) -> bool:
        if cand in {row["anchor"], row["d"], row["c"]}:
            return False
        if frozenset({row["anchor"], cand}) in known_contacts:
            return False
        return True

    by_taxon: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_taxon.setdefault(row.get(taxon_key, ""), []).append(row)
    out: dict[str, str] = {}
    for taxon in sorted(by_taxon):
        group = by_taxon[taxon]
        n = len(group)
        used: set[int] = set()
        pool = [row["c"] for row in group]
        chosen = [""] * n
        for i, row in enumerate(group):
            pick = None
            for prefer_derange in (True, False):
                for j, cand in enumerate(pool):
                    if j in used:
                        continue
                    if prefer_derange and cand == row["c"]:
                        continue
                    if not legal(row, cand):
                        continue
                    pick = j
                    break
                if pick is not None:
                    break
            if pick is None:
                chosen[i] = row["c"]
            else:
                used.add(pick)
                chosen[i] = pool[pick]
        for i, row in enumerate(group):
            out[row["triplet_id"]] = chosen[i]
    return out
