#!/usr/bin/env python3
"""Compare MINT / SPRINT structural-clean vs experimental-direct scores.

Primary endpoint matches Figure 6:
  P = Mann-Whitney U(struct, exp) / (n_struct * n_exp)
    = P(score_structural > score_experimental) + 0.5 P(tie)

Cohorts are the frozen Figure 4/6 sets in cohort_pairs_v1.tsv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import parse_sprint_eval_scores_v1 as sprint_parse


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/mint_sprint_neg_semantics_v1"
SEEDS = [13, 29, 47, 71, 97]
EXPERIMENTAL = ("negatome_direct", "protneg_direct")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE)
    return p.parse_args()


def read_rows(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def wilson(k: float, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(center - half, 4), round(center + half, 4)]


def score_stats(values: list[float], threshold: float | None) -> dict[str, object]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "frac_ge_threshold": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    out: dict[str, object] = {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(median, 4),
    }
    if threshold is not None:
        k = sum(v >= threshold for v in values)
        out["frac_ge_0.5"] = round(k / len(values), 4)
        out["frac_ge_0.5_ci95"] = wilson(k, len(values))
        out["n_ge_0.5"] = k
    return out


def p_structural_gt_experimental(structural: list[float], experimental: list[float]) -> dict[str, object]:
    if not structural or not experimental:
        return {"n_structural": len(structural), "n_experimental": len(experimental),
                "P_structural_score_gt_experimental": None}
    xs = np.asarray(structural, dtype=float)
    ys = np.asarray(experimental, dtype=float)
    gt = int(np.sum(xs[:, None] > ys[None, :]))
    eq = int(np.sum(xs[:, None] == ys[None, :]))
    lt = int(np.sum(xs[:, None] < ys[None, :]))
    n = int(xs.size * ys.size)
    p_half = (gt + 0.5 * eq) / n
    mw = mannwhitneyu(xs, ys, alternative="two-sided").statistic / n
    return {
        "n_structural": int(xs.size),
        "n_experimental": int(ys.size),
        "n_pairs": n,
        "n_structural_gt": gt,
        "n_tie": eq,
        "n_structural_lt": lt,
        "P_structural_score_gt_experimental": round(float(p_half), 4),
        "P_structural_score_gt_experimental_strict": round(gt / n, 4),
        "mannwhitney_u_over_n": round(float(mw), 4),
        "cliffs_delta": round(2 * p_half - 1, 4),
    }


def merge_mint_scores(base: Path, cohort: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in cohort:
        raw = row.get("mint_existing_score", "")
        if raw != "":
            out[row["pair_id"]] = float(raw)
    seed_maps: dict[int, dict[str, float]] = {}
    directed_path = base / "mint_directed_new_v1.tsv"
    if not directed_path.is_file():
        return out
    directed = read_rows(directed_path)
    if not directed:
        return out
    for seed in SEEDS:
        path = base / f"mint_scores_seed{seed}_v1.tsv"
        if not path.is_file():
            raise SystemExit(f"missing MINT seed scores: {path}")
        rows = read_rows(path)
        if len(rows) != len(directed):
            raise SystemExit(f"MINT seed {seed} has {len(rows)} rows, expected {len(directed)}")
        seed_maps[seed] = {row["directed_pair_id"]: float(row["score_mean"]) for row in rows}
    by_pair: dict[str, list[float]] = defaultdict(list)
    for row in directed:
        pair_id = row["pair_id"]
        values = [seed_maps[seed][row["directed_pair_id"]] for seed in SEEDS]
        by_pair[pair_id].append(statistics.mean(values))
    for pair_id, values in by_pair.items():
        if len(values) != 2:
            raise SystemExit(f"MINT pair {pair_id} is not reciprocal")
        out[pair_id] = statistics.mean(values)
    return out


def load_sprint_scores(base: Path) -> dict[str, float]:
    sprint_dir = base / "sprint"
    result = sprint_dir / "sprint_result_v1.txt"
    if not result.is_file():
        return {}
    names = sprint_parse.read_fasta_names(sprint_dir / "sprint_proteins_v1.fasta")
    pairs = sprint_parse.read_space_pairs(sprint_dir / "test_positive_v1.txt")
    scores = sprint_parse.parse_sprint_score_lines(result.read_text(encoding="utf-8"))
    aligned = sprint_parse.align_scores(pairs, names, scores)
    index = read_rows(sprint_dir / "test_pair_index_v1.tsv")
    out: dict[str, float] = {}
    for row in index:
        key = (row["protein_a"], row["protein_b"])
        undirected = key if key[0] <= key[1] else (key[1], key[0])
        if undirected in aligned:
            out[row["pair_id"]] = aligned[undirected]
    return out


def model_block(
    name: str,
    scores: dict[str, float],
    cohort: list[dict[str, str]],
    threshold: float | None,
) -> dict[str, object]:
    by_cohort: dict[str, list[float]] = defaultdict(list)
    missing: dict[str, int] = defaultdict(int)
    for row in cohort:
        pair_id = row["pair_id"]
        if pair_id in scores:
            by_cohort[row["cohort"]].append(scores[pair_id])
        else:
            missing[row["cohort"]] += 1
    comparisons = {}
    for exp_name in EXPERIMENTAL:
        comparisons[f"structural_clean_vs_{exp_name}"] = p_structural_gt_experimental(
            by_cohort.get("structural_clean", []),
            by_cohort.get(exp_name, []),
        )
    return {
        "model": name,
        "score_sets": {key: score_stats(vals, threshold) for key, vals in sorted(by_cohort.items())},
        "comparisons": comparisons,
        "missing_scores": dict(missing),
    }


def main() -> None:
    args = parse_args()
    base = args.base
    cohort = read_rows(base / "cohort_pairs_v1.tsv")
    plm = {row["pair_id"]: float(row["plm_score"]) for row in cohort}
    mint = merge_mint_scores(base, cohort)
    sprint = load_sprint_scores(base)
    if len(mint) < 2480:
        raise SystemExit(f"MINT scores incomplete: {len(mint)}")
    if not sprint:
        raise SystemExit("SPRINT scores missing")

    write_tsv(
        base / "mint_pair_scores_v1.tsv",
        [{"pair_id": pair_id, "score": f"{score:.8f}"} for pair_id, score in sorted(mint.items())],
        ["pair_id", "score"],
    )
    write_tsv(
        base / "sprint_pair_scores_v1.tsv",
        [{"pair_id": pair_id, "score": f"{score:.10g}"} for pair_id, score in sorted(sprint.items())],
        ["pair_id", "score"],
    )

    result = {
        "status": "mint_sprint_neg_semantics_v1",
        "definition": (
            "P(score_structural > score_experimental) uses Mann-Whitney U / (n_s*n_e), "
            "i.e. P(>) + 0.5 P(tie), matching Figure 6."
        ),
        "cohorts": {
            "negatome_direct": "Figure 4 clean experimental direct; n=443",
            "protneg_direct": "Figure 6 independent ProtNeg direct; n=104",
            "structural_clean": "Figure 6 structural clean-core noncontact; n=2480",
        },
        "models": {
            "PLM-Interact": model_block("PLM-Interact", plm, cohort, 0.5),
            "MINT": model_block("MINT", mint, cohort, 0.5),
            "SPRINT": model_block("SPRINT", sprint, cohort, None),
        },
        "no_training": True,
    }
    (base / "neg_semantics_eval_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    headline = {}
    for model_name, block in result["models"].items():
        headline[model_name] = {
            key: val["P_structural_score_gt_experimental"]
            for key, val in block["comparisons"].items()
        }
    print(json.dumps({"headline": headline, "output": str(base / "neg_semantics_eval_v1.json")}, indent=2))


if __name__ == "__main__":
    main()
