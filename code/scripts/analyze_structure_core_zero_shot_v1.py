#!/usr/bin/env python3
"""Zero-shot structure-core evaluation from already scored pairs.

Published PLM-interact / D-SCRIPT / Topsy-Turvy scores cover the full
structure v1 pool. MINT is only available on the 300-triplet structure
pilot, so it is copied from that audit rather than rescoring.

This script does not train, does not touch test_v2, and does not treat
the in-sample finetuned full-pool ranking as a generalization result.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


CORE_STRATA = {"3_10", "11_24"}
RNG_SEED = 20260816
N_BOOT = 1000


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=root / "data/interim/structure_triplet_scoring_v1/scoring_pairs_v1.tsv",
    )
    parser.add_argument(
        "--links",
        type=Path,
        default=root / "data/interim/structure_triplet_scoring_v1/triplet_score_links_v1.tsv",
    )
    parser.add_argument(
        "--plminteract-pairs-csv",
        type=Path,
        default=root / "data/interim/structure_triplet_scoring_v1/plminteract_pairs_v1.csv",
    )
    parser.add_argument(
        "--plminteract-scores",
        type=Path,
        default=root / "data/interim/structure_triplet_scoring_v1/plminteract_scores_v1.csv",
    )
    parser.add_argument(
        "--dscript-scores",
        type=Path,
        default=root / "data/interim/structure_triplet_scoring_v1/dscript_human_v1_scores_v1.tsv",
    )
    parser.add_argument(
        "--topsy-scores",
        type=Path,
        default=root / "data/interim/structure_triplet_scoring_v1/topsy_turvy_human_v1_scores_v1.tsv",
    )
    parser.add_argument(
        "--sprint-scores",
        type=Path,
        default=root / "data/interim/structure_triplet_scoring_v1/sprint_scores_v1.tsv",
    )
    parser.add_argument(
        "--pilot-three-way",
        type=Path,
        default=root
        / "data/interim/counterfactual_1000_triplet_pilot_v1/random_concordant_audit_v1/stratified_three_way_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data/interim/structure_triplet_scoring_v1/core_zero_shot_eval_v1",
    )
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "frac_ge_0.5": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(median, 4),
        "frac_ge_0.5": round(sum(v >= 0.5 for v in values) / len(values), 4),
    }


def load_plminteract(pairs_csv: Path, scores_csv: Path) -> dict[str, float]:
    rows = list(csv.DictReader(pairs_csv.open(encoding="utf-8", newline="")))
    scores = [float(r["score"]) for r in csv.DictReader(scores_csv.open(encoding="utf-8", newline=""))]
    if len(scores) != len(rows):
        raise SystemExit(f"plminteract score count {len(scores)} != pair rows {len(rows)}")
    return {row["pair_id"]: score for row, score in zip(rows, scores)}


def load_protein_pair_scores(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            if len(parts) < 3 or parts[0] == "protein_x":
                continue
            scores[tuple(sorted((parts[0], parts[1])))] = float(parts[2])
    return scores


def map_pair_ids(pairs: list[dict[str, str]], by_proteins: dict[tuple[str, str], float]) -> dict[str, float]:
    return {
        row["pair_id"]: by_proteins[tuple(sorted((row["protein_x"], row["protein_y"])))]
        for row in pairs
        if tuple(sorted((row["protein_x"], row["protein_y"]))) in by_proteins
    }


def win(s_ad: float, s_ac: float) -> float:
    if s_ad > s_ac:
        return 1.0
    if s_ad < s_ac:
        return 0.0
    return 0.5


def bootstrap_mean(values: list[float], n_boot: int = N_BOOT, seed: int = RNG_SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    if not values:
        return (float("nan"), float("nan"))
    samples = []
    n = len(values)
    for _ in range(n_boot):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo = samples[int(0.025 * (n_boot - 1))]
    hi = samples[int(0.975 * (n_boot - 1))]
    return (round(lo, 4), round(hi, 4))


def evaluate_model(name: str, scores: dict[str, float], pairs: list[dict[str, str]], links: list[dict[str, str]]) -> dict:
    by_role: dict[str, list[float]] = defaultdict(list)
    ac_by_stratum: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        score = scores.get(row["pair_id"])
        if score is None:
            continue
        by_role[row["role"]].append(score)
        if row["role"] == "ac_negative":
            ac_by_stratum[row["assembly_size_stratum"]].append(score)

    layers = {
        "core": lambda stratum: stratum in CORE_STRATA,
        "pressure": lambda stratum: stratum == "gt24",
        "all": lambda _stratum: True,
    }
    ranking: dict[str, dict] = {}
    for layer, keep in layers.items():
        per_ac: dict[str, list[float]] = defaultdict(list)
        per_anchor: dict[str, list[float]] = defaultdict(list)
        row_wins: list[float] = []
        row_deltas: list[float] = []
        for link in links:
            if not keep(link["assembly_size_stratum"]):
                continue
            s_ac = scores.get(link["ac_pair_id"])
            s_ad = scores.get(link["ad_pair_id"])
            if s_ac is None or s_ad is None:
                continue
            value = win(s_ad, s_ac)
            row_wins.append(value)
            row_deltas.append(s_ad - s_ac)
            per_ac[link["ac_id"]].append(value)
            per_anchor[link["anchor"]].append(value)
        ac_means = [sum(v) / len(v) for v in per_ac.values()]
        anchor_means = [sum(v) / len(v) for v in per_anchor.values()]
        ac_ci = bootstrap_mean(ac_means)
        ranking[layer] = {
            "n_triplets": len(row_wins),
            "n_unique_ac": len(ac_means),
            "n_anchors": len(anchor_means),
            "row_p_d_gt_c": round(sum(row_wins) / len(row_wins), 4) if row_wins else None,
            "median_delta": stats(row_deltas)["median"] if row_deltas else None,
            "unique_ac_p_d_gt_c": round(sum(ac_means) / len(ac_means), 4) if ac_means else None,
            "unique_ac_p_d_gt_c_ci95": list(ac_ci) if ac_means else None,
            "anchor_block_p_d_gt_c": round(sum(anchor_means) / len(anchor_means), 4) if anchor_means else None,
        }

    return {
        "model": name,
        "pairs_scored": len(scores),
        "by_role": {key: stats(values) for key, values in sorted(by_role.items())},
        "ac_negative_by_stratum": {key: stats(values) for key, values in sorted(ac_by_stratum.items())},
        "ranking": ranking,
    }


def load_pilot_structure(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in rows if row.get("dataset_key") == "structure"]


def main() -> None:
    args = parse_args()
    pairs = list(csv.DictReader(args.pairs.open(encoding="utf-8", newline=""), delimiter="\t"))
    links = list(csv.DictReader(args.links.open(encoding="utf-8", newline=""), delimiter="\t"))

    models = {
        "plminteract_base": load_plminteract(args.plminteract_pairs_csv, args.plminteract_scores),
        "dscript": map_pair_ids(pairs, load_protein_pair_scores(args.dscript_scores)),
        "topsy_turvy": map_pair_ids(pairs, load_protein_pair_scores(args.topsy_scores)),
    }
    if args.sprint_scores.exists():
        models["sprint"] = map_pair_ids(pairs, load_protein_pair_scores(args.sprint_scores))

    report = {
        "status": "zero_shot_structure_eval_not_a_frozen_benchmark",
        "scope": {
            "full_pool_models": [name for name in models if name != "mint_bernett"],
            "mint_coverage": "300-triplet structure core pilot only",
            "sprint_coverage": "same structure scoring pairs when sprint_scores_v1.tsv exists",
            "core_strata": sorted(CORE_STRATA),
            "did_not_touch_test_v2": True,
        },
        "models": {name: evaluate_model(name, scores, pairs, links) for name, scores in models.items()},
        "pilot_structure_three_way": load_pilot_structure(args.pilot_three_way),
        "optimization_already_run": {
            "in_sample_core_ranking_is_not_generalization": True,
            "honest_split": "structure homology split v2, test_v2 sealed",
            "test_v2_triplet_ranking": {
                "base": 0.540,
                "finetuned_v2": 0.572,
                "delta": 0.033,
                "delta_ci95": [0.007, 0.060],
            },
            "source": "docs/audits/STRUCTURE_TRIPLET_HOMOLOGY_SPLIT_V2.md",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "structure_core_zero_shot_eval_v1.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
