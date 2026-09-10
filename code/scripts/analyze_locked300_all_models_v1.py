#!/usr/bin/env python3
"""Locked main300 all-model partner ranking. Reuses existing scores only.

Models: PLM-Interact, MINT, D-SCRIPT, Topsy-Turvy, SPRINT, ESMFold, AF3.
P(D>C) with ties=0.5. Primary 95% CI is anchor-block bootstrap (unique A).
Paired Δ is the bootstrap of P_i - P_j on the same resampled anchors.
No inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esmfold_linker_pilot_lib_v1 import ranking_hit, read_tsv, write_tsv

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/af3_partner_ranking_v1"
N_BOOT = 10000
BOOT_SEED = 20260821
MODELS = (
    ("PLM-Interact", "plm_ad", "plm_ac"),
    ("MINT", "mint_ad", "mint_ac"),
    ("D-SCRIPT", "dscript_ad", "dscript_ac"),
    ("Topsy-Turvy", "topsy_ad", "topsy_ac"),
    ("SPRINT", "sprint_ad", "sprint_ac"),
    ("ESMFold", "esm_ad", "esm_ac"),
    ("AF3", "af3_ad", "af3_ac"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=BASE / "main300_manifest_v1.tsv")
    p.add_argument("--af3-scores", type=Path, default=BASE / "main300_eval_v1/triplet_scores_v1.tsv")
    p.add_argument("--esmfold-scores", type=Path,
                   default=BASE / "esmfold_linker_main300_v1/triplet_comparison_v1.tsv")
    p.add_argument("--mint-scores", type=Path,
                   default=ROOT / "data/interim/structure_core_mint_full_v1/mint_triplet_scores_v1.tsv")
    p.add_argument("--sprint-scores", type=Path,
                   default=ROOT / "data/interim/structure_triplet_scoring_v1/sprint_scores_v1.tsv")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=BOOT_SEED)
    p.add_argument("--output-dir", type=Path, default=BASE / "locked300_all_models_v1")
    return p.parse_args()


def as_float(value) -> float:
    if value is None or str(value).strip() in {"", "NA", "nan", "None"}:
        return float("nan")
    return float(value)


def load_pair_scores(path: Path) -> dict[frozenset[str], float]:
    scores: dict[frozenset[str], float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            scores[frozenset((parts[0], parts[1]))] = float(parts[2])
    return scores


def finite(value: float) -> bool:
    return value is not None and not math.isnan(value)


def p_d_gt_c(hits: list[float]) -> float:
    return float(sum(hits) / len(hits)) if hits else float("nan")


def anchor_block_bootstrap(
    rows: list[dict],
    value_fn,
    n_boot: int,
    seed: int,
) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row["anchor"]].append(row)
    anchors = sorted(groups)
    if not anchors:
        return {"n_blocks": 0, "point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    point = value_fn(rows)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(anchors, size=len(anchors), replace=True)
        sample = [row for anchor in chosen for row in groups[anchor]]
        boots.append(value_fn(sample))
    low, high = np.percentile(boots, [2.5, 97.5])
    return {
        "n_blocks": len(anchors),
        "point": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def model_stats(rows: list[dict], name: str, ad_key: str, ac_key: str, n_boot: int, seed: int) -> dict:
    usable = [row for row in rows if finite(row[ad_key]) and finite(row[ac_key])]
    for row in usable:
        row[f"hit_{name}"] = ranking_hit(row[ad_key], row[ac_key])
    hits = [row[f"hit_{name}"] for row in usable]
    deltas = [row[ad_key] - row[ac_key] for row in usable]
    p_boot = anchor_block_bootstrap(usable, lambda sample: p_d_gt_c([r[f"hit_{name}"] for r in sample]), n_boot, seed)
    d_boot = anchor_block_bootstrap(usable, lambda sample: float(np.mean([r[ad_key] - r[ac_key] for r in sample])), n_boot, seed)
    return {
        "n": len(usable),
        "n_missing": len(rows) - len(usable),
        "P_D_gt_C": p_boot["point"],
        "anchor_block_ci95": [p_boot["ci_low"], p_boot["ci_high"]],
        "n_blocks": p_boot["n_blocks"],
        "n_strict": sum(1 for hit in hits if hit == 1.0),
        "n_tie": sum(1 for hit in hits if hit == 0.5),
        "mean_score_delta": float(np.mean(deltas)) if deltas else float("nan"),
        "mean_score_delta_ci95": [d_boot["ci_low"], d_boot["ci_high"]],
        "mean_AD": float(np.mean([row[ad_key] for row in usable])) if usable else float("nan"),
        "mean_AC": float(np.mean([row[ac_key] for row in usable])) if usable else float("nan"),
    }


def paired_delta(rows: list[dict], left: str, right: str, n_boot: int, seed: int) -> dict:
    usable = [row for row in rows if f"hit_{left}" in row and f"hit_{right}" in row
              and finite(row[f"hit_{left}"]) and finite(row[f"hit_{right}"])]
    boot = anchor_block_bootstrap(
        usable,
        lambda sample: p_d_gt_c([r[f"hit_{left}"] for r in sample]) - p_d_gt_c([r[f"hit_{right}"] for r in sample]),
        n_boot,
        seed,
    )
    return {
        "n": len(usable),
        "delta_P": boot["point"],
        "anchor_block_ci95": [boot["ci_low"], boot["ci_high"]],
        "n_blocks": boot["n_blocks"],
        "excludes_zero": boot["ci_low"] > 0 or boot["ci_high"] < 0,
    }


def join_rows(args: argparse.Namespace) -> list[dict]:
    manifest = read_tsv(args.manifest)
    af3 = {row["triplet_id"]: row for row in read_tsv(args.af3_scores)}
    esm = {row["triplet_id"]: row for row in read_tsv(args.esmfold_scores)}
    mint = {row["triplet_id"]: row for row in read_tsv(args.mint_scores)}
    sprint = load_pair_scores(args.sprint_scores)
    rows = []
    for meta in manifest:
        tid = meta["triplet_id"]
        a, c, d = meta["A"], meta["C"], meta["D"]
        af = af3[tid]
        es = esm[tid]
        mi = mint[tid]
        row = {
            "triplet_id": tid,
            "ac_id": meta["ac_id"],
            "anchor": a,
            "C": c,
            "D": d,
            "plm_ad": as_float(af["seq_PLMInteract_ad"]),
            "plm_ac": as_float(af["seq_PLMInteract_ac"]),
            "dscript_ad": as_float(af["seq_DSCRIPT_ad"]),
            "dscript_ac": as_float(af["seq_DSCRIPT_ac"]),
            "topsy_ad": as_float(af["seq_TopsyTurvy_ad"]),
            "topsy_ac": as_float(af["seq_TopsyTurvy_ac"]),
            "mint_ad": as_float(mi["score_ad"]),
            "mint_ac": as_float(mi["score_ac"]),
            "sprint_ad": sprint.get(frozenset((a, d)), float("nan")),
            "sprint_ac": sprint.get(frozenset((a, c)), float("nan")),
            "esm_ad": as_float(es["esm_score_ad"]),
            "esm_ac": as_float(es["esm_score_ac"]),
            "af3_ad": as_float(af["af3_iptm_ad"]),
            "af3_ac": as_float(af["af3_iptm_ac"]),
        }
        rows.append(row)
    return rows


def complete_case(rows: list[dict]) -> list[dict]:
    keys = [ad for _, ad, ac in MODELS] + [ac for _, ad, ac in MODELS]
    return [row for row in rows if all(finite(row[key]) for key in keys)]


def evaluate(rows: list[dict], n_boot: int, seed: int) -> dict:
    models = {}
    for name, ad_key, ac_key in MODELS:
        models[name] = model_stats(rows, name, ad_key, ac_key, n_boot, seed)
    deltas = {}
    for name, _, _ in MODELS:
        if name == "AF3":
            continue
        deltas[f"{name}_minus_AF3"] = paired_delta(rows, name, "AF3", n_boot, seed)
        deltas[f"AF3_minus_{name}"] = paired_delta(rows, "AF3", name, n_boot, seed)
    return {
        "n_triplets": len(rows),
        "n_anchors": len({row["anchor"] for row in rows}),
        "models": models,
        "paired_delta_P": deltas,
    }


def main() -> None:
    args = parse_args()
    rows = join_rows(args)
    if len(rows) != 300:
        raise SystemExit(f"expected 300 locked triplets, got {len(rows)}")
    if len({row["ac_id"] for row in rows}) != 300:
        raise SystemExit("locked set is not unique A-C")
    common = complete_case(rows)
    available = evaluate(rows, args.n_boot, args.seed)
    common_eval = evaluate(common, args.n_boot, args.seed)
    report = {
        "protocol": {
            "set": "AF3 main300 unique A-C; not the older 1000-pilot structure 300",
            "tie": 0.5,
            "ci": "anchor-block bootstrap over unique anchor A",
            "n_boot": args.n_boot,
            "seed": args.seed,
            "af3_score": "ipTM",
            "esmfold_score": "-mean inter-chain PAE after dropping G25 linker",
            "rescoring": "none; reused existing pair/triplet scores",
        },
        "coverage": {name: {"n": available["models"][name]["n"],
                            "n_missing": available["models"][name]["n_missing"]}
                     for name, _, _ in MODELS},
        "available": available,
        "all_model_complete": common_eval,
    }
    out_rows = []
    for row in rows:
        item = {key: row[key] for key in ("triplet_id", "ac_id", "anchor", "C", "D")}
        for name, ad_key, ac_key in MODELS:
            item[f"{name}_ad"] = row[ad_key]
            item[f"{name}_ac"] = row[ac_key]
            item[f"hit_{name}"] = row.get(f"hit_{name}", ranking_hit(row[ad_key], row[ac_key]))
        out_rows.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "triplet_scores_v1.tsv", out_rows)
    (args.output_dir / "summary_v1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_triplets": available["n_triplets"],
        "n_anchors": available["n_anchors"],
        "n_all_model_complete": common_eval["n_triplets"],
        "coverage": report["coverage"],
        "available_P": {name: available["models"][name] for name, _, _ in MODELS},
        "paired_AF3_minus_other": {key: val for key, val in available["paired_delta_P"].items()
                                   if key.startswith("AF3_minus_")},
    }, indent=2))


if __name__ == "__main__":
    main()
