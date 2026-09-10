#!/usr/bin/env python3
"""Test experimental-direct versus structural-noncontact semantics in 3 models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/revision_v2_matching_fig5"
FIG5 = ROOT / "data/interim/figure5_structure_score_drivers_v2"
STRUCT = ROOT / "data/interim/structure_triplet_scoring_v1"


def read(path: Path, delimiter: str = "\t", fields=None) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h, delimiter=delimiter, fieldnames=fields))


def score_map(path: Path) -> dict[tuple[str, str], float]:
    rows = read(path, fields=["a", "b", "score"])
    return {(r["a"], r["b"]): float(r["score"]) for r in rows}


def p_greater(left: list[float], right: list[float]) -> float:
    """P(left > right), with half credit for ties."""
    r = np.sort(np.asarray(right))
    x = np.asarray(left)
    less = np.searchsorted(r, x, side="left")
    leq = np.searchsorted(r, x, side="right")
    return float(np.sum(less + .5*(leq-less)) / (len(x)*len(r)))


def compare(structural: list[float], experimental: list[float], n_boot: int,
            seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    observed = p_greater(structural, experimental)
    boots = []
    for _ in range(n_boot):
        a = [rng.choice(structural) for _ in structural]
        b = [rng.choice(experimental) for _ in experimental]
        boots.append(p_greater(a, b))
    return {"structural_n": len(structural), "experimental_n": len(experimental),
            "structural_median": statistics.median(structural),
            "experimental_median": statistics.median(experimental),
            "p_structural_gt_experimental": observed,
            "cliffs_delta": 2*observed-1,
            "bootstrap_ci95": [float(np.quantile(boots,.025)), float(np.quantile(boots,.975))]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=BASE)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260817)
    args = p.parse_args()

    exp = read(BASE / "experimental_pair_manifest_v2.tsv")
    d_exp = score_map(BASE / "experimental_dscript_scores_v2.tsv")
    t_exp = score_map(BASE / "experimental_topsy_turvy_scores_v2.tsv")
    d_struct = score_map(STRUCT / "dscript_human_v1_scores_v1.tsv")
    t_struct = score_map(STRUCT / "topsy_turvy_human_v1_scores_v1.tsv")

    exp_rows = []
    for r in exp:
        if not (r["eligible_common_qc"] == "1" and
                r["assay_ontology"] == "reconstituted_direct" and
                r["exact_string_positive_overlap"] == "0"):
            continue
        key = (r["qid_a"], r["qid_b"])
        exp_rows.append({"collection": r["source"], "record_id": r["record_id"],
                         "human": int(r["taxon_a"] == "9606" and r["taxon_b"] == "9606"),
                         "PLM-Interact": float(r["score"]),
                         "D-SCRIPT": d_exp.get(key), "Topsy-Turvy": t_exp.get(key)})

    struct_rows = []
    for r in read(FIG5 / "figure5_ac_features_v2.tsv"):
        if not (r["core_flag"] == "1" and r["truncated"] == "0" and
                r["train_pair"] == "0" and r["association_class"] != "direct_conflict"):
            continue
        key = (r["protein_a"], r["protein_c"])
        struct_rows.append({"collection": "Structural", "record_id": r["ac_id"],
                            "human": int(r["human"]), "PLM-Interact": float(r["score"]),
                            "D-SCRIPT": d_struct.get(key), "Topsy-Turvy": t_struct.get(key)})

    report: dict[str, object] = {
        "protocol": {"effect": "within-model P(score_structural > score_experimental)",
                     "threshold": "none; scores are not calibrated across models",
                     "structural_scope": "clean core, <=1024 PLM tokens, no exact STRING-positive or IntAct direct conflict",
                     "experimental_scope": "harmonized reconstituted-direct, sequence-resolved, no exact STRING-positive"},
        "all_taxa": {}, "human_only": {},
    }
    source_rows = []
    for scope, human_only in (("all_taxa", False), ("human_only", True)):
        sr = [r for r in struct_rows if not human_only or r["human"]]
        for model_i, model in enumerate(("PLM-Interact", "D-SCRIPT", "Topsy-Turvy")):
            svals = [float(r[model]) for r in sr if r[model] is not None]
            report[scope][model] = {}
            for source_i, source in enumerate(("Negatome", "ProtNeg")):
                er = [r for r in exp_rows if r["collection"] == source and
                      (not human_only or r["human"]) and r[model] is not None]
                evals = [float(r[model]) for r in er]
                report[scope][model][source] = compare(
                    svals, evals, args.bootstrap,
                    args.seed + 1000*int(human_only) + 100*model_i + source_i)
                for r in er:
                    source_rows.append({"scope": scope, "model": model,
                                        "collection": source, "record_id": r["record_id"],
                                        "score": r[model]})
            for r in sr:
                if r[model] is not None:
                    source_rows.append({"scope": scope, "model": model,
                                        "collection": "Structural", "record_id": r["record_id"],
                                        "score": r[model]})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cross_model_semantic_gap_v2.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (args.output_dir / "cross_model_semantic_scores_v2.tsv").open(
            "w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(source_rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(source_rows)
    print(json.dumps({scope: {model: {source: round(v["p_structural_gt_experimental"],3)
                                            for source,v in sources.items()}
                                    for model,sources in report[scope].items()}
                      for scope in ("all_taxa","human_only")}, indent=2))


if __name__ == "__main__": main()
