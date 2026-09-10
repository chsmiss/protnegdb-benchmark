#!/usr/bin/env python3
"""Analyze Figure 6: independent reviewed literature negatives vs noncontact.

All 480 supplied records are accepted as manually verified negatives.  Source
rule tiers are ignored.  Primary semantics are derived only from the reported
assay method, while mapping, sequence availability, overlap, and truncation are
handled as technical QC dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import analyze_negatome_figure4_subset_v1 as neg4


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/figure6_protneg_assay_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=BASE / "protneg_assay_manifest_v1.tsv")
    p.add_argument("--scores", type=Path, default=BASE / "protneg_assay_negative_scores_v1.csv")
    p.add_argument("--paired-manifest", type=Path, default=BASE / "protneg_assay_paired_manifest_v1.tsv")
    p.add_argument("--paired-scores", type=Path, default=BASE / "protneg_assay_paired_scores_v1.csv")
    p.add_argument("--negatome-pairs", type=Path, default=ROOT / "data/interim/negatome_manual_plminteract_v1/negatome_manual_pairs_v1.tsv")
    p.add_argument("--negatome-sequences", type=Path, default=ROOT / "data/interim/negatome_manual_plminteract_v1/negatome_manual_pairs_v1.csv")
    p.add_argument("--negatome-scores", type=Path, default=ROOT / "data/interim/negatome_manual_plminteract_v1/negatome_manual_scores_v1.csv")
    p.add_argument("--swissprot", type=Path, default=Path("/data/chs/12.codebuddy/bridge/data/uniprot_sprot.fasta.gz"))
    p.add_argument("--string-train", type=Path, default=ROOT / "data/external/model_training_sets/plm_interact/string_v12/protein.pairs_human_V12_train.tsv")
    p.add_argument("--structure-pairs", type=Path, default=ROOT / "data/interim/structure_triplet_scoring_v1/scoring_pairs_v1.tsv")
    p.add_argument("--structure-scores", type=Path, default=ROOT / "data/interim/structure_triplet_scoring_v1/plminteract_scores_v1.csv")
    p.add_argument("--structure-clean", type=Path, default=ROOT / "data/interim/figure5_structure_score_drivers_v2/figure5_ac_features_v2.tsv")
    p.add_argument("--figure3", type=Path, default=ROOT / "data/interim/structure_family_transfer_v4/id50/figure3_eval_v4.json")
    p.add_argument("--output-dir", type=Path, default=BASE)
    return p.parse_args()


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def as_bool(row: dict[str, str], key: str) -> bool:
    return row.get(key, "") == "1"


def wilson(k: float, n: int, z: float = 1.96) -> list[float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(center - half, 4), round(center + half, 4)]


def stats(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean": None, "median": None, "frac_ge_0.5": None, "ci95": None}
    k = int(np.sum(arr >= 0.5))
    return {
        "n": int(len(arr)),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "frac_ge_0.5": round(k / len(arr), 4),
        "ci95": wilson(k, len(arr)),
        "n_ge_0.5": k,
    }


def paired_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"n": 0, "win_rate": None, "ci95": None, "median_margin": None}
    margins = np.asarray([float(row["positive_score"]) - float(row["negative_score"]) for row in rows])
    wins = float(np.sum(margins > 0) + 0.5 * np.sum(margins == 0))
    return {
        "n": len(rows),
        "wins_with_half_ties": round(wins, 1),
        "ties": int(np.sum(margins == 0)),
        "win_rate": round(wins / len(rows), 4),
        "ci95": wilson(wins, len(rows)),
        "mean_margin": round(float(np.mean(margins)), 4),
        "median_margin": round(float(np.median(margins)), 4),
    }


def load_score_map(path: Path) -> dict[str, float]:
    rows = read_rows(path, ",")
    if not rows or "directed_pair_id" not in rows[0]:
        raise SystemExit(f"keyed score file required: {path}")
    out = {row["directed_pair_id"]: float(row["score"]) for row in rows}
    if len(out) != len(rows):
        raise SystemExit(f"duplicate score keys in {path}")
    return out


def load_negatome_clean(args: argparse.Namespace) -> list[dict[str, object]]:
    pairs = read_rows(args.negatome_pairs, "\t")
    scores = [float(row["score"]) for row in read_rows(args.negatome_scores, ",")]
    accessions = {row["protein_a"] for row in pairs} | {row["protein_b"] for row in pairs}
    tax = neg4.load_swissprot_tax(args.swissprot, accessions)
    rows = neg4.attach_flags(pairs, scores, tax, 1024)
    sequences = read_rows(args.negatome_sequences, ",")
    if len(sequences) != len(rows):
        raise SystemExit("Negatome sequence/metadata mismatch")
    string_keys = neg4.load_string_positive_keys(args.string_train)
    clean = []
    for row, seq in zip(rows, sequences):
        overlap = neg4.seq_pair_key(seq["query"], seq["text"]) in string_keys
        if (
            row["stringent"] and row["canonical"] and row["human_human"]
            and row["pair_fits"] and row["method_class"] == "direct_binary"
            and not overlap
        ):
            clean.append(row)
    return clean


def load_random(args: argparse.Namespace) -> list[float]:
    pairs = read_rows(args.structure_pairs, "\t")
    scores = [float(row["score"]) for row in read_rows(args.structure_scores, ",")]
    if len(pairs) != len(scores):
        raise SystemExit("structure pair/score mismatch")
    return [score for row, score in zip(pairs, scores) if row["role"] == "random_unlabeled"]


def load_structural_clean(args: argparse.Namespace) -> list[float]:
    rows = read_rows(args.structure_clean, "\t")
    return [
        float(row["score"])
        for row in rows
        if row["core_flag"] == "1"
        and row["truncated"] == "0"
        and row["train_pair"] == "0"
        and row["association_class"] != "direct_conflict"
    ]


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_rows(args.manifest, "\t")
    score_map = load_score_map(args.scores)
    scored: list[dict[str, object]] = []
    for row in manifest:
        score_id = row.get("negative_score_id", "")
        if not score_id:
            continue
        if score_id not in score_map:
            raise SystemExit(f"missing negative score {score_id}")
        scored.append({**row, "score": score_map[score_id]})

    # Remove exact Negatome pair overlaps from all modern-source comparisons.
    independent = [row for row in scored if not as_bool(row, "exact_negatome_pair_overlap")]
    modern_direct = [row for row in independent if as_bool(row, "direct_assay_negative")]
    modern_direct_no_trunc = [row for row in modern_direct if as_bool(row, "pair_fits_1024")]
    modern_direct_human = [row for row in modern_direct if as_bool(row, "human_human")]
    modern_direct_human_no_trunc = [
        row for row in modern_direct_human if as_bool(row, "pair_fits_1024")
    ]

    by_semantics: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in independent:
        by_semantics[str(row["evidence_semantics"])].append(row)

    negatome = load_negatome_clean(args)
    random = load_random(args)
    structural = load_structural_clean(args)

    paired_manifest = read_rows(args.paired_manifest, "\t")
    paired_score_map = load_score_map(args.paired_scores)
    paired: list[dict[str, object]] = []
    for row in paired_manifest:
        neg_id, pos_id = row["negative_score_id"], row["positive_score_id"]
        if neg_id not in paired_score_map or pos_id not in paired_score_map:
            raise SystemExit(f"missing paired score for {row['pair_root']}")
        paired.append({
            **row,
            "negative_score": paired_score_map[neg_id],
            "positive_score": paired_score_map[pos_id],
            "margin": paired_score_map[pos_id] - paired_score_map[neg_id],
            "positive_wins": int(paired_score_map[pos_id] > paired_score_map[neg_id]),
        })
    paired_no_trunc = [row for row in paired if as_bool(row, "both_pairs_fit_1024")]

    paired_by_semantics = {
        key: paired_stats([row for row in paired if row["evidence_semantics"] == key])
        for key in ("biophysical_direct", "other_direct_binding", "association_negative")
    }

    fig3 = json.loads(args.figure3.read_text(encoding="utf-8"))
    structure_rank = fig3["arms"]["original"]["family_unseen"]

    auc = float(mannwhitneyu(structural, [float(row["score"]) for row in modern_direct],
                             alternative="two-sided").statistic / (len(structural) * len(modern_direct)))

    category_raw = defaultdict(int)
    for row in manifest:
        category_raw[row["evidence_semantics"]] += 1
    reliable = [row for row in manifest if as_bool(row, "reliable_reported_taxon_mapping")]
    sequence_ready = [row for row in reliable if as_bool(row, "sequence_available")]
    independent_ready = [row for row in sequence_ready if not as_bool(row, "exact_negatome_pair_overlap")]

    result = {
        "status": "figure6_protneg_assay_independent_v1",
        "model": "published original PLM-Interact-650M-humanV12; no training",
        "source_policy": "all 480 supplied rows accepted; rule-tier fields ignored",
        "method_semantics": {
            "biophysical_direct": ["SPR", "ITC", "BLI", "MST", "fluorescence_anisotropy"],
            "other_direct_binding": ["pull_down", "in_vitro_binding_unspecified", "far_western"],
            "association_negative": ["coIP", "Y2H", "PLA", "BiFC", "FRET"],
        },
        "6a_funnel": [
            {"step": "manually_verified_unique_pairs", "n": len(manifest)},
            {"step": "reported_taxon_mapping", "n": len(reliable)},
            {"step": "canonical_sequence_available", "n": len(sequence_ready)},
            {"step": "exclude_exact_Negatome_pair_overlap", "n": len(independent_ready)},
            {"step": "direct_assay_primary", "n": len(modern_direct)},
            {"step": "direct_assay_no_truncation_sensitivity", "n": len(modern_direct_no_trunc)},
        ],
        "6a_raw_method_counts": dict(category_raw),
        "6b_negative_sets": {
            "random_unlabeled": stats(random),
            "negatome_direct_clean": stats([float(row["score"]) for row in negatome]),
            "protneg_direct_primary": stats([float(row["score"]) for row in modern_direct]),
            "structural_clean_core": stats(structural),
        },
        "6b_effect": {
            "P_structural_score_gt_protneg_direct": round(auc, 4),
            "cliffs_delta_structural_vs_protneg_direct": round(2 * auc - 1, 4),
        },
        "6c_paired_ranking": {
            "all_selected": paired_stats(paired),
            "both_pairs_no_truncation": paired_stats(paired_no_trunc),
            "by_negative_assay_semantics": paired_by_semantics,
            "structure_family_unseen_original": {
                "n": structure_rank["n"],
                "win_rate": structure_rank["p"],
                "ci95": [structure_rank["p_lo"], structure_rank["p_hi"]],
            },
            "positive_evidence_guard": (
                "same-taxon IntAct MI:0407 direct-positive candidates; database-direct, "
                "not individually re-curated positive experiments"
            ),
        },
        "6d_method_semantics": {
            key: stats([float(row["score"]) for row in by_semantics.get(key, [])])
            for key in ("biophysical_direct", "other_direct_binding", "association_negative")
        } | {"structural_clean_core": stats(structural)},
        "6e_source_concordance": {
            "negatome_direct_clean": stats([float(row["score"]) for row in negatome]),
            "protneg_direct_primary": stats([float(row["score"]) for row in modern_direct]),
            "protneg_direct_no_truncation": stats([float(row["score"]) for row in modern_direct_no_trunc]),
            "protneg_direct_human": stats([float(row["score"]) for row in modern_direct_human]),
            "protneg_direct_human_no_truncation": stats(
                [float(row["score"]) for row in modern_direct_human_no_trunc]
            ),
        },
        "qc": {
            "unique_pmids": len({row["pmid"] for row in manifest if row.get("pmid")}),
            "scored_modern": len(scored),
            "exact_negatome_overlaps_all_source": sum(
                as_bool(row, "exact_negatome_pair_overlap") for row in manifest
            ),
            "exact_negatome_overlaps_scored": sum(
                as_bool(row, "exact_negatome_pair_overlap") for row in scored
            ),
            "exact_string_positive_overlaps_scored": sum(
                as_bool(row, "exact_string_positive_overlap") for row in scored
            ),
        },
    }

    plot_rows: list[dict[str, object]] = []
    for name, values in (
        ("Random negative", random),
        ("Negatome direct", [float(row["score"]) for row in negatome]),
        ("ProtNeg direct", [float(row["score"]) for row in modern_direct]),
        ("Structural noncontact", structural),
    ):
        plot_rows.extend({"panel": "6b", "dataset": name, "score": value} for value in values)
    for key, label in (
        ("biophysical_direct", "Biophysical direct"),
        ("other_direct_binding", "Other direct-binding"),
        ("association_negative", "Association negative"),
    ):
        plot_rows.extend(
            {"panel": "6d", "dataset": label, "score": float(row["score"])}
            for row in by_semantics.get(key, [])
        )
    plot_rows.extend(
        {"panel": "6d", "dataset": "Structural noncontact", "score": value}
        for value in structural
    )

    write_tsv(
        args.output_dir / "figure6_scored_manifest_v1.tsv",
        scored,
        list(scored[0]),
    )
    write_tsv(
        args.output_dir / "figure6_paired_scores_v1.tsv",
        paired,
        list(paired[0]),
    )
    write_tsv(
        args.output_dir / "figure6_plot_values_v1.tsv",
        plot_rows,
        ["panel", "dataset", "score"],
    )
    (args.output_dir / "figure6_analysis_v1.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "modern_direct_n": len(modern_direct),
        "modern_direct_fpr": result["6b_negative_sets"]["protneg_direct_primary"]["frac_ge_0.5"],
        "modern_direct_median": result["6b_negative_sets"]["protneg_direct_primary"]["median"],
        "paired_n": len(paired),
        "paired_win_rate": result["6c_paired_ranking"]["all_selected"]["win_rate"],
        "output": str(args.output_dir / "figure6_analysis_v1.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
