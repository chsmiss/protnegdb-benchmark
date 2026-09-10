#!/usr/bin/env python3
"""Build Figure 1/2 master tables and lock the zero-shot ledger.

No training. Does not read test_v2. Literature assay negatives stay a
separate evidence class from structural non-contacts.

Figure 1 rows = unique structural A-C.
Figure 2 rows = partner-specific triplets with published PPI scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CORE = ("3_10", "11_24")
SEQID_BINS = (
    ("lt20", 0.0, 0.20),
    ("20_30", 0.20, 0.30),
    ("30_40", 0.30, 0.40),
    ("40_50", 0.40, 0.50),
    ("ge50", 0.50, 1.01),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ac-table", type=Path,
                   default=root / "data/interim/structure_counterfactual_v1/structure_ac_base_v1.tsv")
    p.add_argument("--triplets", type=Path,
                   default=root / "data/interim/structure_counterfactual_v1/counterfactual_structure_triplets_v1.tsv")
    p.add_argument("--triplet-scores", type=Path,
                   default=root / "data/interim/structure_triplet_scoring_v1/report_v1/triplet_model_scores_v1.tsv")
    p.add_argument("--scoring-pairs", type=Path,
                   default=root / "data/interim/structure_triplet_scoring_v1/scoring_pairs_v1.tsv")
    p.add_argument("--plminteract-scores", type=Path,
                   default=root / "data/interim/structure_triplet_scoring_v1/plminteract_scores_v1.csv")
    p.add_argument("--construction-summary", type=Path,
                   default=root / "data/interim/structure_counterfactual_v1/summary_v1.json")
    p.add_argument("--v2-summary", type=Path,
                   default=root / "data/interim/structure_triplet_finetune_v2/summary_v2.json")
    p.add_argument("--clusters", type=Path,
                   default=root / "data/interim/structure_family_transfer_v4/clusters_id50.tsv")
    p.add_argument("--pilot-three-way", type=Path,
                   default=root / "data/interim/counterfactual_1000_triplet_pilot_v1/random_concordant_audit_v1/stratified_three_way_v1.json")
    p.add_argument("--output-dir", type=Path,
                   default=root / "data/interim/figure12_master_v1")
    return p.parse_args()


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_clusters(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                out[fields[1]] = fields[0]
    return out


def fnum(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def seqid_bin(seqid: float | None) -> str:
    if seqid is None or math.isnan(seqid):
        return "missing"
    for name, lo, hi in SEQID_BINS:
        if lo <= seqid < hi:
            return name
    return "missing"


def rank_hit(s_ad: float | None, s_ac: float | None) -> float | None:
    if s_ad is None or s_ac is None:
        return None
    if s_ad > s_ac:
        return 1.0
    if s_ad < s_ac:
        return 0.0
    return 0.5


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "frac_ge_0.5": None}
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 6),
        "median": round(float(statistics.median(values)), 6),
        "frac_ge_0.5": round(sum(1 for v in values if v >= 0.5) / len(values), 6),
    }


def ranking_block(rows: list[dict[str, object]], ad_key: str, ac_key: str) -> dict[str, object]:
    hits = []
    deltas = []
    ties = 0
    by_ac: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        s_ad = row.get(ad_key)
        s_ac = row.get(ac_key)
        hit = rank_hit(s_ad if isinstance(s_ad, float) else None,
                       s_ac if isinstance(s_ac, float) else None)
        if hit is None:
            continue
        hits.append(hit)
        delta = float(s_ad) - float(s_ac)  # type: ignore[arg-type]
        deltas.append(delta)
        if hit == 0.5:
            ties += 1
        by_ac[str(row["ac_id"])].append(hit)
    ac_means = [sum(v) / len(v) for v in by_ac.values()]
    return {
        "n_triplets": len(hits),
        "n_unique_ac": len(ac_means),
        "row_p_d_gt_c": round(sum(hits) / len(hits), 6) if hits else None,
        "unique_ac_p_d_gt_c": round(sum(ac_means) / len(ac_means), 6) if ac_means else None,
        "median_delta": round(float(statistics.median(deltas)), 6) if deltas else None,
        "mean_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
        "tie_fraction": round(ties / len(hits), 6) if hits else None,
    }


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ac_rows = load_tsv(args.ac_table)
    triplets = load_tsv(args.triplets)
    scores = {r["triplet_id"]: r for r in load_tsv(args.triplet_scores)}
    clusters = load_clusters(args.clusters)
    construction = json.loads(args.construction_summary.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2_summary.read_text(encoding="utf-8")) if args.v2_summary.exists() else {}
    pairs = load_tsv(args.scoring_pairs)
    plm_scores = []
    with args.plminteract_scores.open(encoding="utf-8", newline="") as handle:
        plm_scores = [float(r["score"]) for r in csv.DictReader(handle)]
    if len(plm_scores) != len(pairs):
        raise SystemExit(f"score/pair mismatch {len(plm_scores)} vs {len(pairs)}")
    by_role: dict[str, list[float]] = defaultdict(list)
    core_role: dict[str, list[float]] = defaultdict(list)
    for pair, score in zip(pairs, plm_scores):
        by_role[pair["role"]].append(score)
        if pair.get("assembly_size_stratum") in CORE:
            core_role[pair["role"]].append(score)

    ac_ids_in_core_trips = {r["ac_id"] for r in triplets if r["assembly_size_stratum"] in CORE}
    fig1 = []
    conflict_core = Counter()
    for row in ac_rows:
        core = row["assembly_size_stratum"] in CORE
        fig1.append({
            "ac_id": row["ac_id"],
            "A_uniprot": row["protein_a"],
            "C_uniprot": row["protein_c"],
            "taxon": row["tax_id"],
            "pdb_id": row["pdb_id"],
            "complex_size": row["num_protein_chains"],
            "complex_size_bin": row["assembly_size_stratum"],
            "n_pdb": row["aggregate_distinct_pdb_count"],
            "n_noncontact_assemblies": row["aggregate_noncontact_assemblies"],
            "coordinate_qc": row["qc_status"],
            "coverage_A": row["a_canonical_coverage"],
            "coverage_C": row["c_canonical_coverage"],
            "min_distance": row["ac_min_heavy_a"],
            "direct_positive_conflict": row["direct_positive_conflict_status"],
            "structure_grade": row["structure_grade_provisional"],
            "core_flag": "1" if core else "0",
            "gt24_flag": "1" if row["assembly_size_stratum"] == "gt24" else "0",
            "replicated_flag": "1" if int(row["aggregate_distinct_pdb_count"] or 0) >= 2 else "0",
            "has_triplet": "1" if row["ac_id"] in ac_ids_in_core_trips else "0",
            "mmseqs_family_A": clusters.get(row["protein_a"], ""),
            "mmseqs_family_C": clusters.get(row["protein_c"], ""),
            "negative_label_semantics": row["negative_label_semantics"],
        })
        if core:
            conflict_core[row["direct_positive_conflict_status"]] += 1

    fig1_fields = list(fig1[0].keys())
    write_tsv(args.output_dir / "figure1_ac_master_v1.tsv", fig1, fig1_fields)

    fig2 = []
    ac_map = {r["ac_id"]: r for r in ac_rows}
    for trip in triplets:
        sc = scores.get(trip["triplet_id"], {})
        s_ad = fnum(sc.get("plminteract_ad"))
        s_ac = fnum(sc.get("plminteract_ac"))
        s_ds_ad = fnum(sc.get("dscript_ad"))
        s_ds_ac = fnum(sc.get("dscript_ac"))
        s_tt_ad = fnum(sc.get("topsy_turvy_ad"))
        s_tt_ac = fnum(sc.get("topsy_turvy_ac"))
        seqid = fnum(trip.get("sequence_identity"))
        ac = ac_map.get(trip["ac_id"], {})
        n_pdb = int(ac.get("aggregate_distinct_pdb_count") or 0)
        fig2.append({
            "triplet_id": trip["triplet_id"],
            "ac_id": trip["ac_id"],
            "A": trip["anchor"],
            "C": trip["negative_partner_c"],
            "D": trip["positive_partner_d"],
            "pdb_assembly": trip.get("ac_structure_file", ""),
            "complex_size_bin": trip["assembly_size_stratum"],
            "core_flag": "1" if trip["assembly_size_stratum"] in CORE else "0",
            "n_pdb_AC": str(n_pdb),
            "replicated_AC": "1" if n_pdb >= 2 else "0",
            "A_C_min_heavy": trip.get("ac_min_heavy_a", ""),
            "A_D_min_heavy": trip.get("positive_min_heavy_a", ""),
            "positive_source": trip.get("positive_source", ""),
            "seqid_CD": "" if seqid is None else f"{seqid:.6f}",
            "seqid_CD_bin": seqid_bin(seqid),
            "family_A": clusters.get(trip["anchor"], ""),
            "family_C": clusters.get(trip["negative_partner_c"], ""),
            "family_D": clusters.get(trip["positive_partner_d"], ""),
            "score_AD_PLMInteract": "" if s_ad is None else f"{s_ad:.6f}",
            "score_AC_PLMInteract": "" if s_ac is None else f"{s_ac:.6f}",
            "delta_PLMInteract": "" if s_ad is None or s_ac is None else f"{s_ad - s_ac:.6f}",
            "rank_correct_PLMInteract": "" if rank_hit(s_ad, s_ac) is None else str(rank_hit(s_ad, s_ac)),
            "score_AD_DSCRIPT": "" if s_ds_ad is None else f"{s_ds_ad:.6f}",
            "score_AC_DSCRIPT": "" if s_ds_ac is None else f"{s_ds_ac:.6f}",
            "score_AD_TopsyTurvy": "" if s_tt_ad is None else f"{s_tt_ad:.6f}",
            "score_AC_TopsyTurvy": "" if s_tt_ac is None else f"{s_tt_ac:.6f}",
            "_s_ad_plm": s_ad,
            "_s_ac_plm": s_ac,
            "_s_ad_ds": s_ds_ad,
            "_s_ac_ds": s_ds_ac,
            "_s_ad_tt": s_tt_ad,
            "_s_ac_tt": s_tt_ac,
        })
    public_fields = [k for k in fig2[0] if not k.startswith("_")]
    write_tsv(args.output_dir / "figure2_triplet_master_v1.tsv",
              [{k: r[k] for k in public_fields} for r in fig2], public_fields)

    core_ac = [r for r in fig1 if r["core_flag"] == "1"]
    core_trips = [r for r in fig2 if r["core_flag"] == "1"]
    replicated_ac = [r for r in core_ac if r["replicated_flag"] == "1"]
    replicated_trips = [r for r in core_trips if r["replicated_AC"] == "1"]
    conflict_values = dict(conflict_core)
    n_clear_no_pdb_contact = sum(
        1 for r in core_ac if r["direct_positive_conflict"].startswith("no_PDB_contact")
    )

    plm_core = ranking_block(core_trips, "_s_ad_plm", "_s_ac_plm")
    plm_rep = ranking_block(replicated_trips, "_s_ad_plm", "_s_ac_plm")
    ds_core = ranking_block(core_trips, "_s_ad_ds", "_s_ac_ds")
    tt_core = ranking_block(core_trips, "_s_ad_tt", "_s_ac_tt")

    seqid_rows = []
    for name, lo, hi in SEQID_BINS:
        subset = [r for r in core_trips if r["seqid_CD_bin"] == name]
        block = ranking_block(subset, "_s_ad_plm", "_s_ac_plm")
        seqid_rows.append({"bin": name, "lo": lo, "hi": hi, **block})

    pilot = []
    if args.pilot_three_way.exists():
        pilot = json.loads(args.pilot_three_way.read_text(encoding="utf-8"))
    structure_pilot = [
        r for r in pilot
        if r.get("dataset_key") == "structure" and r.get("model_key") == "plminteract_base"
    ]

    proteins_core = set()
    for r in core_trips:
        proteins_core.update((r["A"], r["C"], r["D"]))
    tax_core = {r["taxon"] for r in core_ac}
    pdb_core = {r["pdb_id"] for r in core_ac}
    fams = {r["mmseqs_family_A"] for r in core_ac if r["mmseqs_family_A"]}
    fams |= {r["mmseqs_family_C"] for r in core_ac if r["mmseqs_family_C"]}
    fams |= {r["family_D"] for r in core_trips if r["family_D"]}
    size_bin = Counter(r["complex_size_bin"] for r in core_ac)

    sel = construction["selection_counts"]
    qc = construction["ac_qc_counts"]
    ledger = {
        "status": "figure12_zero_shot_ledger_not_a_frozen_benchmark",
        "did_not_train": True,
        "did_not_touch_test_v2": True,
        "label_semantics": "assembly_context_direct_noncontact_not_universal_nonbinder",
        "structural_resource": {
            "class_a_total": sel["aggregate_class_a"],
            "same_taxon_mapped": sel["eligible_before_caps"],
            "excluded_id_or_sequence": sel["excluded_id_or_sequence"],
            "excluded_taxonomy": sel["excluded_taxonomy"],
            "cap_pdb": sel["cap_pdb"],
            "cap_anchor": sel["cap_anchor"],
            "coordinate_candidates": sel["coordinate_candidates_selected"],
            "qc_passed_before_stratum_cap": qc["Structure-A-provisional|3_10"]
                + qc["Structure-B-provisional|3_10"]
                + qc["Structure-A-provisional|11_24"]
                + qc["Structure-B-provisional|11_24"]
                + qc["Structure-C-provisional|gt24"],
            "assembly_exceeds_60_chains": qc["assembly_exceeds_chain_cap"],
            "gt24_qc_passed_then_capped": qc["Structure-C-provisional|gt24"],
            "gt24_trimmed": qc["ac_trimmed_beyond_target"],
            "stratified_mother_pool": qc["ac_accepted"],
            "core_ac": len(core_ac),
            "core_ac_3_10": size_bin["3_10"],
            "core_ac_11_24": size_bin["11_24"],
            "core_triplets": len(core_trips),
            "core_ac_with_triplet": sum(1 for r in core_ac if r["has_triplet"] == "1"),
            "gt24_share_of_same_taxon_class_a_from_construction_audit": 0.903,
        },
        "evidence_strength": {
            "core_ac": len(core_ac),
            "replicated_ge2_pdb": len(replicated_ac),
            "replicated_fraction": round(len(replicated_ac) / len(core_ac), 6),
            "direct_positive_conflict_status_counts": dict(conflict_core),
            "pdb_index_without_other_direct_contact": n_clear_no_pdb_contact,
            "pdb_index_direct_contact_conflicts": len(core_ac) - n_clear_no_pdb_contact,
            "known_direct_positive_conflicts": len(core_ac) - n_clear_no_pdb_contact,
            "conflict_note": "zero conflicts among core A-C is a construction filter, not independent validation",
        },
        "coverage": {
            "unique_anchor_A": len({r["A"] for r in core_trips}),
            "unique_negative_C": len({r["C"] for r in core_trips}),
            "unique_positive_D": len({r["D"] for r in core_trips}),
            "unique_proteins": len(proteins_core),
            "unique_taxon": len(tax_core),
            "unique_pdb": len(pdb_core),
            "mmseqs50_families": len(fams),
        },
        "previous_bce_split_v2_not_figure3": {
            "unique_labelled_pairs": v2.get("unique_labelled_pairs"),
            "train_rows": (v2.get("rows") or {}).get("train"),
        },
        "original_plminteract": {
            "all_pool_by_role": {k: stats(v) for k, v in by_role.items()},
            "core_ac_scores": stats(core_role.get("ac_negative", [])),
            "core_ad_scores": stats(core_role.get("ad_positive", [])),
            "random_unlabeled": stats(by_role.get("random_unlabeled", [])),
            "core_triplets": plm_core,
            "replicated_triplets": plm_rep,
            "seqid_CD_bins": seqid_rows,
            "structure_pilot_three_way": structure_pilot[0] if structure_pilot else None,
        },
        "other_models_core_triplets": {
            "dscript": ds_core,
            "topsy_turvy": tt_core,
        },
        "figure_notes": {
            "figure1_shrinkage": "task-aware filtering, not data loss",
            "gt24": "exploratory evaluation only; excluded from primary hard-negative training",
            "figure2_sim_CD": "hardness of the negative; not test-to-train transfer",
            "figure3_sim_test_train": "transfer of hard-negative supervision; separate question",
        },
    }
    (args.output_dir / "ledger_v1.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "core_ac": len(core_ac),
        "replicated": len(replicated_ac),
        "core_triplets": len(core_trips),
        "plm_core": plm_core,
        "plm_replicated": plm_rep,
        "random": stats(by_role.get("random_unlabeled", [])),
        "conflicts": dict(conflict_core),
    }, indent=2))


if __name__ == "__main__":
    main()
