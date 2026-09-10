#!/usr/bin/env python3
"""Scale up the structure-context counterfactual triplet pool (v1).

v1 keeps the v0 coordinate QC, open-wedge definition and triplet semantics
unchanged — the judgement functions are imported from
``build_structure_counterfactual_v0`` — and changes sampling scale, assembly
size handling and execution strategy:

- the 0.75 MB compressed-file pilot cap is removed (50 MB hard cap), so the
  full same-taxon Class-A pool (70,830 pairs / 4,899 distinct assemblies)
  becomes eligible instead of a small-assembly biased subset;
- instead of rejecting assemblies above 10 chains, accepted A-C rows are
  labelled with an ``assembly_size_stratum`` following the project brief's
  Structure-A/B/C split:

  * ``3_10``    — identical to the v0 pilot scope; may reach
                  Structure-A-provisional and CF-A-provisional;
  * ``11_24``   — same coordinate QC, conservative grading: structure grade
                  still A/B-provisional but counterfactual grade is capped at
                  CF-B-provisional;
  * ``gt24``    — 25-60 chain large complexes: Structure-C-provisional /
                  CF-C-provisional exploration layer for topology and
                  direct-vs-contextual tasks only, capped separately and
                  excluded from the hard-negative core counts;

- candidates are grouped by structure file so each assembly is parsed once,
  and chain-pair contact/distance results are cached per assembly across
  that file's candidates (mandatory for 25+ chain assemblies);
- coordinate validation runs in a multiprocessing pool;
- accepted A-C rows are deterministically re-ordered with the run seed
  before sequential ID assignment, so output does not depend on worker
  scheduling order.

No output row is a universal biological non-binder.  A-C means only that the
two mapped constructs do not form a direct interface in the cited assembly.
PDB contact positives remain provisional until interface-area or
PINDER-grade validation is available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v0 = _load_module(
    "structure_counterfactual_v0",
    Path(__file__).with_name("build_structure_counterfactual_v0.py"),
)
index_v1 = _load_module(
    "structure_assembly_index_v1",
    Path(__file__).with_name("build_structure_assembly_index_v1.py"),
)

chain_count_stratum = index_v1.chain_count_stratum

CORE_STRATA = ("3_10", "11_24")
LARGE_STRATUM = "gt24"

AC_FIELDS_V1 = [*v0.AC_FIELDS, "assembly_size_stratum"]
TRIPLET_FIELDS_V1 = [
    *v0.TRIPLET_FIELDS[:1],
    "assembly_size_stratum",
    *v0.TRIPLET_FIELDS[1:],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--chain-map", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-ac", type=int, default=8000)
    parser.add_argument("--target-ac-large", type=int, default=3000)
    parser.add_argument("--candidate-ac", type=int, default=100000)
    parser.add_argument("--max-chains", type=int, default=60,
                        help="hard ceiling; assemblies beyond this are rejected for coordinate cost")
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--max-canonical-coverage", type=float, default=1.2)
    parser.add_argument("--max-per-anchor", type=int, default=8)
    parser.add_argument("--max-per-pdb", type=int, default=30)
    parser.add_argument("--max-per-tax", type=int, default=4000)
    parser.add_argument("--top-d", type=int, default=3)
    parser.add_argument("--audit-size", type=int, default=300)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", default="20260806")
    return parser.parse_args()


def group_by_structure_file(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group selected candidate rows by their example structure file."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["structure_file"]].append(row)
    return dict(groups)


def deterministic_ac_order(rows: list[dict[str, str]], seed: str) -> list[dict[str, str]]:
    """Order accepted A-C rows deterministically, independent of worker order."""
    return sorted(
        rows,
        key=lambda r: v0.stable_key(seed, r["protein_a"], r["protein_c"], r["structure_file"]),
    )


def apply_stratum_caps(
    rows: list[dict[str, str]], seed: str, target_ac: int, target_ac_large: int,
) -> list[dict[str, str]]:
    """Deterministically trim accepted rows: core strata up to target_ac,
    the large-complex exploration stratum up to target_ac_large."""
    ordered = deterministic_ac_order(rows, seed)
    core = [r for r in ordered if r["assembly_size_stratum"] in CORE_STRATA][:target_ac]
    large = [r for r in ordered if r["assembly_size_stratum"] == LARGE_STRATUM][:target_ac_large]
    final = core + large
    for idx, row in enumerate(final, 1):
        row["ac_id"] = f"AC:{idx:06d}"
    return final


def counterfactual_grade_with_stratum(cf_grade: str, stratum: str) -> str:
    """Conservative grading per assembly-size stratum (v1 decision)."""
    if stratum == LARGE_STRATUM:
        return "CF-C-provisional"
    if stratum == "11_24" and cf_grade == "CF-A-provisional":
        return "CF-B-provisional"
    return cf_grade


# ---------------------------------------------------------------------------
# Worker stage: parse each assembly once, validate all its candidate pairs.
# Metadata and chain map are loaded once per worker via the pool initializer
# and inherited through fork, avoiding per-task pickling of the big tables.
# ---------------------------------------------------------------------------

_WORKER_METADATA: dict[str, v0.ProteinMeta] = {}
_WORKER_CHAIN_MAP: dict[tuple[str, str], str] = {}


def _init_worker(metadata_path: str, chain_map_path: str) -> None:
    global _WORKER_METADATA, _WORKER_CHAIN_MAP
    _WORKER_METADATA = v0.load_metadata(Path(metadata_path))
    _WORKER_CHAIN_MAP = v0.load_chain_map(Path(chain_map_path))


def _process_file(task: tuple[str, list[dict[str, str]], str, int, float]):
    structure_file, rows, structures_dir, max_chains, max_canonical_coverage = task
    path = Path(structures_dir) / structure_file
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    counts: Counter = Counter()
    try:
        assembly = v0.parse_assembly(path)
    except Exception as exc:  # parse failure rejects every candidate of the file
        for row in rows:
            rejected.append({
                "protein_a": row["uniprot_a"], "protein_c": row["uniprot_b"],
                "example_evidence": row.get("example_evidence", ""),
                "reason": "coordinate_parse_error",
                "details": f"{type(exc).__name__}: {exc}",
            })
        counts["coordinate_parse_error"] += len(rows)
        return structure_file, accepted, rejected, counts

    num_chains = len(assembly.chains)
    stratum = chain_count_stratum(num_chains)
    # Chain-pair contact/distance caches shared across all candidates of this
    # assembly; quadratic pair enumeration happens at most once per file.
    shared_caches: dict[str, dict] = {}
    for row in rows:
        reject_base = {
            "protein_a": row["uniprot_a"], "protein_c": row["uniprot_b"],
            "example_evidence": row.get("example_evidence", ""),
        }
        if num_chains > max_chains:
            rejected.append({**reject_base, "reason": "assembly_exceeds_chain_cap",
                             "details": f"chains={num_chains}"})
            counts["assembly_exceeds_chain_cap"] += 1
            continue
        if stratum == "lt3":
            rejected.append({**reject_base,
                             "reason": "fewer_than_three_protein_chains", "details": ""})
            counts["fewer_than_three_protein_chains"] += 1
            continue
        try:
            wedge, reason = v0.choose_wedge(
                assembly, row["uniprot_a"], row["uniprot_b"],
                _WORKER_METADATA, _WORKER_CHAIN_MAP,
                shared_caches=shared_caches,
            )
        except Exception as exc:
            rejected.append({**reject_base, "reason": "coordinate_parse_error",
                             "details": f"{type(exc).__name__}: {exc}"})
            counts["coordinate_parse_error"] += 1
            continue
        if wedge is None:
            rejected.append({**reject_base, "reason": reason, "details": ""})
            counts[reason] += 1
            continue
        # max_chains already enforced above; pass a ceiling that cannot trigger
        # so v0 QC applies every non-size criterion unchanged.
        reason = v0.ac_qc_reason(wedge, max_chains, num_chains, max_canonical_coverage)
        if reason:
            rejected.append({
                **reject_base, "reason": reason,
                "details": f"chains={num_chains};heavy={wedge['ac_min_heavy_a']};cbca={wedge['ac_min_cb_or_ca_a']}",
            })
            counts[reason] += 1
            continue
        margin = v0.distance_margin_status(
            v0.as_float(wedge["ac_min_heavy_a"]), v0.as_float(wedge["ac_min_cb_or_ca_a"])
        )
        if stratum == LARGE_STRATUM:
            grade = "Structure-C-provisional"
        else:
            grade = "Structure-A-provisional" if (
                wedge["bridge_edges_interface_pass"] == "1" and margin == "robust_ge_10A"
            ) else "Structure-B-provisional"
        accepted.append({
            "protein_a": row["uniprot_a"], "protein_c": row["uniprot_b"],
            "tax_id": _WORKER_METADATA[row["uniprot_a"]].tax_id,
            "pdb_id": assembly.pdb_id, "assembly_id": assembly.assembly_id,
            "structure_file": assembly.structure_file,
            **wedge,
            "num_protein_chains": str(num_chains),
            "experimental_method": assembly.method,
            "resolution": assembly.resolution, "r_free": assembly.r_free,
            "ac_distance_margin_status": margin,
            "aggregate_noncontact_assemblies": row.get("noncontact_assemblies", ""),
            "aggregate_class_a_assemblies": row.get("class_a_assemblies", ""),
            "aggregate_distinct_pdb_count": row.get("distinct_pdb_count", ""),
            "assembly_coordinate_source": "RCSB_downloaded_biological_assembly",
            "assembly_rebuild_audit_status": "official_coordinates_not_independently_rebuilt",
            "direct_positive_conflict_status": "no_PDB_contact_in_exact_pair_aggregate_external_sources_pending",
            "structure_grade_provisional": grade,
            "negative_label_semantics": "assembly_context_direct_noncontact_not_universal_nonbinder",
            "qc_status": "coordinate_verified_open_wedge",
            "assembly_size_stratum": stratum,
        })
        counts[f"{grade}|{stratum}"] += 1
    return structure_file, accepted, rejected, counts


def validate_candidates_parallel(
    selected: list[dict[str, str]], args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter]:
    groups = group_by_structure_file(selected)
    tasks = [
        (structure_file, rows, str(args.structures), args.max_chains, args.max_canonical_coverage)
        for structure_file, rows in groups.items()
    ]
    # Large files first so stragglers start early (imap_unordered keeps order
    # of completion; deterministic re-ordering happens after the pool joins).
    tasks.sort(key=lambda t: -(args.structures / t[0]).stat().st_size)
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    counts: Counter = Counter()
    done = 0
    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(str(args.metadata), str(args.chain_map)),
    ) as pool:
        for _file, acc, rej, cnt in pool.imap_unordered(_process_file, tasks, chunksize=1):
            accepted.extend(acc)
            rejected.extend(rej)
            counts.update(cnt)
            done += 1
            if done % 250 == 0 or done == len(tasks):
                print(json.dumps({
                    "files_processed": done, "files_total": len(tasks),
                    "ac_accepted": len(accepted),
                }), flush=True)
    total_accepted = len(accepted)
    final = apply_stratum_caps(accepted, args.seed, args.target_ac, args.target_ac_large)
    counts["ac_accepted"] = len(final)
    counts["ac_rejected"] = len(rejected)
    counts["ac_trimmed_beyond_target"] = total_accepted - len(final)
    return final, rejected, counts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = v0.load_metadata(args.metadata)
    chain_map = v0.load_chain_map(args.chain_map)

    # Selection reuses the v0 funnel unchanged (stable-seed order, same caps),
    # only with the file-size cap lifted via --max-file-mb.
    selected, selection_counts = v0.select_ac_candidates(
        args.pair_table, metadata, args.structures, args
    )
    ac_rows, rejected, ac_counts = validate_candidates_parallel(selected, args)
    ac_stratum = {row["ac_id"]: row["assembly_size_stratum"] for row in ac_rows}
    ad_rows, triplets, triplet_counts = v0.build_triplets(
        ac_rows, args.pair_table, metadata, args
    )
    # Stratum-aware conservative grading (v1 decision, see DECISIONS.md).
    # v0 grade counters are replaced by recomputed post-adjustment counts.
    triplet_counts.pop("CF-A-provisional", None)
    triplet_counts.pop("CF-B-provisional", None)
    grade_counts: Counter = Counter()
    for row in triplets:
        stratum = ac_stratum[row["ac_id"]]
        row["assembly_size_stratum"] = stratum
        original = row["counterfactual_grade_provisional"]
        adjusted = counterfactual_grade_with_stratum(original, stratum)
        if adjusted != original:
            triplet_counts[f"grade_adjusted_{original}_to_{adjusted}"] += 1
        row["counterfactual_grade_provisional"] = adjusted
        grade_counts[adjusted] += 1
    for grade, n in grade_counts.items():
        triplet_counts[grade] = n
    for row in ad_rows:
        row["assembly_size_stratum"] = ac_stratum[row["ac_id"]]
    sample = v0.audit_sample(triplets, args.audit_size, args.seed)

    v0.write_tsv(args.output_dir / "structure_ac_base_v1.tsv", AC_FIELDS_V1, ac_rows)
    v0.write_tsv(args.output_dir / "structure_ac_qc_rejections_v1.tsv", v0.REJECT_FIELDS, rejected)
    v0.write_tsv(args.output_dir / "direct_positive_ad_v1.tsv",
                 [*v0.AD_FIELDS, "assembly_size_stratum"], ad_rows)
    v0.write_tsv(args.output_dir / "counterfactual_structure_triplets_v1.tsv",
                 TRIPLET_FIELDS_V1, triplets)
    v0.write_tsv(args.output_dir / "triplet_manual_audit_sample_v1.tsv",
                 TRIPLET_FIELDS_V1, sample)

    summary = {
        "status": "scaled_candidate_pool_not_frozen_training_data",
        "parameters": {
            "target_ac": args.target_ac, "target_ac_large": args.target_ac_large,
            "candidate_ac": args.candidate_ac,
            "max_chains": args.max_chains, "max_file_mb": args.max_file_mb,
            "max_canonical_coverage": args.max_canonical_coverage,
            "max_per_anchor": args.max_per_anchor, "max_per_pdb": args.max_per_pdb,
            "max_per_tax": args.max_per_tax, "top_d": args.top_d,
            "workers": args.workers, "seed": args.seed,
            "v1_changes_vs_v0": [
                "0.75MB pilot file cap removed (50MB hard cap): full same-taxon Class-A pool eligible",
                "assembly_size_stratum replaces the flat >10-chain rejection: 3_10 = v0 scope, 11_24 = same QC with CF grade capped at CF-B, gt24 (25-60 chains) = Structure-C/CF-C exploration layer with separate cap",
                "candidates grouped by structure file; each assembly parsed once with shared chain-pair caches",
                "multiprocessing coordinate validation; deterministic seed re-ordering before ID assignment",
                "all other QC, wedge, grading and triplet semantics identical to v0",
            ],
        },
        "metadata_sequences": len(metadata), "chain_mappings": len(chain_map),
        "selection_counts": dict(selection_counts), "ac_qc_counts": dict(ac_counts),
        "triplet_counts": dict(triplet_counts), "manual_audit_sample_rows": len(sample),
        "semantic_constraints": {
            "negative": "assembly-context direct noncontact; not universal non-binding",
            "positive": "PDB exact contact; interface size is verified only for same-assembly bridge edges",
            "similarity": "local BLOSUM62 full-sequence identity and bidirectional coverage; domain and structure pending",
            "strata": "3_10/11_24 feed the hard-negative pool (CF-A/CF-B); gt24 is Structure-C/CF-C exploration only",
        },
        "blocking_gates_before_final_training": [
            "independent biological-assembly rebuild audit",
            "external direct-positive conflict filtering",
            "PINDER-grade interface validation for external A-D positives",
            "Pfam/domain-architecture similarity",
            "Foldseek or TM-align monomer/partner similarity",
            "homology/structure/publication grouped leakage-safe split",
            "manual stratified review",
        ],
    }
    (args.output_dir / "summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
