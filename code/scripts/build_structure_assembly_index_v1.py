#!/usr/bin/env python3
"""Build an assembly chain-count index for the structure route (v1).

The v0 pilot showed that the 0.75 MB file-size cap was a poor proxy for
assembly size.  This index parses every biological-assembly file referenced
by the same-taxon Class-A pair pool once and records the protein-chain count
plus basic quality fields, so that later sampling can stratify directly by
chain count instead of compressed file size.

Scope: only files referenced by Class-A non_binding rows that pass the v0
identity/sequence/taxonomy filters (the eligible pool), not the whole
197k-file structure mirror.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

_V0_PATH = Path(__file__).with_name("build_structure_counterfactual_v0.py")
_SPEC = importlib.util.spec_from_file_location("structure_counterfactual_v0", _V0_PATH)
assert _SPEC is not None and _SPEC.loader is not None
v0 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = v0
_SPEC.loader.exec_module(v0)

INDEX_FIELDS = [
    "structure_file", "pdb_id", "assembly_id", "num_protein_chains",
    "experimental_method", "resolution", "r_free", "file_mb",
    "chain_count_stratum", "eligible_pair_count", "parse_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    return parser.parse_args()


def chain_count_stratum(num_chains: int) -> str:
    if num_chains < 3:
        return "lt3"
    if num_chains <= 10:
        return "3_10"
    if num_chains <= 24:
        return "11_24"
    return "gt24"


def collect_eligible_files(pair_table: Path, metadata: dict[str, v0.ProteinMeta]) -> Counter:
    """Same filters as v0 selection up to (but excluding) the file-size cap."""
    counts: Counter = Counter()
    with pair_table.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("final_label") != "non_binding" or row.get("best_evidence_class") != "Class A":
                continue
            a, c = row.get("uniprot_a", ""), row.get("uniprot_b", "")
            if not a or not c or a == c or a not in metadata or c not in metadata:
                continue
            if not metadata[a].tax_id or metadata[a].tax_id != metadata[c].tax_id:
                continue
            structure_file = v0.parse_example_file(row.get("example_evidence", ""))
            if structure_file:
                counts[structure_file] += 1
    return counts


def _parse_one(task: tuple[str, int, str]) -> dict[str, str]:
    structure_file, pair_count, structures_dir = task
    path = Path(structures_dir) / structure_file
    file_mb = f"{path.stat().st_size / 1024 / 1024:.3f}" if path.exists() else "nan"
    base = {
        "structure_file": structure_file,
        "eligible_pair_count": str(pair_count),
        "file_mb": file_mb,
    }
    try:
        assembly = v0.parse_assembly(path)
    except Exception as exc:
        return {**base, "pdb_id": "", "assembly_id": "", "num_protein_chains": "",
                "experimental_method": "", "resolution": "", "r_free": "",
                "chain_count_stratum": "", "parse_status": f"error:{type(exc).__name__}"}
    n = len(assembly.chains)
    return {
        **base,
        "pdb_id": assembly.pdb_id, "assembly_id": assembly.assembly_id,
        "num_protein_chains": str(n),
        "experimental_method": assembly.method,
        "resolution": assembly.resolution, "r_free": assembly.r_free,
        "chain_count_stratum": chain_count_stratum(n),
        "parse_status": "ok",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = v0.load_metadata(args.metadata)
    eligible = collect_eligible_files(args.pair_table, metadata)
    tasks = [(name, count, str(args.structures)) for name, count in sorted(eligible.items())]

    rows: list[dict[str, str]] = []
    with mp.Pool(args.workers) as pool:
        for idx, row in enumerate(pool.imap_unordered(_parse_one, tasks, chunksize=8), 1):
            rows.append(row)
            if idx % 500 == 0 or idx == len(tasks):
                print(json.dumps({"files_processed": idx, "files_total": len(tasks)}), flush=True)
    rows.sort(key=lambda r: r["structure_file"])
    v0.write_tsv(args.output_dir / "assembly_chain_count_index_v1.tsv", INDEX_FIELDS, rows)

    stratum_counts: Counter = Counter()
    stratum_pairs: Counter = Counter()
    for row in rows:
        if row["parse_status"] == "ok":
            stratum_counts[row["chain_count_stratum"]] += 1
            stratum_pairs[row["chain_count_stratum"]] += int(row["eligible_pair_count"])
        else:
            stratum_counts["parse_error"] += 1
    summary = {
        "eligible_pairs": sum(eligible.values()),
        "distinct_files": len(eligible),
        "files_by_chain_count_stratum": dict(stratum_counts),
        "eligible_pairs_by_chain_count_stratum": dict(stratum_pairs),
    }
    (args.output_dir / "summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
