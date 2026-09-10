#!/usr/bin/env python3
"""Merge cached AF3 monomer data JSONs into dimer inference inputs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/af3_partner_ranking_v1"
SEED = 20260818
DEFAULT_GPUS = "6,7"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", dest="set_name", default="pilot48", choices=("pilot48", "main300", "seed60"))
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--gpus", default=DEFAULT_GPUS, help="comma-separated GPU indices, e.g. 0,1,2,3")
    p.add_argument("--allow-missing", action="store_true",
                   help="write ready pairs only; do not exit if some monomers are still running")
    p.add_argument("--skip-completed", action="store_true",
                   help="omit pairs that already have a non-empty summary_confidences.json")
    return p.parse_args()


def parse_gpus(text: str) -> list[int]:
    gpus = [int(part) for part in text.split(",") if part.strip() != ""]
    if not gpus or any(gpu < 0 or gpu > 7 for gpu in gpus):
        raise ValueError(f"gpus must be a non-empty subset of 0-7, got {text!r}")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"duplicate GPU indices: {text!r}")
    return gpus


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def monomer_json(base: Path, accession: str) -> Path:
    return base / "monomer_data_v1" / accession / f"AF3MONO_{accession}" / f"AF3MONO_{accession}_data.json"


def completed_job_ids(base: Path) -> set[str]:
    done: set[str] = set()
    for path in base.glob("dimer_outputs_*/*/*_summary_confidences.json"):
        if path.is_file() and path.stat().st_size > 0:
            done.add(path.parent.name)
    return done


def main() -> None:
    args = parse_args(); base = args.base
    gpus = parse_gpus(args.gpus)
    pairs = read_tsv(base / f"{args.set_name}_unique_pairs_v1.tsv")
    already = completed_job_ids(base) if args.skip_completed else set()
    output = base / f"dimer_inputs_{args.set_name}_seed1_v1"
    output.mkdir(parents=True, exist_ok=True)
    for child in output.glob("*.json"):
        child.unlink()
    missing = []; skipped = []; written = []; costs = []
    for row in pairs:
        if args.skip_completed and row["job_id"] in already:
            skipped.append(row["job_id"]); continue
        proteins = [row["protein_1"], row["protein_2"]]
        paths = [monomer_json(base, accession) for accession in proteins]
        if any(not path.is_file() for path in paths):
            missing.append({"pair_key": row["pair_key"], "missing": ",".join(
                accession for accession, path in zip(proteins, paths) if not path.is_file())})
            continue
        entities = []
        for chain_id, path in zip(("A", "B"), paths):
            data = json.loads(path.read_text(encoding="utf-8"))
            protein = copy.deepcopy(data["sequences"][0]["protein"])
            protein["id"] = chain_id
            entities.append({"protein": protein})
        payload = {
            "name": row["job_id"], "modelSeeds": [SEED], "sequences": entities,
            "dialect": "alphafold3", "version": 3,
        }
        destination = output / f"{row['job_id']}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(row["job_id"]); costs.append((int(row["bucket"]) ** 3, row, destination))

    gpu_dirs = {gpu: base / f"dimer_inputs_{args.set_name}_gpu{gpu}_v1" for gpu in gpus}
    for directory in gpu_dirs.values():
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
    loads = {gpu: 0 for gpu in gpus}; assignments = []
    for cost, row, path in sorted(costs, reverse=True, key=lambda x: (x[0], x[1]["pair_key"])):
        gpu = min(loads, key=lambda key: (loads[key], key))
        (gpu_dirs[gpu] / path.name).symlink_to(path)
        loads[gpu] += cost
        assignments.append({"job_id": row["job_id"], "gpu": gpu, "bucket": row["bucket"],
                            "pair_tokens": row["pair_tokens"], "pair_key": row["pair_key"]})
    fields = ["job_id", "gpu", "bucket", "pair_tokens", "pair_key"]
    with (base / f"{args.set_name}_gpu_assignments_v1.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(sorted(assignments, key=lambda x: (x["gpu"], int(x["bucket"]), x["job_id"])))
    report = {"set": args.set_name, "requested_pairs": len(pairs), "written": len(written),
              "skipped_completed": len(skipped), "missing": missing,
              "gpus": gpus,
              "gpu_jobs": {str(gpu): len(list(path.glob("*.json"))) for gpu, path in gpu_dirs.items()},
              "gpu_cubic_load": loads}
    (base / f"build_dimer_inputs_{args.set_name}_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if missing and not args.allow_missing:
        raise SystemExit(f"{len(missing)} dimer inputs are waiting for monomer data")
    if not written:
        raise SystemExit("no dimer inputs written")


if __name__ == "__main__":
    main()
