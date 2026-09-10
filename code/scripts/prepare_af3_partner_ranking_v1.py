#!/usr/bin/env python3
"""Prepare a leakage-aware AF3 partner-ranking pilot and provisional main set.

This script performs no AlphaFold inference. It freezes a deterministic set of
post-AF3-cutoff A-D+/A-C- comparisons, writes pair/monomer manifests and emits
AF3 monomer input JSON files for reusable MSA/template generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/interim/af3_partner_ranking_v1"
CUTOFF = "2021-09-30"
AF3_SEED = 20260818


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--main-n", type=int, default=300)
    p.add_argument("--pilot-n", type=int, default=48)
    p.add_argument("--seed-robustness-n", type=int, default=60)
    p.add_argument("--max-pair-tokens", type=int, default=1024)
    p.add_argument("--same-assembly-target", type=int, default=70)
    return p.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_fasta(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}; name = ""; chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if name: result[name] = "".join(chunks)
                name = line[1:].split()[0]; chunks = []
            else:
                chunks.append(line)
    if name: result[name] = "".join(chunks)
    return result


def stable(*parts: str) -> str:
    return hashlib.sha256(("af3-partner-v1\t" + "\t".join(parts)).encode()).hexdigest()


def pdb_id(evidence: str) -> str:
    return evidence.split(":", 1)[0].split("-", 1)[0].upper()


def release_dates(pdb_ids: list[str], cache: Path) -> dict[str, str]:
    current: dict[str, str] = {}
    if cache.is_file():
        current = json.loads(cache.read_text(encoding="utf-8"))
    missing = sorted(set(pdb_ids) - set(current))
    endpoint = "https://data.rcsb.org/graphql"
    for start in range(0, len(missing), 100):
        ids = missing[start:start + 100]
        query = "query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id rcsb_accession_info{initial_release_date}}}"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps({"query": query, "variables": {"ids": ids}}).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "ProtNegDB-AF3-plan/1.0"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.load(response)
        for entry in payload.get("data", {}).get("entries", []) or []:
            if entry and entry.get("rcsb_accession_info"):
                current[entry["rcsb_id"]] = entry["rcsb_accession_info"]["initial_release_date"][:10]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return current


def bucket(tokens: int) -> int:
    return 256 * math.ceil(tokens / 256)


def proportional_sample(rows: list[dict], n: int, keys: tuple[str, ...], label: str) -> list[dict]:
    if n >= len(rows):
        return list(rows)
    groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    quotas: dict[tuple[str, ...], int] = {}; fractions = []
    for key, values in groups.items():
        exact = n * len(values) / len(rows)
        quotas[key] = math.floor(exact)
        fractions.append((exact - quotas[key], key))
    for _fraction, key in sorted(fractions, reverse=True)[:n - sum(quotas.values())]:
        quotas[key] += 1
    selected = []
    for key, values in groups.items():
        values = sorted(values, key=lambda r: stable(label, r["ac_id"], r["triplet_id"]))
        selected.extend(values[:quotas[key]])
    if len(selected) != n:
        raise RuntimeError(f"stratified selection mismatch: {len(selected)} != {n}")
    return sorted(selected, key=lambda r: stable(label, r["ac_id"], r["triplet_id"]))


def pair_outputs(rows: list[dict], prefix: str) -> tuple[list[dict], list[dict]]:
    links = []; unique: dict[tuple[str, str], dict] = {}
    for row in rows:
        for role, partner in (("ad", row["D"]), ("ac", row["C"])):
            pair = tuple(sorted((row["A"], partner)))
            pair_key = "|".join(pair)
            job_id = f"AF3_{pair[0]}_{pair[1]}"
            links.append({
                "set": prefix, "triplet_id": row["triplet_id"], "ac_id": row["ac_id"],
                "role": role, "anchor": row["A"], "partner": partner,
                "pair_key": pair_key, "job_id": job_id,
            })
            unique.setdefault(pair, {
                "set": prefix, "pair_key": pair_key, "job_id": job_id,
                "protein_1": pair[0], "protein_2": pair[1],
                "pair_tokens": row["len_A"] + (row["len_D"] if role == "ad" else row["len_C"]),
                "bucket": bucket(row["len_A"] + (row["len_D"] if role == "ad" else row["len_C"])),
            })
    return links, sorted(unique.values(), key=lambda r: (int(r["bucket"]), r["pair_key"]))


def main() -> None:
    args = parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    master = [r for r in read_tsv(ROOT / "data/interim/figure12_master_v1/figure2_triplet_master_v1.tsv")
              if r["core_flag"] == "1"]
    raw = {r["triplet_id"]: r for r in read_tsv(
        ROOT / "data/interim/structure_counterfactual_v1/counterfactual_structure_triplets_v1.tsv")}
    assignments = {r["triplet_id"]: r for r in read_tsv(
        ROOT / "data/interim/structure_family_transfer_v4/id50/triplet_assignments_v4.tsv")}
    seqs = read_fasta(ROOT / "data/interim/structure_triplet_scoring_v1/dscript_proteins_v1.fasta")

    all_pdb = []
    for row in master:
        all_pdb.extend([pdb_id(row["pdb_assembly"]), pdb_id(raw[row["triplet_id"]]["positive_example_evidence"])])
    dates = release_dates(all_pdb, out / "rcsb_initial_release_dates_v1.json")

    reasons = Counter(); eligible = []
    for row in master:
        detail = raw[row["triplet_id"]]; split = assignments.get(row["triplet_id"], {})
        negative_pdb = pdb_id(row["pdb_assembly"])
        positive_pdb = pdb_id(detail["positive_example_evidence"])
        lengths = {role: len(seqs.get(row[role], "")) for role in ("A", "C", "D")}
        max_tokens = max(lengths["A"] + lengths["C"], lengths["A"] + lengths["D"])
        checks = [
            (not all(lengths.values()), "missing_sequence"),
            (dates.get(negative_pdb, "") <= CUTOFF, "negative_pdb_not_post_cutoff"),
            (dates.get(positive_pdb, "") <= CUTOFF, "positive_example_not_post_cutoff"),
            (max_tokens > args.max_pair_tokens, "pair_too_long"),
            (row["replicated_AC"] != "1", "ac_not_replicated"),
            (detail["all_role_shared_tax"] != "1", "roles_not_same_taxon"),
        ]
        failed = next((reason for condition, reason in checks if condition), "")
        if failed:
            reasons[failed] += 1; continue
        eligible.append({
            **{key: row[key] for key in ["triplet_id", "ac_id", "A", "C", "D", "complex_size_bin",
                                         "n_pdb_AC", "replicated_AC", "A_C_min_heavy", "A_D_min_heavy",
                                         "positive_source", "seqid_CD", "seqid_CD_bin"]},
            "negative_pdb": negative_pdb, "negative_pdb_release": dates[negative_pdb],
            "positive_example_pdb": positive_pdb, "positive_example_release": dates[positive_pdb],
            "positive_interface_status": detail["positive_interface_status"],
            "positive_rank": detail["positive_rank_for_ac_orientation"],
            "counterfactual_grade": detail["counterfactual_grade_provisional"],
            "family_split": split.get("split", ""),
            "len_A": lengths["A"], "len_C": lengths["C"], "len_D": lengths["D"],
            "tokens_AC": lengths["A"] + lengths["C"], "tokens_AD": lengths["A"] + lengths["D"],
            "max_pair_tokens": max_tokens, "bucket": bucket(max_tokens),
            "cutoff_scope": "negative_and_positive_example_PDB_post_2021-09-30; exact-pair-history audit pending",
        })

    by_ac: dict[str, list[dict]] = defaultdict(list)
    for row in eligible: by_ac[row["ac_id"]].append(row)
    one_per_ac = []
    for ac_id, rows in by_ac.items():
        rows.sort(key=lambda r: (int(r["positive_rank"] or 99),
                                 r["counterfactual_grade"] != "CF-A-provisional",
                                 stable("one-per-ac", ac_id, r["triplet_id"])))
        one_per_ac.append(rows[0])

    same = [r for r in one_per_ac if r["positive_source"] == "same_assembly_bridge"]
    aggregate = [r for r in one_per_ac if r["positive_source"] != "same_assembly_bridge"]
    n_same = min(args.same_assembly_target, len(same), args.main_n)
    chosen_same = proportional_sample(same, n_same, ("complex_size_bin", "seqid_CD_bin", "bucket"), "main-same")
    chosen_agg = proportional_sample(aggregate, args.main_n - n_same,
                                     ("complex_size_bin", "seqid_CD_bin", "bucket"), "main-aggregate")
    main_rows = sorted(chosen_same + chosen_agg, key=lambda r: stable("main", r["ac_id"], r["triplet_id"]))
    pilot_rows = proportional_sample(main_rows, args.pilot_n,
                                     ("positive_source", "complex_size_bin", "bucket"), "pilot")
    seed_rows = proportional_sample(main_rows, args.seed_robustness_n,
                                    ("positive_source", "complex_size_bin", "bucket"), "seed-robustness")

    fields = list(one_per_ac[0])
    write_tsv(out / "eligible_one_per_ac_v1.tsv", sorted(one_per_ac, key=lambda r: r["ac_id"]), fields)
    write_tsv(out / "main300_manifest_v1.tsv", main_rows, fields)
    write_tsv(out / "pilot48_manifest_v1.tsv", pilot_rows, fields)
    write_tsv(out / "seed_robustness60_manifest_v1.tsv", seed_rows, fields)

    for label, rows in (("main300", main_rows), ("pilot48", pilot_rows), ("seed60", seed_rows)):
        links, pairs = pair_outputs(rows, label)
        write_tsv(out / f"{label}_pair_links_v1.tsv", links)
        write_tsv(out / f"{label}_unique_pairs_v1.tsv", pairs)

    proteins = sorted({row[role] for row in main_rows for role in ("A", "C", "D")})
    pilot_proteins = sorted({row[role] for row in pilot_rows for role in ("A", "C", "D")})
    seed_proteins = sorted({row[role] for row in seed_rows for role in ("A", "C", "D")})
    (out / "main300_proteins_v1.txt").write_text("\n".join(proteins) + "\n", encoding="utf-8")
    (out / "pilot48_proteins_v1.txt").write_text("\n".join(pilot_proteins) + "\n", encoding="utf-8")
    (out / "seed60_proteins_v1.txt").write_text("\n".join(seed_proteins) + "\n", encoding="utf-8")
    input_dir = out / "monomer_inputs_v1"; input_dir.mkdir(exist_ok=True)
    for accession in proteins:
        payload = {
            "name": f"AF3MONO_{accession}", "modelSeeds": [AF3_SEED],
            "sequences": [{"protein": {"id": "A", "sequence": seqs[accession]}}],
            "dialect": "alphafold3", "version": 3,
        }
        (input_dir / f"{accession}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def summary(rows: list[dict]) -> dict:
        links, pairs = pair_outputs(rows, "summary")
        return {
            "triplets": len(rows), "unique_ac": len({r["ac_id"] for r in rows}),
            "unique_pairs": len(pairs), "unique_proteins": len({r[x] for r in rows for x in ("A", "C", "D")}),
            "positive_source": dict(Counter(r["positive_source"] for r in rows)),
            "complex_size": dict(Counter(r["complex_size_bin"] for r in rows)),
            "token_bucket": dict(Counter(str(r["bucket"]) for r in rows)),
        }
    report = {
        "status": "prepared_no_af3_inference",
        "protocol": {"af3_version": "3.0.3", "model_seed": AF3_SEED, "diffusion_samples": 5,
                     "training_and_template_cutoff": CUTOFF, "max_pair_tokens": args.max_pair_tokens,
                     "exact_pair_pre_cutoff_history_audit": "required before formal main300 inference"},
        "source_core_triplets": len(master), "filter_exclusions_first_failure": dict(reasons),
        "eligible_triplets_before_one_per_ac": len(eligible), "eligible_one_per_ac": len(one_per_ac),
        "main300": summary(main_rows), "pilot48": summary(pilot_rows), "seed60": summary(seed_rows),
    }
    (out / "prepare_summary_v1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
