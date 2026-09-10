#!/usr/bin/env python3
"""Build an auditable structure-context counterfactual triplet pilot.

The pilot starts from exact UniProt-level Class-A noncontact pairs, re-opens
the cited RCSB biological-assembly coordinate file, and verifies an explicit
open wedge.  It then expands each oriented A-C context-negative with observed
PDB contact partners D of A and measures full-sequence similarity D~C.

No output row is a universal biological non-binder.  A-C means only that the
two mapped constructs do not form a direct interface in the cited assembly.
Likewise, PDB contact positives remain provisional until interface-area or
PINDER-grade validation is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:  # Runtime pilot environment; kept optional so pure unit tests can import.
    import gemmi
except ImportError:  # pragma: no cover
    gemmi = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from Bio.Align import PairwiseAligner, substitution_matrices
except ImportError:  # pragma: no cover
    PairwiseAligner = None
    substitution_matrices = None


AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "SEC", "PYL", "ASX", "GLX", "UNK",
}


AC_FIELDS = [
    "ac_id", "protein_a", "protein_c", "tax_id", "pdb_id", "assembly_id",
    "structure_file", "chain_a", "chain_c", "bridge_chain",
    "bridge_protein", "num_protein_chains", "experimental_method",
    "resolution", "r_free", "ac_min_heavy_a", "ac_min_cb_or_ca_a",
    "ac_distance_margin_status", "ac_mutual_nearest", "a_observed_residues", "c_observed_residues",
    "a_construct_coverage", "c_construct_coverage", "a_canonical_coverage",
    "c_canonical_coverage", "ab_min_heavy_a", "bc_min_heavy_a",
    "ab_cbca_contact_pairs_8a", "bc_cbca_contact_pairs_8a",
    "ab_interface_residues_anchor", "ab_interface_residues_bridge",
    "bc_interface_residues_bridge", "bc_interface_residues_negative",
    "bridge_edges_interface_pass", "aggregate_noncontact_assemblies",
    "aggregate_class_a_assemblies", "aggregate_distinct_pdb_count",
    "assembly_coordinate_source", "assembly_rebuild_audit_status",
    "direct_positive_conflict_status", "structure_grade_provisional",
    "negative_label_semantics", "qc_status",
]


AD_FIELDS = [
    "ad_id", "ac_id", "anchor", "negative_partner_c", "positive_partner_d",
    "positive_source", "positive_example_evidence", "positive_min_heavy_a",
    "positive_binding_assemblies", "positive_distinct_pdb_count",
    "positive_interface_status", "anchor_tax_id", "c_tax_id", "d_tax_id",
    "all_role_shared_tax", "sequence_identity", "sequence_coverage_c",
    "sequence_coverage_d", "sequence_aligned_residues",
    "sequence_similarity_bin", "domain_similarity_status",
    "structure_similarity_status", "positive_rank_for_ac_orientation",
]


TRIPLET_FIELDS = [
    "triplet_id", *AD_FIELDS[1:], "ac_structure_file", "ac_chain_anchor",
    "ac_chain_negative", "ac_min_heavy_a", "ac_min_cb_or_ca_a",
    "ac_mutual_nearest", "ac_structure_grade_provisional",
    "negative_label_semantics", "counterfactual_grade_provisional",
    "training_pool_status", "leakage_control_status",
]


REJECT_FIELDS = [
    "protein_a", "protein_c", "example_evidence", "reason", "details",
]


@dataclass
class ProteinMeta:
    accession: str
    tax_id: str
    sequence: str
    organism: str = ""
    domain_group: str = ""


@dataclass
class ChainAtoms:
    label_chain: str
    auth_chain: str = ""
    entity_id: str = ""
    heavy: list[tuple[float, float, float]] = field(default_factory=list)
    cbca: list[tuple[float, float, float]] = field(default_factory=list)
    residues: set[str] = field(default_factory=set)


@dataclass
class Assembly:
    pdb_id: str
    assembly_id: str
    structure_file: str
    chains: dict[str, ChainAtoms]
    entity_lengths: dict[str, int]
    method: str = ""
    resolution: str = ""
    r_free: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-table", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--chain-map", type=Path, required=True)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-ac", type=int, default=750)
    parser.add_argument("--candidate-ac", type=int, default=2500)
    parser.add_argument("--max-chains", type=int, default=10)
    parser.add_argument("--max-file-mb", type=float, default=20.0)
    parser.add_argument("--max-canonical-coverage", type=float, default=1.2)
    parser.add_argument("--max-per-anchor", type=int, default=4)
    parser.add_argument("--max-per-pdb", type=int, default=12)
    parser.add_argument("--max-per-tax", type=int, default=60)
    parser.add_argument("--top-d", type=int, default=3)
    parser.add_argument("--audit-size", type=int, default=150)
    parser.add_argument("--seed", default="20260722")
    return parser.parse_args()


def stable_key(seed: str, *parts: str) -> str:
    return hashlib.sha256((seed + "\t" + "\t".join(parts)).encode()).hexdigest()


def parse_example_file(example: str) -> str:
    return example.split(":", 1)[0].strip()


def parse_structure_ids(filename: str) -> tuple[str, str]:
    match = re.match(r"([0-9A-Za-z]{4})-assembly([^.]+)\.cif(?:\.gz)?$", filename)
    if not match:
        return filename[:4].lower(), ""
    return match.group(1).lower(), match.group(2)


def as_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def fmt(value: float, digits: int = 3) -> str:
    return "nan" if math.isnan(value) else f"{value:.{digits}f}"


def load_metadata(path: Path) -> dict[str, ProteinMeta]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accession = row.get("accession", "")
            sequence = row.get("canonical_sequence", "").replace(" ", "").upper()
            if accession and sequence:
                result[accession] = ProteinMeta(
                    accession=accession,
                    tax_id=row.get("tax_id", ""),
                    sequence=sequence,
                    organism=row.get("organism", ""),
                    domain_group=row.get("domain_group", ""),
                )
    return result


def load_chain_map(path: Path) -> dict[tuple[str, str], str]:
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            pdb_id = row.get("pdb_id", "").lower()
            chain = row.get("chain", "")
            accession = row.get("uniprot_id", "")
            if pdb_id and chain and accession:
                result[(pdb_id, chain)] = accession
    return result


def _column(block, tag: str):
    return block.find_values(tag)


def _first_value(block, tags: Iterable[str]) -> str:
    for tag in tags:
        value = block.find_value(tag)
        if value and value not in {"?", "."}:
            return str(value).strip("'\"")
        for candidate in block.find_values(tag):
            if candidate and str(candidate) not in {"?", "."}:
                return str(candidate).strip("'\"")
    return ""


def parse_assembly(path: Path) -> Assembly:
    if gemmi is None or np is None:  # pragma: no cover
        raise RuntimeError("gemmi and numpy are required for coordinate validation")
    pdb_id, assembly_id = parse_structure_ids(path.name)
    block = gemmi.cif.read(str(path)).sole_block()
    tags = {
        key: _column(block, tag)
        for key, tag in {
            "group": "_atom_site.group_PDB",
            "label_chain": "_atom_site.label_asym_id",
            "auth_chain": "_atom_site.auth_asym_id",
            "entity": "_atom_site.label_entity_id",
            "seq": "_atom_site.label_seq_id",
            "comp": "_atom_site.label_comp_id",
            "atom": "_atom_site.label_atom_id",
            "element": "_atom_site.type_symbol",
            "x": "_atom_site.Cartn_x",
            "y": "_atom_site.Cartn_y",
            "z": "_atom_site.Cartn_z",
            "model": "_atom_site.pdbx_PDB_model_num",
        }.items()
    }
    chains: dict[str, ChainAtoms] = {}
    first_model = str(tags["model"][0]) if len(tags["model"]) else "1"
    for i in range(len(tags["group"])):
        if str(tags["group"][i]) != "ATOM":
            continue
        if len(tags["model"]) and str(tags["model"][i]) != first_model:
            continue
        comp = str(tags["comp"][i]).upper()
        if comp not in AA3 or str(tags["element"][i]).upper().startswith("H"):
            continue
        label = str(tags["label_chain"][i])
        chain = chains.setdefault(label, ChainAtoms(label_chain=label))
        if not chain.auth_chain:
            chain.auth_chain = str(tags["auth_chain"][i])
        if not chain.entity_id:
            chain.entity_id = str(tags["entity"][i])
        try:
            coord = (float(tags["x"][i]), float(tags["y"][i]), float(tags["z"][i]))
        except ValueError:
            continue
        chain.heavy.append(coord)
        seq_id = str(tags["seq"][i])
        if seq_id not in {"", ".", "?"}:
            chain.residues.add(seq_id)
        atom = str(tags["atom"][i]).upper()
        if atom == "CB" or (atom == "CA" and comp == "GLY"):
            chain.cbca.append(coord)

    entity_lengths: dict[str, int] = defaultdict(int)
    ent = _column(block, "_entity_poly_seq.entity_id")
    num = _column(block, "_entity_poly_seq.num")
    mon = _column(block, "_entity_poly_seq.mon_id")
    seen: dict[str, set[str]] = defaultdict(set)
    for i in range(min(len(ent), len(num), len(mon))):
        if str(mon[i]).upper() in AA3:
            seen[str(ent[i])].add(str(num[i]))
    for entity, residues in seen.items():
        entity_lengths[entity] = len(residues)

    return Assembly(
        pdb_id=pdb_id,
        assembly_id=assembly_id,
        structure_file=path.name,
        chains=chains,
        entity_lengths=dict(entity_lengths),
        method=_first_value(block, ["_exptl.method"]),
        resolution=_first_value(block, ["_refine.ls_d_res_high", "_em_3d_reconstruction.resolution"]),
        r_free=_first_value(block, ["_refine.ls_R_factor_R_free"]),
    )


def accession_for(
    assembly: Assembly, chain: ChainAtoms, chain_map: dict[tuple[str, str], str]
) -> str:
    return chain_map.get((assembly.pdb_id, chain.label_chain), "") or chain_map.get(
        (assembly.pdb_id, chain.auth_chain), ""
    )


def array(coords: list[tuple[float, float, float]]):
    return np.asarray(coords, dtype=np.float32)


def min_distance(coords_a, coords_b, target_pairs: int = 1_500_000) -> float:
    if not coords_a or not coords_b:
        return float("nan")
    aa, bb = array(coords_a), array(coords_b)
    if len(aa) > len(bb):
        aa, bb = bb, aa
    block = max(1, target_pairs // max(1, len(bb)))
    best2 = float("inf")
    for start in range(0, len(aa), block):
        diff = aa[start : start + block, None, :] - bb[None, :, :]
        local = float(np.einsum("ijk,ijk->ij", diff, diff, optimize=True).min())
        best2 = min(best2, local)
        if best2 == 0:
            break
    return math.sqrt(best2)


def has_contact(coords_a, coords_b, cutoff: float = 8.0) -> bool:
    if not coords_a or not coords_b:
        return False
    if len(coords_a) > len(coords_b):
        coords_a, coords_b = coords_b, coords_a
    cells: dict[tuple[int, int, int], list[tuple[float, float, float]]] = defaultdict(list)
    for x, y, z in coords_b:
        cells[(math.floor(x / cutoff), math.floor(y / cutoff), math.floor(z / cutoff))].append((x, y, z))
    cutoff2 = cutoff * cutoff
    for x, y, z in coords_a:
        base = (math.floor(x / cutoff), math.floor(y / cutoff), math.floor(z / cutoff))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for qx, qy, qz in cells.get((base[0] + dx, base[1] + dy, base[2] + dz), []):
                        if (x - qx) ** 2 + (y - qy) ** 2 + (z - qz) ** 2 <= cutoff2:
                            return True
    return False


def contact_counts(coords_a, coords_b, cutoff: float = 8.0) -> tuple[int, int, int]:
    """Return residue-proxy pair count and participating residues on each side."""
    if not coords_a or not coords_b:
        return 0, 0, 0
    aa, bb = array(coords_a), array(coords_b)
    cutoff2 = cutoff * cutoff
    pair_count = 0
    hit_a: set[int] = set()
    hit_b: set[int] = set()
    block = max(1, 1_500_000 // max(1, len(bb)))
    for start in range(0, len(aa), block):
        diff = aa[start : start + block, None, :] - bb[None, :, :]
        hits = np.argwhere(np.einsum("ijk,ijk->ij", diff, diff, optimize=True) <= cutoff2)
        pair_count += len(hits)
        hit_a.update(start + int(i) for i in hits[:, 0])
        hit_b.update(int(i) for i in hits[:, 1])
    return pair_count, len(hit_a), len(hit_b)


def interface_pass(metrics: tuple[int, int, int]) -> bool:
    pairs, left, right = metrics
    return pairs >= 5 and left >= 3 and right >= 3


def distance_margin_status(heavy: float, cbca: float) -> str:
    return "robust_ge_10A" if heavy >= 10.0 and cbca >= 10.0 else "borderline_8_to_10A"


def construct_coverage(chain: ChainAtoms, assembly: Assembly) -> float:
    denom = assembly.entity_lengths.get(chain.entity_id, 0)
    return len(chain.residues) / denom if denom else float("nan")


def canonical_coverage(chain: ChainAtoms, meta: ProteinMeta) -> float:
    return len(chain.residues) / len(meta.sequence) if meta.sequence else float("nan")


def choose_wedge(
    assembly: Assembly,
    accession_a: str,
    accession_c: str,
    metadata: dict[str, ProteinMeta],
    chain_map: dict[tuple[str, str], str],
    shared_caches: dict[str, dict] | None = None,
) -> tuple[dict[str, str] | None, str]:
    """Find the best open-wedge assignment for an A-C pair in one assembly.

    ``shared_caches`` optionally provides ``contact``/``distance`` dicts that
    are reused across repeated calls on the same assembly (v1 scale-up);
    the default None keeps the original per-call caching behaviour.
    """
    names = sorted(assembly.chains)
    if len(names) < 3:
        return None, "fewer_than_three_protein_chains"
    accessions = {
        name: accession_for(assembly, assembly.chains[name], chain_map) for name in names
    }
    a_chains = [name for name in names if accessions[name] == accession_a]
    c_chains = [name for name in names if accessions[name] == accession_c]
    if not a_chains or not c_chains:
        return None, "example_assembly_does_not_map_both_uniprot_roles"

    if shared_caches is None:
        contact_cache: dict[tuple[str, str], bool] = {}
        distance_cache: dict[tuple[str, str], float] = {}
    else:
        contact_cache = shared_caches.setdefault("contact", {})
        distance_cache = shared_caches.setdefault("distance", {})

    def key(x: str, y: str) -> tuple[str, str]:
        return (x, y) if x < y else (y, x)

    def contact(x: str, y: str) -> bool:
        pair = key(x, y)
        if pair not in contact_cache:
            contact_cache[pair] = has_contact(
                assembly.chains[pair[0]].heavy, assembly.chains[pair[1]].heavy
            )
        return contact_cache[pair]

    def distance(x: str, y: str) -> float:
        pair = key(x, y)
        if pair not in distance_cache:
            distance_cache[pair] = min_distance(
                assembly.chains[pair[0]].heavy, assembly.chains[pair[1]].heavy
            )
        return distance_cache[pair]

    candidates = []
    for chain_a in a_chains:
        for chain_c in c_chains:
            if chain_a == chain_c or contact(chain_a, chain_c):
                continue
            bridges = [
                b for b in names if b not in {chain_a, chain_c}
                and contact(chain_a, b) and contact(chain_c, b)
            ]
            for bridge in bridges:
                ac_heavy = distance(chain_a, chain_c)
                ac_cbca = min_distance(
                    assembly.chains[chain_a].cbca, assembly.chains[chain_c].cbca
                )
                ab_metrics = contact_counts(
                    assembly.chains[chain_a].cbca, assembly.chains[bridge].cbca
                )
                bc_metrics = contact_counts(
                    assembly.chains[bridge].cbca, assembly.chains[chain_c].cbca
                )
                ab_heavy = distance(chain_a, bridge)
                bc_heavy = distance(bridge, chain_c)
                nearest_a = min(
                    (distance(chain_a, other), other) for other in names if other != chain_a
                )[1]
                nearest_c = min(
                    (distance(chain_c, other), other) for other in names if other != chain_c
                )[1]
                mutual = nearest_a == chain_c and nearest_c == chain_a
                ca, cc, cb = (
                    assembly.chains[chain_a], assembly.chains[chain_c], assembly.chains[bridge]
                )
                cov_a, cov_c = construct_coverage(ca, assembly), construct_coverage(cc, assembly)
                canon_a = canonical_coverage(ca, metadata[accession_a])
                canon_c = canonical_coverage(cc, metadata[accession_c])
                robust = interface_pass(ab_metrics) and interface_pass(bc_metrics)
                score = (
                    int(robust), min(ab_metrics[1], ab_metrics[2], bc_metrics[1], bc_metrics[2]),
                    min(cov_a if not math.isnan(cov_a) else 0, cov_c if not math.isnan(cov_c) else 0),
                    ac_heavy,
                )
                candidates.append((score, {
                    "chain_a": chain_a,
                    "chain_c": chain_c,
                    "bridge_chain": bridge,
                    "bridge_protein": accessions.get(bridge, ""),
                    "ac_min_heavy_a": fmt(ac_heavy),
                    "ac_min_cb_or_ca_a": fmt(ac_cbca),
                    "ac_mutual_nearest": "1" if mutual else "0",
                    "a_observed_residues": str(len(ca.residues)),
                    "c_observed_residues": str(len(cc.residues)),
                    "a_construct_coverage": fmt(cov_a, 4),
                    "c_construct_coverage": fmt(cov_c, 4),
                    "a_canonical_coverage": fmt(canon_a, 4),
                    "c_canonical_coverage": fmt(canon_c, 4),
                    "ab_min_heavy_a": fmt(ab_heavy),
                    "bc_min_heavy_a": fmt(bc_heavy),
                    "ab_cbca_contact_pairs_8a": str(ab_metrics[0]),
                    "bc_cbca_contact_pairs_8a": str(bc_metrics[0]),
                    "ab_interface_residues_anchor": str(ab_metrics[1]),
                    "ab_interface_residues_bridge": str(ab_metrics[2]),
                    "bc_interface_residues_bridge": str(bc_metrics[1]),
                    "bc_interface_residues_negative": str(bc_metrics[2]),
                    "bridge_edges_interface_pass": "1" if robust else "0",
                }))
    if not candidates:
        return None, "mapped_pair_is_not_an_explicit_open_wedge_in_example_assembly"
    return max(candidates, key=lambda item: item[0])[1], ""


def ac_qc_reason(
    wedge: dict[str, str], max_chains: int, num_chains: int,
    max_canonical_coverage: float = 1.2,
) -> str:
    if num_chains > max_chains:
        return "assembly_exceeds_chain_cap"
    if as_float(wedge["ac_min_heavy_a"]) <= 8:
        return "ac_heavy_atom_contact"
    if as_float(wedge["ac_min_cb_or_ca_a"]) <= 8:
        return "ac_cbca_contact"
    if wedge["ac_mutual_nearest"] == "1":
        return "ac_mutual_nearest_neighbor"
    if min(as_int(wedge["a_observed_residues"]), as_int(wedge["c_observed_residues"])) < 30:
        return "short_observed_construct"
    construct_cov = [as_float(wedge["a_construct_coverage"]), as_float(wedge["c_construct_coverage"])]
    if any(not math.isnan(x) and x < 0.5 for x in construct_cov):
        return "low_construct_coverage"
    canonical_cov = [as_float(wedge["a_canonical_coverage"]), as_float(wedge["c_canonical_coverage"])]
    if any(not math.isnan(x) and x < 0.3 for x in canonical_cov):
        return "low_canonical_coverage"
    if any(not math.isnan(x) and x > max_canonical_coverage for x in canonical_cov):
        return "canonical_coverage_exceeds_construct_guardrail"
    return ""


def similarity_bin(identity: float, coverage_c: float, coverage_d: float) -> str:
    coverage = min(coverage_c, coverage_d)
    if identity >= 0.50 and coverage >= 0.70:
        return "high"
    if identity >= 0.30 and coverage >= 0.60:
        return "medium"
    if identity >= 0.20 and coverage >= 0.50:
        return "remote_sequence"
    return "low_or_unresolved"


def make_aligner():
    if PairwiseAligner is None or substitution_matrices is None:  # pragma: no cover
        raise RuntimeError("Biopython is required for sequence similarity")
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


# BLOSUM62 covers the 20 standard residues plus B/Z/X/*; canonical UniProt
# sequences can also carry U (selenocysteine, observed in the metadata),
# O (pyrrolysine) or J. Map them to the nearest standard residue so the
# local aligner stays inside its alphabet. Length-preserving, so alignment
# coordinates still index the original strings.
ALIGNMENT_RESIDUE_MAP = str.maketrans({"U": "C", "O": "K", "J": "L"})


def sequence_similarity(seq_c: str, seq_d: str, aligner) -> tuple[float, float, float, int]:
    seq_c = seq_c.translate(ALIGNMENT_RESIDUE_MAP)
    seq_d = seq_d.translate(ALIGNMENT_RESIDUE_MAP)
    alignment = aligner.align(seq_c, seq_d)[0]
    blocks_c, blocks_d = alignment.aligned
    matches = 0
    aligned = 0
    for (start_c, end_c), (start_d, end_d) in zip(blocks_c, blocks_d):
        length = min(int(end_c - start_c), int(end_d - start_d))
        aligned += length
        matches += sum(
            x == y for x, y in zip(seq_c[int(start_c):int(start_c) + length], seq_d[int(start_d):int(start_d) + length])
        )
    identity = matches / aligned if aligned else 0.0
    return identity, aligned / len(seq_c), aligned / len(seq_d), aligned


def select_ac_candidates(
    pair_table: Path, metadata: dict[str, ProteinMeta], structures: Path, args: argparse.Namespace
) -> tuple[list[dict[str, str]], Counter]:
    eligible = []
    counts = Counter()
    with pair_table.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("final_label") != "non_binding" or row.get("best_evidence_class") != "Class A":
                continue
            counts["aggregate_class_a"] += 1
            a, c = row.get("uniprot_a", ""), row.get("uniprot_b", "")
            if not a or not c or a == c or a not in metadata or c not in metadata:
                counts["excluded_id_or_sequence"] += 1
                continue
            if not metadata[a].tax_id or metadata[a].tax_id != metadata[c].tax_id:
                counts["excluded_taxonomy"] += 1
                continue
            structure_file = parse_example_file(row.get("example_evidence", ""))
            path = structures / structure_file
            if not structure_file or not path.exists():
                counts["excluded_missing_structure"] += 1
                continue
            if path.stat().st_size > args.max_file_mb * 1024 * 1024:
                counts["excluded_large_structure"] += 1
                continue
            row = dict(row)
            row["structure_file"] = structure_file
            eligible.append(row)
    eligible.sort(key=lambda r: stable_key(args.seed, r["uniprot_a"], r["uniprot_b"], r["structure_file"]))
    anchor_counts = Counter()
    pdb_counts = Counter()
    tax_counts = Counter()
    selected = []
    for row in eligible:
        a, c = row["uniprot_a"], row["uniprot_b"]
        pdb_id, _ = parse_structure_ids(row["structure_file"])
        tax = metadata[a].tax_id
        if anchor_counts[a] >= args.max_per_anchor or anchor_counts[c] >= args.max_per_anchor:
            counts["cap_anchor"] += 1
            continue
        if pdb_counts[pdb_id] >= args.max_per_pdb:
            counts["cap_pdb"] += 1
            continue
        if tax_counts[tax] >= args.max_per_tax:
            counts["cap_tax"] += 1
            continue
        selected.append(row)
        anchor_counts[a] += 1
        anchor_counts[c] += 1
        pdb_counts[pdb_id] += 1
        tax_counts[tax] += 1
        if len(selected) >= args.candidate_ac:
            break
    counts["eligible_before_caps"] = len(eligible)
    counts["coordinate_candidates_selected"] = len(selected)
    return selected, counts


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_ac_candidates(
    candidates: list[dict[str, str]], metadata: dict[str, ProteinMeta],
    chain_map: dict[tuple[str, str], str], args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter]:
    accepted, rejected = [], []
    counts = Counter()
    for idx, row in enumerate(candidates, 1):
        if idx % 100 == 0:
            print(json.dumps({"coordinate_candidates_processed": idx, "ac_accepted": len(accepted)}), flush=True)
        path = args.structures / row["structure_file"]
        try:
            assembly = parse_assembly(path)
            if len(assembly.chains) > args.max_chains:
                rejected.append({
                    "protein_a": row["uniprot_a"], "protein_c": row["uniprot_b"],
                    "example_evidence": row.get("example_evidence", ""),
                    "reason": "assembly_exceeds_chain_cap",
                    "details": f"chains={len(assembly.chains)}",
                })
                counts["assembly_exceeds_chain_cap"] += 1
                continue
            wedge, reason = choose_wedge(
                assembly, row["uniprot_a"], row["uniprot_b"], metadata, chain_map
            )
        except Exception as exc:
            rejected.append({
                "protein_a": row["uniprot_a"], "protein_c": row["uniprot_b"],
                "example_evidence": row.get("example_evidence", ""),
                "reason": "coordinate_parse_error", "details": f"{type(exc).__name__}: {exc}",
            })
            counts["coordinate_parse_error"] += 1
            continue
        if wedge is None:
            rejected.append({
                "protein_a": row["uniprot_a"], "protein_c": row["uniprot_b"],
                "example_evidence": row.get("example_evidence", ""),
                "reason": reason, "details": "",
            })
            counts[reason] += 1
            continue
        reason = ac_qc_reason(
            wedge, args.max_chains, len(assembly.chains), args.max_canonical_coverage
        )
        if reason:
            rejected.append({
                "protein_a": row["uniprot_a"], "protein_c": row["uniprot_b"],
                "example_evidence": row.get("example_evidence", ""),
                "reason": reason,
                "details": f"chains={len(assembly.chains)};heavy={wedge['ac_min_heavy_a']};cbca={wedge['ac_min_cb_or_ca_a']}",
            })
            counts[reason] += 1
            continue
        margin = distance_margin_status(
            as_float(wedge["ac_min_heavy_a"]), as_float(wedge["ac_min_cb_or_ca_a"])
        )
        grade = "Structure-A-provisional" if (
            wedge["bridge_edges_interface_pass"] == "1" and margin == "robust_ge_10A"
        ) else "Structure-B-provisional"
        accepted.append({
            "ac_id": f"AC:{len(accepted) + 1:06d}",
            "protein_a": row["uniprot_a"], "protein_c": row["uniprot_b"],
            "tax_id": metadata[row["uniprot_a"]].tax_id,
            "pdb_id": assembly.pdb_id, "assembly_id": assembly.assembly_id,
            "structure_file": assembly.structure_file,
            **wedge,
            "num_protein_chains": str(len(assembly.chains)),
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
        })
        counts[grade] += 1
        if len(accepted) >= args.target_ac:
            break
    counts["ac_accepted"] = len(accepted)
    counts["ac_rejected"] = len(rejected)
    return accepted, rejected, counts


def load_positive_adjacency(
    pair_table: Path, anchors: set[str], metadata: dict[str, ProteinMeta]
) -> dict[str, list[dict[str, str]]]:
    adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
    with pair_table.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("final_label") != "binding":
                continue
            left, right = row.get("uniprot_a", ""), row.get("uniprot_b", "")
            if left not in metadata or right not in metadata or left == right:
                continue
            if left in anchors:
                adjacency[left].append({**row, "partner": right})
            if right in anchors:
                adjacency[right].append({**row, "partner": left})
    return adjacency


def positive_tier(row: dict[str, str]) -> tuple[str, int]:
    distance = as_float(row.get("min_heavy_atom_distance", ""))
    pdbs = as_int(row.get("distinct_pdb_count", ""))
    if not math.isnan(distance) and distance <= 5 and pdbs >= 2:
        return "repeated_exact_PDB_contact_interface_size_pending", 3
    if not math.isnan(distance) and distance <= 5:
        return "single_exact_PDB_contact_interface_size_pending", 2
    return "threshold_PDB_contact_interface_size_pending", 1


def build_triplets(
    ac_rows: list[dict[str, str]], pair_table: Path, metadata: dict[str, ProteinMeta],
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter]:
    anchors = {row[role] for row in ac_rows for role in ("protein_a", "protein_c")}
    adjacency = load_positive_adjacency(pair_table, anchors, metadata)
    aligner = make_aligner()
    similarity_cache: dict[tuple[str, str], tuple[float, float, float, int]] = {}
    ad_rows, triplets = [], []
    counts = Counter()

    for ac in ac_rows:
        for anchor_role, negative_role, anchor_chain_role, negative_chain_role in (
            ("protein_a", "protein_c", "chain_a", "chain_c"),
            ("protein_c", "protein_a", "chain_c", "chain_a"),
        ):
            anchor, negative = ac[anchor_role], ac[negative_role]
            candidates: dict[str, dict[str, str]] = {}
            for positive in adjacency.get(anchor, []):
                d = positive["partner"]
                if d in {anchor, negative}:
                    continue
                candidates[d] = {
                    **positive,
                    "positive_source": "PDB_exact_pair_aggregate",
                }
            bridge = ac.get("bridge_protein", "")
            if bridge and bridge in metadata and bridge not in {anchor, negative}:
                bridge_distance = ac["ab_min_heavy_a"] if anchor_role == "protein_a" else ac["bc_min_heavy_a"]
                bridge_interface = ac["bridge_edges_interface_pass"]
                current = candidates.get(bridge, {})
                candidates[bridge] = {
                    **current,
                    "partner": bridge,
                    "positive_source": "same_assembly_bridge",
                    "example_evidence": f"{ac['structure_file']}:{ac[anchor_chain_role]}-{ac['bridge_chain']}",
                    "min_heavy_atom_distance": bridge_distance,
                    "binding_assemblies": current.get("binding_assemblies", "1"),
                    "distinct_pdb_count": current.get("distinct_pdb_count", "1"),
                    "bridge_interface_pass": bridge_interface,
                }
            ranked = []
            for d, positive in candidates.items():
                key = tuple(sorted((negative, d)))
                if key not in similarity_cache:
                    similarity_cache[key] = sequence_similarity(
                        metadata[negative].sequence, metadata[d].sequence, aligner
                    )
                identity, cov_c, cov_d, aligned = similarity_cache[key]
                sim_bin = similarity_bin(identity, cov_c, cov_d)
                tier, tier_rank = positive_tier(positive)
                if positive.get("positive_source") == "same_assembly_bridge":
                    tier = (
                        "same_assembly_bridge_interface_pass"
                        if positive.get("bridge_interface_pass") == "1"
                        else "same_assembly_bridge_small_interface"
                    )
                    tier_rank = 4 if positive.get("bridge_interface_pass") == "1" else 2
                sim_rank = {"high": 4, "medium": 3, "remote_sequence": 2, "low_or_unresolved": 1}[sim_bin]
                ranked.append((sim_rank, identity, min(cov_c, cov_d), tier_rank, d, positive, tier, cov_c, cov_d, aligned))
            ranked.sort(reverse=True)
            for rank, item in enumerate(ranked[: args.top_d], 1):
                _, identity, _, _, d, positive, tier, cov_c, cov_d, aligned = item
                sim_bin = similarity_bin(identity, cov_c, cov_d)
                shared_tax = len({metadata[anchor].tax_id, metadata[negative].tax_id, metadata[d].tax_id}) == 1
                ad = {
                    "ad_id": f"AD:{len(ad_rows) + 1:07d}", "ac_id": ac["ac_id"],
                    "anchor": anchor, "negative_partner_c": negative, "positive_partner_d": d,
                    "positive_source": positive.get("positive_source", ""),
                    "positive_example_evidence": positive.get("example_evidence", ""),
                    "positive_min_heavy_a": positive.get("min_heavy_atom_distance", ""),
                    "positive_binding_assemblies": positive.get("binding_assemblies", ""),
                    "positive_distinct_pdb_count": positive.get("distinct_pdb_count", ""),
                    "positive_interface_status": tier,
                    "anchor_tax_id": metadata[anchor].tax_id,
                    "c_tax_id": metadata[negative].tax_id, "d_tax_id": metadata[d].tax_id,
                    "all_role_shared_tax": "1" if shared_tax else "0",
                    "sequence_identity": fmt(identity, 4), "sequence_coverage_c": fmt(cov_c, 4),
                    "sequence_coverage_d": fmt(cov_d, 4),
                    "sequence_aligned_residues": str(aligned),
                    "sequence_similarity_bin": sim_bin,
                    "domain_similarity_status": "pending_Pfam_or_domain_architecture_mapping",
                    "structure_similarity_status": "pending_Foldseek_or_TM_align",
                    "positive_rank_for_ac_orientation": str(rank),
                }
                ad_rows.append(ad)
                counts[f"ad_similarity_{sim_bin}"] += 1
                if sim_bin == "low_or_unresolved":
                    continue
                if tier == "threshold_PDB_contact_interface_size_pending":
                    continue
                cf_grade = "CF-A-provisional" if (
                    sim_bin in {"high", "medium"}
                    and tier in {"same_assembly_bridge_interface_pass", "repeated_exact_PDB_contact_interface_size_pending"}
                    and shared_tax
                    and ac["structure_grade_provisional"] == "Structure-A-provisional"
                ) else "CF-B-provisional"
                triplets.append({
                    "triplet_id": f"SCF:{len(triplets) + 1:07d}", **ad,
                    "ac_structure_file": ac["structure_file"],
                    "ac_chain_anchor": ac[anchor_chain_role],
                    "ac_chain_negative": ac[negative_chain_role],
                    "ac_min_heavy_a": ac["ac_min_heavy_a"],
                    "ac_min_cb_or_ca_a": ac["ac_min_cb_or_ca_a"],
                    "ac_mutual_nearest": ac["ac_mutual_nearest"],
                    "ac_structure_grade_provisional": ac["structure_grade_provisional"],
                    "negative_label_semantics": ac["negative_label_semantics"],
                    "counterfactual_grade_provisional": cf_grade,
                    "training_pool_status": "pilot_candidate_pending_external_conflict_and_structure_similarity",
                    "leakage_control_status": "pending_homology_structure_publication_cluster_split",
                })
                counts[cf_grade] += 1
    counts["oriented_ac_with_any_positive"] = sum(
        bool(adjacency.get(ac[role])) for ac in ac_rows for role in ("protein_a", "protein_c")
    )
    counts["ad_rows"] = len(ad_rows)
    counts["triplet_rows"] = len(triplets)
    return ad_rows, triplets, counts


def audit_sample(rows: list[dict[str, str]], size: int, seed: str) -> list[dict[str, str]]:
    if len(rows) <= size:
        return rows
    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["sequence_similarity_bin"], row["positive_source"],
            row["counterfactual_grade_provisional"],
        )
        strata[key].append(row)
    for key, values in strata.items():
        values.sort(key=lambda r: stable_key(seed, "audit", *key, r["triplet_id"]))
    sample = []
    keys = sorted(strata)
    while len(sample) < size and keys:
        next_keys = []
        for key in keys:
            if strata[key] and len(sample) < size:
                sample.append(strata[key].pop(0))
            if strata[key]:
                next_keys.append(key)
        keys = next_keys
    return sample


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.metadata)
    chain_map = load_chain_map(args.chain_map)
    selected, selection_counts = select_ac_candidates(
        args.pair_table, metadata, args.structures, args
    )
    ac_rows, rejected, ac_counts = validate_ac_candidates(
        selected, metadata, chain_map, args
    )
    ad_rows, triplets, triplet_counts = build_triplets(
        ac_rows, args.pair_table, metadata, args
    )
    sample = audit_sample(triplets, args.audit_size, args.seed)

    write_tsv(args.output_dir / "structure_ac_base_v0.tsv", AC_FIELDS, ac_rows)
    write_tsv(args.output_dir / "structure_ac_qc_rejections_v0.tsv", REJECT_FIELDS, rejected)
    write_tsv(args.output_dir / "direct_positive_ad_v0.tsv", AD_FIELDS, ad_rows)
    write_tsv(args.output_dir / "counterfactual_structure_triplets_v0.tsv", TRIPLET_FIELDS, triplets)
    write_tsv(args.output_dir / "triplet_manual_audit_sample_v0.tsv", TRIPLET_FIELDS, sample)

    summary = {
        "status": "feasibility_pilot_not_frozen_training_data",
        "parameters": {
            "target_ac": args.target_ac, "candidate_ac": args.candidate_ac,
            "max_chains": args.max_chains, "max_file_mb": args.max_file_mb,
            "max_canonical_coverage": args.max_canonical_coverage,
            "max_per_anchor": args.max_per_anchor, "max_per_pdb": args.max_per_pdb,
            "max_per_tax": args.max_per_tax, "top_d": args.top_d,
            "seed": args.seed,
            "pilot_sampling_caveat": "compressed-file-size cap enriches smaller assemblies and is not a representative full-PDB sample",
        },
        "metadata_sequences": len(metadata), "chain_mappings": len(chain_map),
        "selection_counts": dict(selection_counts), "ac_qc_counts": dict(ac_counts),
        "triplet_counts": dict(triplet_counts), "manual_audit_sample_rows": len(sample),
        "semantic_constraints": {
            "negative": "assembly-context direct noncontact; not universal non-binding",
            "positive": "PDB exact contact; interface size is verified only for same-assembly bridge edges",
            "similarity": "local BLOSUM62 full-sequence identity and bidirectional coverage; domain and structure pending",
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
    (args.output_dir / "summary_v0.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
