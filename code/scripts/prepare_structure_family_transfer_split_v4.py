#!/usr/bin/env python3
"""Triplet-atomic family-transfer split for structure-core hard-negative FT.

Split happens on (A, D+, C-) rows before any 0/1 pair explosion.

Family = MMseqs cluster (identity + bidirectional coverage), never PDB
co-assembly union and never Pfam clan.

Held-out bins (generalization):
  protein_unseen_family_seen
      A, C, D accessions are absent from train proteins, but each of their
      clusters has at least one other member in train.
  family_unseen
      all three clusters are absent from train (all-protein-family-disjoint).

Train/local fit is recorded for the training triplets only. It is not a
held-out test set.

Matched-random negatives permute C inside the train protein pool, within
the same C taxon, and drop pairs that are known structure contacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

CORE_STRATA = ("3_10", "11_24")
LOCKED_SEEDS = (20260816, 20260817, 20260818)
LOCKED_TRAIN = {
    "epochs": 3,
    "batch_size": 8,
    "grad_accum": 4,
    "lr": 1e-5,
    "head_lr_scale": 10.0,
    "warmup_frac": 0.05,
    "max_length": 1024,
    "patience": 2,
    "early_stop_metric": "internal_dev_auprc",
    "checkpoint": "PLM-interact-650M-humanV12",
}
SPLITS = (
    "train",
    "protein_unseen_family_seen",
    "family_unseen",
    "dropped_mixed_family",
    "dropped_mixed_protein",
)


@dataclass(frozen=True)
class Triplet:
    triplet_id: str
    ac_id: str
    anchor: str
    c: str
    d: str
    pdb_id: str
    stratum: str
    tax_a: str
    tax_c: str
    tax_d: str
    pdb_count: int

    @property
    def proteins(self) -> tuple[str, str, str]:
        return (self.anchor, self.c, self.d)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", type=Path, required=True)
    parser.add_argument("--ac-table", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=None,
                        help="scoring_pairs_v1.tsv; used as known-contact filter")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-seq-id", type=float, default=0.5)
    parser.add_argument("--unseen-frac", type=float, default=0.15)
    parser.add_argument("--family-seen-frac", type=float, default=0.15)
    parser.add_argument("--min-train-frac", type=float, default=0.50)
    parser.add_argument("--drop-pdb-overlap", action="store_true", default=True)
    parser.add_argument("--keep-pdb-overlap", action="store_true",
                        help="report PDB overlap but keep overlapping test triplets")
    parser.add_argument("--internal-dev-frac", type=float, default=0.1,
                        help="hash-split of eligible train triplets for early stop; not a test bin")
    parser.add_argument("--partition-seed", type=int, default=0,
                        help="0 reproduces the locked partition; nonzero values randomize legal tie/order choices")
    parser.add_argument("--fasta-only", type=Path, default=None,
                        help="write core-triplet protein FASTA and exit")
    return parser.parse_args()


def load_metadata(path: Path) -> dict[str, str]:
    seqs = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            acc, seq = row.get("accession", ""), row.get("canonical_sequence", "")
            if acc and seq:
                seqs[acc] = seq.replace(" ", "").upper()
    return seqs


def load_clusters(path: Path) -> dict[str, str]:
    member_to_rep: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            member_to_rep[fields[1]] = fields[0]
    return member_to_rep


def load_ac_table(path: Path) -> dict[str, dict[str, str]]:
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["ac_id"]] = row
    return rows


def load_known_contacts(path: Path) -> set[frozenset[str]]:
    contacts: set[frozenset[str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("role") != "ad_positive":
                continue
            contacts.add(frozenset({row["protein_x"], row["protein_y"]}))
    return contacts


def load_core_triplets(path: Path, ac_table: dict[str, dict[str, str]]) -> list[Triplet]:
    out: list[Triplet] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["assembly_size_stratum"] not in CORE_STRATA:
                continue
            ac = ac_table[row["ac_id"]]
            out.append(Triplet(
                triplet_id=row["triplet_id"],
                ac_id=row["ac_id"],
                anchor=row["anchor"],
                c=row["negative_partner_c"],
                d=row["positive_partner_d"],
                pdb_id=ac["pdb_id"],
                stratum=row["assembly_size_stratum"],
                tax_a=row.get("anchor_tax_id", ac.get("tax_id", "")),
                tax_c=row.get("c_tax_id", ac.get("tax_id", "")),
                tax_d=row.get("d_tax_id", ""),
                pdb_count=int(ac.get("aggregate_distinct_pdb_count") or 0),
            ))
    out.sort(key=lambda t: t.triplet_id)
    return out


def family_of(protein: str, clusters: dict[str, str]) -> str:
    return clusters.get(protein, protein)


def family_set(trip: Triplet, clusters: dict[str, str]) -> frozenset[str]:
    return frozenset(family_of(p, clusters) for p in trip.proteins)


def seeded_order(value: str, seed: int) -> str:
    if seed == 0:
        return value
    return hashlib.sha256(f"{seed}\t{value}".encode("utf-8")).hexdigest()


def family_components(triplets: list[Triplet], clusters: dict[str, str]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for trip in triplets:
        fams = list(family_set(trip, clusters))
        for fam in fams:
            find(fam)
        for other in fams[1:]:
            union(fams[0], other)
    return {fam: find(fam) for fam in parent}


def pack_unseen_families(
    triplets: list[Triplet],
    clusters: dict[str, str],
    unseen_frac: float,
    min_train_frac: float,
    partition_seed: int = 0,
) -> set[str]:
    """Hold out whole family-co-occurrence components, smallest first.

    Does not cut a giant component. Cutting is a separate step if the packed
    unseen set is still too small.
    """
    comp = family_components(triplets, clusters)
    by_comp: dict[str, list[Triplet]] = defaultdict(list)
    for trip in triplets:
        by_comp[comp[next(iter(family_set(trip, clusters)))]].append(trip)
    if partition_seed:
        # Sensitivity partitions sample whole, disconnected family components
        # in seeded order. Components are never cut, so family disjointness is
        # invariant while the held-out family composition genuinely changes.
        ranked = sorted(by_comp.items(), key=lambda item: seeded_order(item[0], partition_seed))
    else:
        ranked = sorted(by_comp.items(), key=lambda item: (len(item[1]), item[0]))
    n = len(triplets)
    unseen: set[str] = set()
    unseen_n = 0
    for cid, rows in ranked:
        fams = {f for trip in rows for f in family_set(trip, clusters)}
        if (unseen_n or partition_seed) and (unseen_n + len(rows)) / n > unseen_frac + 1e-12:
            continue
        train_after = n - unseen_n - len(rows)
        # remaining after this assignment is an upper bound on train; mixed
        # from later protein split still comes out of it.
        if train_after / n < min_train_frac:
            continue
        unseen |= fams
        unseen_n += len(rows)
        if unseen_n / n >= unseen_frac:
            break
    return unseen


def grow_unseen_inside_giant(
    triplets: list[Triplet],
    clusters: dict[str, str],
    unseen: set[str],
    unseen_frac: float,
    min_train_frac: float,
    partition_seed: int = 0,
) -> set[str]:
    """Grow a closed family set if small co-occurrence components are not enough."""
    n = len(triplets)

    def counts(s: set[str]) -> tuple[int, int, int]:
        internal = disjoint = cut = 0
        for trip in triplets:
            fams = family_set(trip, clusters)
            if fams <= s:
                internal += 1
            elif fams.isdisjoint(s):
                disjoint += 1
            else:
                cut += 1
        return internal, disjoint, cut

    internal, _disjoint, _cut = counts(unseen)
    if internal / n >= unseen_frac:
        return unseen

    iteration = 0
    while internal / n < unseen_frac:
        feasible: list[tuple[tuple, frozenset[str]]] = []
        seen_sets: set[frozenset[str]] = set()
        for trip in triplets:
            fams = family_set(trip, clusters)
            if fams <= unseen or fams in seen_sets:
                continue
            seen_sets.add(fams)
            trial = unseen | set(fams)
            inn, dis, cut = counts(trial)
            if dis / n < min_train_frac:
                continue
            if inn <= internal:
                continue
            score = (inn - internal, -cut, -len(trial - unseen),
                     seeded_order(min(fams), partition_seed))
            feasible.append((score, fams))
        if not feasible:
            break
        feasible.sort(key=lambda item: item[0], reverse=True)
        if partition_seed:
            # Alternative legal partitions sample deterministically among the
            # 100 best closure-growth moves. This changes held-out families,
            # while the same downstream leakage and minimum-train gates apply.
            top = feasible[:min(100, len(feasible))]
            digest = hashlib.sha256(
                f"{partition_seed}\t{iteration}".encode("utf-8")).hexdigest()
            best = top[int(digest[:12], 16) % len(top)]
        else:
            best = feasible[0]
        unseen = unseen | set(best[1])
        internal, _disjoint, _cut = counts(unseen)
        iteration += 1
    return unseen


def assign_family_bins(
    triplets: list[Triplet],
    clusters: dict[str, str],
    unseen_families: set[str],
) -> dict[str, str]:
    assigned: dict[str, str] = {}
    for trip in triplets:
        fams = family_set(trip, clusters)
        if fams <= unseen_families:
            assigned[trip.triplet_id] = "family_unseen"
        elif fams.isdisjoint(unseen_families):
            assigned[trip.triplet_id] = "train_side"
        else:
            assigned[trip.triplet_id] = "dropped_mixed_family"
    return assigned


def protein_counts(triplets: list[Triplet]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for trip in triplets:
        counts.update(trip.proteins)
    return counts


def reserve_train_proteins(
    triplets: list[Triplet],
    clusters: dict[str, str],
) -> set[str]:
    members: dict[str, set[str]] = defaultdict(set)
    for trip in triplets:
        for protein in trip.proteins:
            members[family_of(protein, clusters)].add(protein)
    counts = protein_counts(triplets)
    reserved: set[str] = set()
    for fam, proteins in members.items():
        if len(proteins) == 1:
            reserved.add(next(iter(proteins)))
            continue
        reserved.add(max(proteins, key=lambda p: (counts[p], p)))
    return reserved


def assign_family_seen(
    train_side: list[Triplet],
    clusters: dict[str, str],
    family_seen_frac: float,
    min_train_frac: float,
    universe_n: int,
    partition_seed: int = 0,
) -> tuple[set[str], list[Triplet], list[Triplet], list[Triplet]]:
    reserved = reserve_train_proteins(train_side, clusters)
    members: dict[str, set[str]] = defaultdict(set)
    for trip in train_side:
        for protein in trip.proteins:
            members[family_of(protein, clusters)].add(protein)
    held: set[str] = set()
    selected: list[Triplet] = []
    candidates = [
        trip for trip in train_side
        if all(p not in reserved for p in trip.proteins)
        and all(len(members[family_of(p, clusters)]) >= 2 for p in trip.proteins)
    ]
    if partition_seed:
        candidates.sort(key=lambda t: seeded_order(t.triplet_id, partition_seed))
    target_n = max(1, int(round(family_seen_frac * universe_n)))
    min_train_n = int(round(min_train_frac * universe_n))
    for trip in candidates:
        trial = held | set(trip.proteins)
        if any(not (members[family_of(p, clusters)] - trial) for p in trip.proteins):
            continue
        train_proteins = {p for t in train_side for p in t.proteins} - trial
        train_n = sum(1 for other in train_side if set(other.proteins) <= train_proteins)
        if train_n < min_train_n:
            continue
        held = trial
        selected.append(trip)
        if len(selected) >= target_n:
            break

    train_proteins = {p for t in train_side for p in t.proteins} - held
    train_rows: list[Triplet] = []
    seen_rows: list[Triplet] = []
    mixed: list[Triplet] = []
    for trip in train_side:
        prots = set(trip.proteins)
        if prots <= train_proteins:
            train_rows.append(trip)
        elif prots <= held:
            ok = all(
                members[family_of(p, clusters)] - held
                for p in trip.proteins
            )
            if ok:
                seen_rows.append(trip)
            else:
                mixed.append(trip)
        else:
            mixed.append(trip)
    train_p = {p for t in train_rows for p in t.proteins}
    kept_seen: list[Triplet] = []
    for trip in seen_rows:
        ok = all(
            any(family_of(q, clusters) == family_of(p, clusters) for q in train_p)
            for p in trip.proteins
        ) and set(trip.proteins).isdisjoint(train_p)
        if ok:
            kept_seen.append(trip)
        else:
            mixed.append(trip)
    return held, train_rows, kept_seen, mixed


def pdb_overlap_ids(train: list[Triplet], test: list[Triplet]) -> set[str]:
    train_pdb = {t.pdb_id for t in train if t.pdb_id}
    return {t.triplet_id for t in test if t.pdb_id in train_pdb}


def apply_pdb_filter(
    train: list[Triplet],
    seen: list[Triplet],
    unseen: list[Triplet],
    enabled: bool,
) -> tuple[list[Triplet], list[Triplet], list[Triplet], dict[str, int]]:
    seen_overlap = pdb_overlap_ids(train, seen)
    unseen_overlap = pdb_overlap_ids(train, unseen)
    stats = {
        "family_seen_pdb_overlap": len(seen_overlap),
        "family_unseen_pdb_overlap": len(unseen_overlap),
    }
    dropped = []
    if not enabled:
        return seen, unseen, dropped, stats
    dropped.extend(t for t in seen if t.triplet_id in seen_overlap)
    dropped.extend(t for t in unseen if t.triplet_id in unseen_overlap)
    seen = [t for t in seen if t.triplet_id not in seen_overlap]
    unseen = [t for t in unseen if t.triplet_id not in unseen_overlap]
    return seen, unseen, dropped, stats


def matched_random_c(
    train: list[Triplet],
    known_contacts: set[frozenset[str]],
) -> list[dict[str, str]]:
    """Permute C inside each taxon, keeping the C multiset and A/D proteins."""

    def legal(trip: Triplet, cand: str) -> bool:
        if cand in {trip.anchor, trip.d, trip.c}:
            return False
        if frozenset({trip.anchor, cand}) in known_contacts:
            return False
        return True

    by_taxon: dict[str, list[Triplet]] = defaultdict(list)
    for trip in train:
        by_taxon[trip.tax_c].append(trip)
    out: list[dict[str, str]] = []
    for taxon, rows in sorted(by_taxon.items()):
        n = len(rows)
        chosen = [""] * n
        used: set[int] = set()
        pool = [trip.c for trip in rows]
        for i, trip in enumerate(rows):
            pick = None
            for prefer_derange in (True, False):
                for j, cand in enumerate(pool):
                    if j in used:
                        continue
                    if prefer_derange and cand == trip.c:
                        continue
                    if not legal(trip, cand):
                        continue
                    pick = j
                    break
                if pick is not None:
                    break
            if pick is None:
                chosen[i] = trip.c
            else:
                used.add(pick)
                chosen[i] = pool[pick]
        for i, trip in enumerate(rows):
            status = "ok"
            if n == 1:
                status = "unpermuted_singleton_taxon"
            elif chosen[i] == trip.c:
                status = "no_legal_permutation"
            out.append({
                "triplet_id": trip.triplet_id,
                "anchor": trip.anchor,
                "c_structure": trip.c,
                "c_random": chosen[i],
                "d": trip.d,
                "taxon_c": taxon,
                "permutation_status": status,
            })
    out.sort(key=lambda r: r["triplet_id"])
    return out


def unique_pairs(triplets: list[Triplet]) -> tuple[list[dict[str, str]], int]:
    labelled: dict[frozenset[str], dict[str, str]] = {}
    conflicts = 0
    skip: set[frozenset[str]] = set()
    for trip in triplets:
        for other, label in ((trip.d, "1"), (trip.c, "0")):
            key = frozenset({trip.anchor, other})
            if key in skip:
                continue
            rec = labelled.get(key)
            if rec is None:
                labelled[key] = {
                    "protein_x": trip.anchor,
                    "protein_y": other,
                    "label": label,
                }
                continue
            if rec["label"] != label:
                conflicts += 1
                skip.add(key)
                labelled.pop(key, None)
    rows = sorted(labelled.values(), key=lambda r: (r["protein_x"], r["protein_y"], r["label"]))
    return rows, conflicts


def triplet_pair_rows(triplets: list[Triplet]) -> list[dict[str, str]]:
    """One positive and one negative per triplet. Do not collapse duplicates.

    Collapsing unique pairs would make Structure-FT and Random-FT have different
    optimizer step counts, because random C pairings collide less often.
    """
    rows = []
    for trip in sorted(triplets, key=lambda t: t.triplet_id):
        rows.append({"protein_x": trip.anchor, "protein_y": trip.d, "label": "1"})
        rows.append({"protein_x": trip.anchor, "protein_y": trip.c, "label": "0"})
    return rows


def split_stats(
    name: str,
    rows: list[Triplet],
    clusters: dict[str, str],
) -> dict[str, int]:
    proteins_a = {t.anchor for t in rows}
    proteins_c = {t.c for t in rows}
    proteins_d = {t.d for t in rows}
    fams = {family_of(p, clusters) for t in rows for p in t.proteins}
    return {
        "split": name,
        "triplets": len(rows),
        "unique_A": len(proteins_a),
        "unique_C": len(proteins_c),
        "unique_D": len(proteins_d),
        "unique_proteins": len(proteins_a | proteins_c | proteins_d),
        "mmseqs_families": len(fams),
        "unique_ac": len({t.ac_id for t in rows}),
        "replicated_ac_triplets": sum(1 for t in rows if t.pdb_count >= 2),
        "replicated_unique_ac": len({t.ac_id for t in rows if t.pdb_count >= 2}),
    }


def family_size_histogram(rows: list[Triplet], clusters: dict[str, str]) -> dict[str, int]:
    fam_n: Counter[str] = Counter()
    for trip in rows:
        for fam in family_set(trip, clusters):
            fam_n[fam] += 1
    buckets = Counter()
    for n in fam_n.values():
        if n == 1:
            buckets["1"] += 1
        elif n <= 5:
            buckets["2-5"] += 1
        elif n <= 20:
            buckets["6-20"] += 1
        elif n <= 100:
            buckets["21-100"] += 1
        else:
            buckets["gt100"] += 1
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pair_csv_rows(
    pairs: list[dict[str, str]],
    sequences: dict[str, str],
) -> list[dict[str, str]]:
    out = []
    for rec in pairs:
        sx, sy = sequences.get(rec["protein_x"], ""), sequences.get(rec["protein_y"], "")
        if not sx or not sy:
            continue
        out.append({"query": sx, "text": sy, "label": rec["label"]})
    return out


def leakage_report(
    train: list[Triplet],
    seen: list[Triplet],
    unseen: list[Triplet],
    clusters: dict[str, str],
) -> dict[str, object]:
    train_p = {p for t in train for p in t.proteins}
    seen_p = {p for t in seen for p in t.proteins}
    unseen_p = {p for t in unseen for p in t.proteins}
    train_f = {family_of(p, clusters) for p in train_p}
    seen_f = {family_of(p, clusters) for p in seen_p}
    unseen_f = {family_of(p, clusters) for p in unseen_p}
    seen_ok = True
    for trip in seen:
        if any(p in train_p for p in trip.proteins):
            seen_ok = False
        for p in trip.proteins:
            fam = family_of(p, clusters)
            others = {q for q in train_p if family_of(q, clusters) == fam}
            if not others:
                seen_ok = False
    return {
        "accession_overlap_train_seen": len(train_p & seen_p),
        "accession_overlap_train_unseen": len(train_p & unseen_p),
        "accession_overlap_seen_unseen": len(seen_p & unseen_p),
        "family_overlap_train_unseen": len(train_f & unseen_f),
        "family_overlap_train_seen": len(train_f & seen_f),
        "pdb_overlap_train_seen": len(pdb_overlap_ids(train, seen)),
        "pdb_overlap_train_unseen": len(pdb_overlap_ids(train, unseen)),
        "family_seen_definition_holds": seen_ok,
        "family_unseen_all_three_disjoint": len(train_f & unseen_f) == 0,
    }


def run_split(
    triplets: list[Triplet],
    clusters: dict[str, str],
    unseen_frac: float,
    family_seen_frac: float,
    min_train_frac: float,
    drop_pdb_overlap: bool,
    partition_seed: int = 0,
) -> dict[str, object]:
    unseen_families = pack_unseen_families(
        triplets, clusters, unseen_frac, min_train_frac, partition_seed)
    unseen_families = grow_unseen_inside_giant(
        triplets, clusters, unseen_families, unseen_frac, min_train_frac, partition_seed)
    family_assign = assign_family_bins(triplets, clusters, unseen_families)
    train_side = [t for t in triplets if family_assign[t.triplet_id] == "train_side"]
    family_unseen = [t for t in triplets if family_assign[t.triplet_id] == "family_unseen"]
    dropped_family = [t for t in triplets if family_assign[t.triplet_id] == "dropped_mixed_family"]
    held, train, seen, mixed_protein = assign_family_seen(
        train_side, clusters, family_seen_frac, min_train_frac, len(triplets), partition_seed)
    seen, family_unseen, dropped_pdb, pdb_stats = apply_pdb_filter(
        train, seen, family_unseen, drop_pdb_overlap)
    return {
        "unseen_families": unseen_families,
        "held_proteins": held,
        "train": train,
        "protein_unseen_family_seen": seen,
        "family_unseen": family_unseen,
        "dropped_mixed_family": dropped_family,
        "dropped_mixed_protein": mixed_protein,
        "dropped_pdb_overlap": dropped_pdb,
        "pdb_stats": pdb_stats,
        "family_assign": family_assign,
    }


def write_fasta(path: Path, sequences: dict[str, str], accessions: list[str]) -> None:
    missing = [acc for acc in accessions if acc not in sequences]
    if missing:
        raise SystemExit(f"missing sequences for {len(missing)} accessions e.g. {missing[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for acc in accessions:
            handle.write(f">{acc}\n{sequences[acc]}\n")


def eligible_random_rows(random_map: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in random_map if row["permutation_status"] == "ok"]


def close_eligible_train(
    train: list[Triplet],
    known_contacts: set[frozenset[str]],
) -> tuple[list[Triplet], list[dict[str, str]], list[Triplet]]:
    """Drop triplets until every remaining row has a legal within-pool permutation."""
    remaining = list(train)
    dropped: list[Triplet] = []
    random_map: list[dict[str, str]] = []
    while remaining:
        random_map = matched_random_c(remaining, known_contacts)
        ok_ids = {row["triplet_id"] for row in eligible_random_rows(random_map)}
        if len(ok_ids) == len(remaining):
            break
        dropped.extend(t for t in remaining if t.triplet_id not in ok_ids)
        remaining = [t for t in remaining if t.triplet_id in ok_ids]
    return remaining, random_map, dropped


def revalidate_family_seen(
    seen: list[Triplet],
    train: list[Triplet],
    clusters: dict[str, str],
) -> tuple[list[Triplet], list[Triplet]]:
    train_p = {p for trip in train for p in trip.proteins}
    kept: list[Triplet] = []
    dropped: list[Triplet] = []
    for trip in seen:
        if any(p in train_p for p in trip.proteins):
            dropped.append(trip)
            continue
        ok = all(
            any(family_of(q, clusters) == family_of(p, clusters) for q in train_p)
            for p in trip.proteins
        )
        if ok:
            kept.append(trip)
        else:
            dropped.append(trip)
    return kept, dropped


def hash_dev_ids(triplet_ids: list[str], frac: float, partition_seed: int = 0) -> set[str]:
    ranked = sorted(
        triplet_ids,
        key=lambda tid: (hashlib.sha256(
            (tid if partition_seed == 0 else f"{partition_seed}\t{tid}").encode("utf-8")
        ).hexdigest(), tid),
    )
    n_dev = int(round(frac * len(ranked)))
    return set(ranked[:n_dev])


def dominant_family(rows: list[Triplet], clusters: dict[str, str]) -> tuple[str, int]:
    counts: Counter[str] = Counter()
    for trip in rows:
        for protein in trip.proteins:
            counts[family_of(protein, clusters)] += 1
    if not counts:
        return "", 0
    fam, n = counts.most_common(1)[0]
    return fam, n
    missing = [acc for acc in accessions if acc not in sequences]
    if missing:
        raise SystemExit(f"missing sequences for {len(missing)} accessions e.g. {missing[:5]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for acc in accessions:
            handle.write(f">{acc}\n{sequences[acc]}\n")


def main() -> None:
    args = parse_args()
    sequences = load_metadata(args.metadata)
    ac_table = load_ac_table(args.ac_table)
    triplets = load_core_triplets(args.triplets, ac_table)
    if not triplets:
        raise SystemExit("no core triplets")
    accessions = sorted({p for trip in triplets for p in trip.proteins})
    if args.fasta_only is not None:
        write_fasta(args.fasta_only, sequences, accessions)
        print(json.dumps({"n_proteins": len(accessions), "fasta": str(args.fasta_only)}))
        return
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --fasta-only")
    if args.clusters is None:
        raise SystemExit("--clusters is required unless --fasta-only")
    drop_pdb = args.drop_pdb_overlap and not args.keep_pdb_overlap
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clusters = load_clusters(args.clusters)
    if args.pairs is None:
        raise SystemExit("--pairs is required unless --fasta-only")
    known = load_known_contacts(args.pairs)
    for trip in triplets:
        known.add(frozenset({trip.anchor, trip.d}))

    result = run_split(
        triplets, clusters, args.unseen_frac, args.family_seen_frac,
        args.min_train_frac, drop_pdb, args.partition_seed)

    train: list[Triplet] = result["train"]  # type: ignore[assignment]
    seen: list[Triplet] = result["protein_unseen_family_seen"]  # type: ignore[assignment]
    unseen: list[Triplet] = result["family_unseen"]  # type: ignore[assignment]
    dropped_f: list[Triplet] = result["dropped_mixed_family"]  # type: ignore[assignment]
    dropped_p: list[Triplet] = result["dropped_mixed_protein"]  # type: ignore[assignment]
    dropped_pdb: list[Triplet] = result["dropped_pdb_overlap"]  # type: ignore[assignment]

    remaining, random_map, ineligible = close_eligible_train(train, known)
    write_tsv(
        args.output_dir / "train_matched_random_v4.tsv",
        random_map,
        ["triplet_id", "anchor", "c_structure", "c_random", "d", "taxon_c",
         "permutation_status"],
    )
    train_full = train
    train = remaining
    eligible_rand = eligible_random_rows(random_map)
    seen, seen_dropped = revalidate_family_seen(seen, train, clusters)
    random_by_id = {row["triplet_id"]: row for row in eligible_rand}

    def as_random(trip: Triplet) -> Triplet:
        rec = random_by_id[trip.triplet_id]
        return Triplet(
            triplet_id=trip.triplet_id, ac_id=trip.ac_id, anchor=trip.anchor,
            c=rec["c_random"], d=trip.d, pdb_id=trip.pdb_id, stratum=trip.stratum,
            tax_a=trip.tax_a, tax_c=trip.tax_c, tax_d=trip.tax_d,
            pdb_count=trip.pdb_count,
        )

    random_eligible = [as_random(trip) for trip in train]
    struct_proteins = {p for trip in train for p in trip.proteins}
    random_proteins = {p for trip in random_eligible for p in trip.proteins}
    if struct_proteins != random_proteins:
        raise SystemExit(
            f"eligible protein pools differ: structure {len(struct_proteins)} "
            f"vs random {len(random_proteins)}")
    if any(trip.c == random_by_id[trip.triplet_id]["c_structure"] for trip in random_eligible):
        raise SystemExit("eligible random arm still contains original structural C")

    dev_ids = hash_dev_ids([trip.triplet_id for trip in train], args.internal_dev_frac,
                           args.partition_seed)
    train_fit = [trip for trip in train if trip.triplet_id not in dev_ids]
    train_dev = [trip for trip in train if trip.triplet_id in dev_ids]
    random_fit = [trip for trip in random_eligible if trip.triplet_id not in dev_ids]
    random_dev = [trip for trip in random_eligible if trip.triplet_id in dev_ids]

    assignment_rows = []
    split_map = {
        "train_eligible": train,
        "train_ineligible_unpermuted": ineligible,
        "protein_unseen_family_seen": seen,
        "family_unseen": unseen,
        "dropped_mixed_family": dropped_f,
        "dropped_mixed_protein": dropped_p,
        "dropped_pdb_overlap": dropped_pdb,
        "dropped_family_seen_after_eligible": seen_dropped,
    }
    for split, rows in split_map.items():
        for trip in rows:
            assignment_rows.append({
                "triplet_id": trip.triplet_id,
                "split": split,
                "internal_dev": "1" if trip.triplet_id in dev_ids else "0",
                "ac_id": trip.ac_id,
                "anchor": trip.anchor,
                "c": trip.c,
                "d": trip.d,
                "fam_a": family_of(trip.anchor, clusters),
                "fam_c": family_of(trip.c, clusters),
                "fam_d": family_of(trip.d, clusters),
                "pdb_id": trip.pdb_id,
                "pdb_count": str(trip.pdb_count),
                "stratum": trip.stratum,
                "replicated_noncontact": "1" if trip.pdb_count >= 2 else "0",
            })
    write_tsv(
        args.output_dir / "triplet_assignments_v4.tsv",
        assignment_rows,
        ["triplet_id", "split", "internal_dev", "ac_id", "anchor", "c", "d",
         "fam_a", "fam_c", "fam_d", "pdb_id", "pdb_count", "stratum",
         "replicated_noncontact"],
    )

    _, struct_conflicts = unique_pairs(train_fit)
    _, random_conflicts = unique_pairs(random_fit)
    structure_pairs = triplet_pair_rows(train_fit)
    random_pairs = triplet_pair_rows(random_fit)
    structure_dev_pairs = triplet_pair_rows(train_dev)
    random_dev_pairs = triplet_pair_rows(random_dev)
    full_struct_pairs = triplet_pair_rows(train_full)

    def write_csv(name: str, pairs: list[dict[str, str]]) -> int:
        rows = pair_csv_rows(pairs, sequences)
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["query", "text", "label"],
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    n_struct = write_csv("train_structure_bce_v4.csv", structure_pairs)
    n_rand = write_csv("train_matched_random_bce_v4.csv", random_pairs)
    n_struct_dev = write_csv("dev_structure_bce_v4.csv", structure_dev_pairs)
    n_rand_dev = write_csv("dev_matched_random_bce_v4.csv", random_dev_pairs)
    n_full = write_csv("train_structure_full2124_bce_v4.csv", full_struct_pairs)
    if n_struct != n_rand:
        raise SystemExit(
            f"train BCE row mismatch after missing-sequence filter: "
            f"structure={n_struct} random={n_rand}"
        )
    if n_struct_dev != n_rand_dev:
        raise SystemExit(
            f"dev BCE row mismatch after missing-sequence filter: "
            f"structure={n_struct_dev} random={n_rand_dev}"
        )

    eval_rows = []
    for split, rows in (
        ("train_eligible_local_fit", train),
        ("protein_unseen_family_seen", seen),
        ("family_unseen", unseen),
    ):
        for trip in rows:
            sx = sequences.get(trip.anchor, "")
            sc = sequences.get(trip.c, "")
            sd = sequences.get(trip.d, "")
            if not sx or not sc or not sd:
                continue
            eval_rows.append({
                "directed_pair_id": f"{trip.triplet_id}|AC",
                "pair_id": f"{trip.triplet_id}|AC",
                "triplet_id": trip.triplet_id,
                "split": split,
                "role": "ac",
                "query": sx,
                "text": sc,
            })
            eval_rows.append({
                "directed_pair_id": f"{trip.triplet_id}|AD",
                "pair_id": f"{trip.triplet_id}|AD",
                "triplet_id": trip.triplet_id,
                "split": split,
                "role": "ad",
                "query": sx,
                "text": sd,
            })
    with (args.output_dir / "eval_pairs_v4.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["directed_pair_id", "pair_id", "triplet_id", "split", "role", "query", "text"],
            lineterminator="\n")
        writer.writeheader()
        writer.writerows(eval_rows)
    write_fasta(
        args.output_dir / "train_eligible_proteins.fasta",
        sequences,
        sorted(struct_proteins),
    )
    eval_acc = sorted({p for t in seen + unseen for p in t.proteins})
    write_fasta(args.output_dir / "eval_heldout_proteins.fasta", sequences, eval_acc)

    table = [
        split_stats("train_eligible", train, clusters),
        split_stats("train_fit_internal", train_fit, clusters),
        split_stats("train_dev_internal", train_dev, clusters),
        split_stats("train_ineligible_unpermuted", ineligible, clusters),
        split_stats("protein_unseen_family_seen", seen, clusters),
        split_stats("family_unseen", unseen, clusters),
        split_stats("dropped_mixed_family", dropped_f, clusters),
        split_stats("dropped_mixed_protein", dropped_p, clusters),
        split_stats("dropped_pdb_overlap", dropped_pdb, clusters),
        split_stats("dropped_family_seen_after_eligible", seen_dropped, clusters),
        split_stats(
            "replicated_subset_all_assigned_tests",
            [t for t in seen + unseen if t.pdb_count >= 2],
            clusters,
        ),
        split_stats(
            "replicated_protein_unseen_family_seen",
            [t for t in seen if t.pdb_count >= 2],
            clusters,
        ),
        split_stats(
            "replicated_family_unseen",
            [t for t in unseen if t.pdb_count >= 2],
            clusters,
        ),
    ]
    write_tsv(
        args.output_dir / "split_size_table_v4.tsv",
        [{k: str(v) for k, v in row.items()} for row in table],
        ["split", "triplets", "unique_A", "unique_C", "unique_D",
         "unique_proteins", "mmseqs_families", "unique_ac",
         "replicated_ac_triplets", "replicated_unique_ac"],
    )

    leak = leakage_report(train, seen, unseen, clusters)
    dom_fam, dom_n = dominant_family(seen, clusters)
    protocol = {
        "figure": "3",
        "partition_seed": args.partition_seed,
        "claim": "structure hard-negative supervision transfer, not pretrained family bias",
        "main_train_triplets": len(train),
        "supplementary_full_structure_triplets": len(train_full),
        "internal_dev_triplets": len(train_dev),
        "family_seen_triplets": len(seen),
        "family_unseen_triplets": len(unseen),
        "family_unseen_sealed": True,
        "family_seen_not_used_for_selection": True,
        "random_arm_keeps_original_c": False,
        "same_protein_pool": sorted(struct_proteins) == sorted(random_proteins),
        "n_protein_pool": len(struct_proteins),
        "collapse_unique_pairs": False,
        "bce_rows_per_triplet": 2,
        "dominant_family_seen_cluster": dom_fam,
        "dominant_family_seen_triplet_mentions": dom_n,
        "leave_dominant_cluster_out": True,
        "primary_endpoint": "triplet P(s(A,D)>s(A,C)); ties 0.5",
        "secondary_margin": "m=s(A,D)-s(A,C); delta vs original and vs random FT",
        "S_partner": "min(max identity of C to train proteins, max identity of D to train proteins)",
        "S_anchor_supplement_only": True,
        "bootstrap": "cluster bootstrap by anchor MMseqs family",
        "seeds": list(LOCKED_SEEDS),
        "train_hparams": LOCKED_TRAIN,
        "original_arm": "published humanV12, no continued training",
        "replicated_noncontact": "aggregate_distinct_pdb_count>=2; panel not a 3d curve",
    }
    (args.output_dir / "figure3_protocol_v4.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    leak = leakage_report(train, seen, unseen, clusters)
    summary = {
        "n_core_triplets": len(triplets),
        "min_seq_id": args.min_seq_id,
        "unseen_frac_target": args.unseen_frac,
        "family_seen_frac_target": args.family_seen_frac,
        "min_train_frac": args.min_train_frac,
        "partition_seed": args.partition_seed,
        "drop_pdb_overlap": drop_pdb,
        "n_unseen_families": len(result["unseen_families"]),  # type: ignore[arg-type]
        "n_held_proteins": len(result["held_proteins"]),  # type: ignore[arg-type]
        "table": table,
        "family_size_hist_train": family_size_histogram(train, clusters),
        "family_size_hist_family_seen": family_size_histogram(seen, clusters),
        "family_size_hist_family_unseen": family_size_histogram(unseen, clusters),
        "pdb_filter": result["pdb_stats"],
        "leakage": leak,
        "matched_random": {
            "n_full_train": len(random_map),
            "n_eligible": len(eligible_rand),
            "n_ineligible_dropped_from_main": len(ineligible),
            "status": dict(Counter(r["permutation_status"] for r in random_map)),
        },
        "train_structure_bce_rows": n_struct,
        "train_matched_random_bce_rows": n_rand,
        "dev_structure_bce_rows": n_struct_dev,
        "dev_matched_random_bce_rows": n_rand_dev,
        "train_structure_full2124_bce_rows": n_full,
        "train_structure_pair_conflicts": struct_conflicts,
        "train_random_pair_conflicts": random_conflicts,
        "label_semantics": "assembly_context_direct_noncontact_not_universal_nonbinder",
        "local_fit_is_not_heldout": True,
        "protocol": protocol,
        "go_no_go": {
            "family_unseen_triplets": len(unseen),
            "family_seen_triplets": len(seen),
            "family_unseen_families": split_stats("family_unseen", unseen, clusters)["mmseqs_families"],
            "enough_for_main_text": len(unseen) >= 200 and split_stats(
                "family_unseen", unseen, clusters)["mmseqs_families"] >= 30,
        },
    }
    (args.output_dir / "summary_v4.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "table": table,
        "leakage": leak,
        "go_no_go": summary["go_no_go"],
        "eligible_train": len(train),
        "ineligible_dropped": len(ineligible),
        "family_seen_after_eligible": len(seen),
    }, indent=2))


if __name__ == "__main__":
    main()
