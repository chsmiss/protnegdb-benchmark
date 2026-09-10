#!/usr/bin/env python3
"""Build the independent manually reviewed ProtNeg-Assay cohort for Figure 6.

The supplied review table is treated as a set of 480 manually accepted
negatives.  Pre-existing rule tiers are carried through only as source columns
and are never used for inclusion or classification.  The Figure 6 semantics
hierarchy depends only on the reported assay method.  Only within-reported-
taxon UniProt mappings are eligible for the primary model audit.  A separate
paired-ranking input uses same-taxon IntAct MI:0407 direct-positive candidates
and never changes the negative evidence labels.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data/external/protneg_assay_v1/ProtNegDB_manual_verified_negatives_480_raw.tsv"
DEFAULT_MAPPING = ROOT / "data/interim/protnegdb_main_negative_partners_v1/main_negative_pair_mapping_v1.tsv"
DEFAULT_SEQUENCES = ROOT / "data/interim/protnegdb_main_negative_partners_v1/partner_similarity_uniprot_sequences_v1.tsv"
DEFAULT_POSITIVES = ROOT / "data/interim/protnegdb_main_negative_partners_v1/literature_direct_positive_candidates_v1.tsv"
DEFAULT_NEGATOME = ROOT / "data/interim/negatome_manual_plminteract_v1/negatome_manual_pairs_v1.tsv"
DEFAULT_STRING = ROOT / "data/external/model_training_sets/plm_interact/string_v12/protein.pairs_human_V12_train.tsv"
DEFAULT_SWISSPROT = Path("/data/chs/12.codebuddy/bridge/data/uniprot_sprot.fasta.gz")
DEFAULT_OUT = ROOT / "data/interim/figure6_protneg_assay_v1"

BIOPHYSICAL_DIRECT_ASSAYS = {
    "SPR", "ITC", "BLI", "MST", "fluorescence_anisotropy",
}
OTHER_DIRECT_ASSAYS = {
    "pull_down", "in_vitro_binding_unspecified", "far_western",
}
ASSOCIATION_ASSAYS = {"coIP", "Y2H", "PLA", "BiFC", "FRET"}
MAX_TOKENS = 1024
SPECIAL_TOKENS = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    p.add_argument("--positive-candidates", type=Path, default=DEFAULT_POSITIVES)
    p.add_argument("--negatome-pairs", type=Path, default=DEFAULT_NEGATOME)
    p.add_argument("--string-train", type=Path, default=DEFAULT_STRING)
    p.add_argument("--swissprot", type=Path, default=DEFAULT_SWISSPROT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_pairs(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["directed_pair_id", "query", "text"], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parent_accession(value: str) -> str:
    return value.strip().split("-", 1)[0]


def accession_pair_key(a: str, b: str) -> str:
    return "|".join(sorted((parent_accession(a), parent_accession(b))))


def sequence_pair_key(a: str, b: str) -> bytes:
    left, right = sorted((a, b))
    return hashlib.sha1(f"{left}\t{right}".encode()).digest()


def evidence_class(row: dict[str, str]) -> str:
    """Classify solely from assay_normalized; ignore supplied rule tiers."""
    assay = row.get("assay_normalized")
    if assay in BIOPHYSICAL_DIRECT_ASSAYS:
        return "biophysical_direct"
    if assay in OTHER_DIRECT_ASSAYS:
        return "other_direct_binding"
    if assay in ASSOCIATION_ASSAYS:
        return "association_negative"
    return "other_assay"


def load_sequences(path: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for row in read_tsv(path):
        acc, seq = row.get("accession", ""), row.get("sequence", "").strip().upper()
        if not acc or not seq:
            continue
        record = {"sequence": seq, "length": len(seq), "source": str(path)}
        old = out.get(acc)
        if old and old["sequence"] != seq:
            raise SystemExit(f"conflicting cached sequences for {acc}")
        out[acc] = record
    return out


def add_swissprot_sequences(
    path: Path, sequences: dict[str, dict[str, object]], needed: set[str]
) -> None:
    """Fill missing reviewed accessions from the local Swiss-Prot FASTA."""
    opener = gzip.open if path.suffix == ".gz" else path.open
    accession = ""
    chunks: list[str] = []

    def flush() -> None:
        if accession and accession in needed and accession not in sequences and chunks:
            seq = "".join(chunks).upper()
            sequences[accession] = {"sequence": seq, "length": len(seq), "source": str(path)}

    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                flush()
                fields = line.split("|", 2)
                accession = fields[1] if len(fields) >= 2 and fields[0].startswith(">sp") else ""
                chunks = []
            elif accession:
                chunks.append(line.strip())
        flush()


def load_string_positive_keys(path: Path) -> set[bytes]:
    keys: set[bytes] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("label") == "1":
                keys.add(sequence_pair_key(row["query"], row["text"]))
    return keys


def select_positive_candidates(
    path: Path,
    eligible: dict[str, dict[str, object]],
    sequences: dict[str, dict[str, object]],
) -> dict[str, dict[str, str]]:
    """Select one deterministic, same-taxon MI:0407 candidate per negative."""
    candidates: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            negative_id = row.get("negative_id", "")
            if negative_id not in eligible:
                continue
            if row.get("positive_evidence_tier") != "strict_direct_interaction":
                continue
            if row.get("same_taxon_anchor_positive") != "1":
                continue
            anchor = row.get("anchor_accession", "")
            negative = row.get("negative_partner_accession", "")
            positive = row.get("positive_partner_accession", "")
            if not positive or positive == negative:
                continue
            if any(acc not in sequences for acc in (anchor, negative, positive)):
                continue
            len_a = int(sequences[anchor]["length"])
            len_c = int(sequences[negative]["length"])
            len_d = int(sequences[positive]["length"])
            row["negative_pair_fits"] = str(int(len_a + len_c + SPECIAL_TOKENS <= MAX_TOKENS))
            row["positive_pair_fits"] = str(int(len_a + len_d + SPECIAL_TOKENS <= MAX_TOKENS))
            candidates.setdefault(negative_id, []).append(row)

    selected: dict[str, dict[str, str]] = {}
    for negative_id, rows in candidates.items():
        # Prefer a fully non-truncated pair, then stronger replicated database
        # support.  Lexical fields make ties reproducible.
        rows.sort(
            key=lambda row: (
                -(int(row["negative_pair_fits"]) * int(row["positive_pair_fits"])),
                -int(row.get("distinct_publications") or 0),
                -int(row.get("evidence_row_count") or 0),
                row.get("orientation_id", ""),
                row.get("positive_partner_accession", ""),
            )
        )
        selected[negative_id] = rows[0]
    return selected


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = read_tsv(args.source)
    if len(source) != 480:
        raise SystemExit(f"expected 480 reviewed rows, found {len(source)}")
    if len({row["pair2_key"] for row in source}) != len(source):
        raise SystemExit("review source contains duplicate pair2_key values")

    mappings = {row["negative_id"]: row for row in read_tsv(args.mapping)}
    sequences = load_sequences(args.sequences)
    needed_accessions = {
        accession
        for raw in source
        for accession in (
            mappings.get(raw["pair2_key"], {}).get("accession_a", ""),
            mappings.get(raw["pair2_key"], {}).get("accession_b", ""),
        )
        if accession
    }
    add_swissprot_sequences(args.swissprot, sequences, needed_accessions)
    string_keys = load_string_positive_keys(args.string_train)
    negatome_keys = {
        accession_pair_key(row["protein_a"], row["protein_b"])
        for row in read_tsv(args.negatome_pairs)
    }

    manifest: list[dict[str, object]] = []
    score_pairs: list[dict[str, str]] = []
    eligible_for_positive: dict[str, dict[str, object]] = {}
    for raw in source:
        mapping = mappings.get(raw["pair2_key"], {})
        acc_a, acc_b = mapping.get("accession_a", ""), mapping.get("accession_b", "")
        seq_a = str(sequences.get(acc_a, {}).get("sequence", ""))
        seq_b = str(sequences.get(acc_b, {}).get("sequence", ""))
        len_a, len_b = len(seq_a), len(seq_b)
        reliable = mapping.get("pair_mapping_status") == "both_in_reported_taxon_mapped"
        sequence_available = bool(seq_a and seq_b)
        fits = sequence_available and len_a + len_b + SPECIAL_TOKENS <= MAX_TOKENS
        overlap_negatome = bool(acc_a and acc_b) and accession_pair_key(acc_a, acc_b) in negatome_keys
        overlap_string = sequence_available and sequence_pair_key(seq_a, seq_b) in string_keys
        score_id = f"PN6:{int(raw['sequence_no']):04d}:negative"
        item: dict[str, object] = {
            **raw,
            "evidence_semantics": evidence_class(raw),
            "accession_a": acc_a,
            "accession_b": acc_b,
            "mapping_status_a": mapping.get("mapping_status_a", ""),
            "mapping_status_b": mapping.get("mapping_status_b", ""),
            "pair_mapping_status": mapping.get("pair_mapping_status", ""),
            "reliable_reported_taxon_mapping": int(reliable),
            "manual_verified_negative": 1,
            "direct_assay_negative": int(
                evidence_class(raw) in {"biophysical_direct", "other_direct_binding"}
            ),
            "sequence_available": int(sequence_available),
            "len_a": len_a or "",
            "len_b": len_b or "",
            "sequence_source_a": sequences.get(acc_a, {}).get("source", ""),
            "sequence_source_b": sequences.get(acc_b, {}).get("source", ""),
            "pair_tokens": (len_a + len_b + SPECIAL_TOKENS) if sequence_available else "",
            "pair_fits_1024": int(fits),
            "human_human": int(raw.get("a_taxid") == "9606" and raw.get("b_taxid") == "9606"),
            "exact_negatome_pair_overlap": int(overlap_negatome),
            "exact_string_positive_overlap": int(overlap_string),
            "negative_score_id": score_id if reliable and sequence_available else "",
        }
        manifest.append(item)
        if reliable and sequence_available:
            score_pairs.append({"directed_pair_id": score_id, "query": seq_a, "text": seq_b})
            if not overlap_negatome:
                eligible_for_positive[raw["pair2_key"]] = item

    selected = select_positive_candidates(args.positive_candidates, eligible_for_positive, sequences)
    paired_manifest: list[dict[str, object]] = []
    paired_pairs: list[dict[str, str]] = []
    by_negative = {str(row["pair2_key"]): row for row in manifest}
    for negative_id in sorted(selected):
        candidate = selected[negative_id]
        base = by_negative[negative_id]
        anchor = candidate["anchor_accession"]
        negative = candidate["negative_partner_accession"]
        positive = candidate["positive_partner_accession"]
        seq_a = str(sequences[anchor]["sequence"])
        seq_c = str(sequences[negative]["sequence"])
        seq_d = str(sequences[positive]["sequence"])
        pair_root = f"PN6P:{int(base['sequence_no']):04d}"
        neg_id, pos_id = f"{pair_root}:negative", f"{pair_root}:positive"
        paired_pairs.extend([
            {"directed_pair_id": neg_id, "query": seq_a, "text": seq_c},
            {"directed_pair_id": pos_id, "query": seq_a, "text": seq_d},
        ])
        paired_manifest.append({
            "pair_root": pair_root,
            "negative_id": negative_id,
            "sequence_no": base["sequence_no"],
            "pmid": base["pmid"],
            "evidence_semantics": base["evidence_semantics"],
            "anchor_accession": anchor,
            "negative_partner_accession": negative,
            "positive_partner_accession": positive,
            "positive_evidence_tier": candidate["positive_evidence_tier"],
            "distinct_positive_publications": candidate.get("distinct_publications", ""),
            "positive_detection_methods": candidate.get("detection_methods", ""),
            "negative_pair_fits_1024": candidate["negative_pair_fits"],
            "positive_pair_fits_1024": candidate["positive_pair_fits"],
            "both_pairs_fit_1024": int(
                candidate["negative_pair_fits"] == "1" and candidate["positive_pair_fits"] == "1"
            ),
            "exact_string_positive_overlap_negative": base["exact_string_positive_overlap"],
            "negative_score_id": neg_id,
            "positive_score_id": pos_id,
        })

    manifest_fields = list(manifest[0])
    write_tsv(args.output_dir / "protneg_assay_manifest_v1.tsv", manifest, manifest_fields)
    core_fields = [
        "sequence_no", "pair2_key", "protein_a", "a_id", "a_taxid", "accession_a",
        "mapping_status_a", "protein_b", "b_id", "b_taxid", "accession_b",
        "mapping_status_b", "assay_normalized", "assay_reported_original",
        "evidence_semantics", "pmid", "pmcid", "article_title",
        "reliable_reported_taxon_mapping", "sequence_available", "len_a", "len_b",
        "pair_tokens", "pair_fits_1024", "human_human",
        "exact_negatome_pair_overlap", "exact_string_positive_overlap", "negative_score_id",
    ]
    write_tsv(args.output_dir / "protneg_assay_core_metadata_v1.tsv", manifest, core_fields)
    write_pairs(args.output_dir / "protneg_assay_negative_pairs_v1.csv", score_pairs)
    paired_fields = list(paired_manifest[0]) if paired_manifest else []
    write_tsv(args.output_dir / "protneg_assay_paired_manifest_v1.tsv", paired_manifest, paired_fields)
    write_pairs(args.output_dir / "protneg_assay_paired_pairs_v1.csv", paired_pairs)

    summary = {
        "source_rows": len(source),
        "source_unique_pairs": len({row["pair2_key"] for row in source}),
        "evidence_semantics_counts": dict(Counter(str(row["evidence_semantics"]) for row in manifest)),
        "manual_verified_negative": len(manifest),
        "direct_assay_negative": sum(int(row["direct_assay_negative"]) for row in manifest),
        "reliable_reported_taxon_mapping": sum(int(row["reliable_reported_taxon_mapping"]) for row in manifest),
        "score_pairs": len(score_pairs),
        "score_pairs_no_truncation": sum(
            int(row["reliable_reported_taxon_mapping"]) * int(row["pair_fits_1024"])
            for row in manifest
        ),
        "exact_negatome_pair_overlaps": sum(int(row["exact_negatome_pair_overlap"]) for row in manifest),
        "exact_string_positive_overlaps_scored": sum(
            int(row["exact_string_positive_overlap"]) * int(bool(row["negative_score_id"]))
            for row in manifest
        ),
        "paired_negative_pairs": len(paired_manifest),
        "paired_both_no_truncation": sum(int(row["both_pairs_fit_1024"]) for row in paired_manifest),
        "positive_rule": (
            "one same-taxon IntAct MI:0407 strict-direct candidate per negative; "
            "prefer both pairs <=1024, then publication/evidence support"
        ),
    }
    (args.output_dir / "build_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
