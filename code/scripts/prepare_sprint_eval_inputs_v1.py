#!/usr/bin/env python3
"""Prepare SPRINT inputs for the four-model counterfactual benchmark.

SPRINT needs a single-line FASTA, known-positive training pairs, and test
pairs whose names match FASTA headers. Training positives are the official
D-SCRIPT human train positives (same pair set as D-SCRIPT / Topsy-Turvy),
mapped onto FASTA names by sequence SHA1. Evaluation accessions keep their
UniProt IDs; train-only proteins keep ``seqsha1:...`` keys.

This script does not train and does not touch test_v2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIN_LEN = 20
ALLOWED = set("ARNDCQEGHILKMFPSTWYVBZXUO")
DEFAULT_OUT = ROOT / "data/interim/sprint_baseline_v1/eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=ROOT / "data/interim/counterfactual_1000_triplet_pilot_v1",
    )
    parser.add_argument(
        "--structure-dir",
        type=Path,
        default=ROOT / "data/interim/structure_triplet_scoring_v1",
    )
    parser.add_argument(
        "--train-map",
        type=Path,
        default=ROOT / "data/interim/train_neighbor_map_v0/dscript_human",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def seq_key(sequence: str) -> str:
    return "seqsha1:" + hashlib.sha1(sequence.encode("ascii")).hexdigest()


def sanitize_sequence(raw: str) -> str:
    seq = "".join(raw.split()).upper().replace("*", "X").replace("-", "X")
    return "".join(ch if ch in ALLOWED else "X" for ch in seq)


def read_fasta(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    name = None
    chunks: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            elif name is not None:
                chunks.append(line)
    if name is not None:
        out[name] = "".join(chunks)
    return out


def add_protein(
    proteins: dict[str, str],
    skipped: list[dict[str, str]],
    name: str,
    raw_seq: str,
    source: str,
) -> None:
    if not name or name in proteins:
        return
    seq = sanitize_sequence(raw_seq)
    if len(seq) < MIN_LEN:
        skipped.append({"name": name, "source": source, "reason": "short_sequence", "length": str(len(seq))})
        return
    proteins[name] = seq


def read_plm_csv_sequences(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ident = row.get("directed_pair_id", "")
            if ":" not in ident or "|" not in ident:
                continue
            pair_key, direction = ident.rsplit(":", 1)
            left, right = pair_key.split("|", 1)
            if direction == "forward":
                out.setdefault(left, row["query"])
                out.setdefault(right, row["text"])
            elif direction == "reverse":
                out.setdefault(right, row["query"])
                out.setdefault(left, row["text"])
    return out


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def add_pair(pairs: dict[tuple[str, str], list[str]], left: str, right: str, split: str) -> None:
    if not left or not right or left == right:
        return
    key = (left, right) if left <= right else (right, left)
    splits = pairs.setdefault(key, [])
    if split not in splits:
        splits.append(split)


def write_pair_file(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for left, right in rows:
            handle.write(f"{left} {right}\n")


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    proteins: dict[str, str] = {}
    skipped: list[dict[str, str]] = []

    pilot_fasta = read_fasta(args.pilot_dir / "pilot_proteins_v1.fasta")
    for name, seq in pilot_fasta.items():
        add_protein(proteins, skipped, name, seq, "pilot_fasta")

    structure_fasta = args.structure_dir / "dscript_proteins_v1.fasta"
    if structure_fasta.exists():
        for name, seq in read_fasta(structure_fasta).items():
            add_protein(proteins, skipped, name, seq, "structure_fasta")

    extra_seq_files = [
        args.pilot_dir / "random_partner_v1" / "random_partners_plminteract_pairs.csv",
        args.pilot_dir / "label_concordant_v1" / "analog_plminteract_pairs.csv",
    ]
    for path in extra_seq_files:
        for name, seq in read_plm_csv_sequences(path).items():
            add_protein(proteins, skipped, name, seq, path.name)

    sha_to_name: dict[str, str] = {}
    for name, seq in proteins.items():
        sha_to_name.setdefault(seq_key(seq), name)

    train_proteins_path = args.train_map / "proteins.tsv"
    n_train_seq_added = 0
    with train_proteins_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw = row.get("sequence") or ""
            if not raw:
                continue
            seq = sanitize_sequence(raw)
            original_key = row["protein_key"]
            hashed = seq_key(seq)
            if hashed in sha_to_name:
                sha_to_name.setdefault(original_key, sha_to_name[hashed])
                continue
            before = len(proteins)
            add_protein(proteins, skipped, original_key, raw, "dscript_human_train")
            if len(proteins) > before:
                sha_to_name[hashed] = original_key
                sha_to_name[original_key] = original_key
                n_train_seq_added += 1

    train_pairs: list[tuple[str, str]] = []
    n_train_pos = 0
    n_train_mapped = 0
    with (args.train_map / "pairs.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["label"] != "1" or row["split"] != "train":
                continue
            n_train_pos += 1
            name_a = sha_to_name.get(row["protein_key_a"])
            name_b = sha_to_name.get(row["protein_key_b"])
            if name_a and name_b and name_a != name_b:
                train_pairs.append((name_a, name_b))
                n_train_mapped += 1

    test_pairs: dict[tuple[str, str], list[str]] = {}
    triplets = read_tsv(args.pilot_dir / "pilot_triplets_v1.tsv")
    for row in triplets:
        for field, split in (("ad_pair_key", "pilot_ad"), ("ac_pair_key", "pilot_ac")):
            left, right = row[field].split("|")
            add_pair(test_pairs, left, right, split)

    random_path = args.pilot_dir / "random_partner_v1" / "random_partners_v1.tsv"
    if random_path.exists():
        for row in read_tsv(random_path):
            add_pair(test_pairs, row["anchor"], row["random_partner"], "pilot_ar")

    analog_path = args.pilot_dir / "label_concordant_v1" / "label_concordant_analogs_v1.tsv"
    if analog_path.exists():
        for row in read_tsv(analog_path):
            add_pair(test_pairs, row["anchor"], row["e_positive"], "pilot_ae")

    scoring_pairs = args.structure_dir / "scoring_pairs_v1.tsv"
    n_structure = 0
    if scoring_pairs.exists():
        for row in read_tsv(scoring_pairs):
            add_pair(test_pairs, row["protein_x"], row["protein_y"], f"structure_{row['role']}")
            n_structure += 1

    fasta_names = set(proteins)
    kept_test = [(a, b) for (a, b) in sorted(test_pairs) if a in fasta_names and b in fasta_names]
    missing_test = [
        {"protein_a": a, "protein_b": b, "splits": ",".join(splits)}
        for (a, b), splits in sorted(test_pairs.items())
        if a not in fasta_names or b not in fasta_names
    ]

    fasta_path = out / "sprint_proteins_v1.fasta"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for name in sorted(proteins):
            handle.write(f">{name}\n{proteins[name]}\n")

    write_pair_file(out / "train_positive_v1.txt", train_pairs)
    write_pair_file(out / "test_positive_v1.txt", kept_test)
    (out / "test_negative_v1.txt").write_text("", encoding="utf-8")

    index_path = out / "test_pair_index_v1.tsv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_a", "protein_b", "splits"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for left, right in kept_test:
            writer.writerow(
                {
                    "protein_a": left,
                    "protein_b": right,
                    "splits": ",".join(test_pairs[(left, right)]),
                }
            )

    if skipped:
        with (out / "skipped_proteins_v1.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["name", "source", "reason", "length"],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(skipped)

    summary = {
        "training_source": "dscript_human official positives",
        "n_proteins": len(proteins),
        "n_train_only_proteins_added": n_train_seq_added,
        "n_train_positives_source": n_train_pos,
        "n_train_positives_mapped": n_train_mapped,
        "n_test_pairs_requested": len(test_pairs),
        "n_test_pairs_kept": len(kept_test),
        "n_test_pairs_missing": len(missing_test),
        "n_structure_scoring_rows": n_structure,
        "n_skipped_proteins": len(skipped),
        "min_sequence_length": MIN_LEN,
        "fasta": str(fasta_path),
        "train_positive": str(out / "train_positive_v1.txt"),
        "test_positive": str(out / "test_positive_v1.txt"),
        "test_negative": str(out / "test_negative_v1.txt"),
        "pair_index": str(index_path),
        "did_not_touch_test_v2": True,
        "did_not_train": True,
    }
    (out / "prepare_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if missing_test:
        (out / "missing_test_pairs_v1.json").write_text(
            json.dumps(missing_test, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
