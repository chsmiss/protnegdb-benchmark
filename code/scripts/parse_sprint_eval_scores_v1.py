#!/usr/bin/env python3
"""Map SPRINT ``score label`` output onto four-model score tables.

``predict_interactions`` writes one ``score label`` line per test pair whose
both IDs are in the FASTA, in input order, positives then negatives. SPRINT
scores are undirected; both directed orientations get the same value.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "data/interim/sprint_baseline_v1/eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--result", type=Path, default=None)
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
    return parser.parse_args()


def parse_sprint_score_lines(text: str) -> list[float]:
    scores: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"expected 'score label', got {raw!r}")
        scores.append(float(parts[0]))
    return scores


def read_fasta_names(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                names.add(line[1:].split()[0])
    return names


def read_space_pairs(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(f"expected two protein names, got {line!r}")
            rows.append((parts[0], parts[1]))
    return rows


def align_scores(
    pairs: list[tuple[str, str]],
    names: set[str],
    scores: list[float],
) -> dict[tuple[str, str], float]:
    kept = [pair for pair in pairs if pair[0] in names and pair[1] in names]
    if len(kept) != len(scores):
        raise ValueError(
            f"SPRINT returned {len(scores)} scores for {len(kept)} FASTA-covered pairs "
            f"({len(pairs)} listed)"
        )
    out: dict[tuple[str, str], float] = {}
    for (left, right), score in zip(kept, scores):
        key = (left, right) if left <= right else (right, left)
        out[key] = score
    return out


def write_directed_headerless(path: Path, pairs: list[tuple[str, str]], scores: dict[tuple[str, str], float]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for left, right in pairs:
            key = (left, right) if left <= right else (right, left)
            if key not in scores:
                continue
            score = scores[key]
            handle.write(f"{left}\t{right}\t{score:.10g}\n")
            handle.write(f"{right}\t{left}\t{score:.10g}\n")
            n += 1
    return n


def write_directed_pair_id(path: Path, pairs: list[tuple[str, str]], scores: dict[tuple[str, str], float]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["directed_pair_id", "score"], lineterminator="\n")
        writer.writeheader()
        for left, right in pairs:
            key = (left, right) if left <= right else (right, left)
            if key not in scores:
                continue
            score = scores[key]
            writer.writerow({"directed_pair_id": f"{left}|{right}:forward", "score": f"{score:.10g}"})
            writer.writerow({"directed_pair_id": f"{left}|{right}:reverse", "score": f"{score:.10g}"})
            n += 1
    return n


def main() -> None:
    args = parse_args()
    eval_dir = args.eval_dir
    result_path = args.result or eval_dir / "sprint_result_v1.txt"
    fasta_names = read_fasta_names(eval_dir / "sprint_proteins_v1.fasta")
    pos_pairs = read_space_pairs(eval_dir / "test_positive_v1.txt")
    neg_pairs = read_space_pairs(eval_dir / "test_negative_v1.txt")
    scores_list = parse_sprint_score_lines(result_path.read_text(encoding="utf-8"))
    aligned = align_scores(pos_pairs + neg_pairs, fasta_names, scores_list)

    pilot_dir = args.pilot_dir
    raw = args.pilot_dir / "raw_scores"
    raw.mkdir(parents=True, exist_ok=True)

    def tsv_pairs(path: Path, left: str, right: str) -> list[tuple[str, str]]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as handle:
            return [(row[left], row[right]) for row in csv.DictReader(handle, delimiter="\t")]

    directed = tsv_pairs(pilot_dir / "pilot_directed_pairs_v1.tsv", "protein_a", "protein_b")
    # Deduplicate to unique undirected requested pairs, then emit both directions.
    pilot_undirected = []
    seen = set()
    for left, right in directed:
        key = (left, right) if left <= right else (right, left)
        if key in seen:
            continue
        seen.add(key)
        pilot_undirected.append(key)

    n_pilot = write_directed_headerless(
        raw / "sprint_directed_scores_v1.tsv", pilot_undirected, aligned
    )
    write_directed_pair_id(raw / "sprint_directed_scores_pairid_v1.tsv", pilot_undirected, aligned)

    random_rows = tsv_pairs(
        pilot_dir / "random_partner_v1" / "random_partners_v1.tsv",
        "anchor",
        "random_partner",
    )
    n_random = write_directed_headerless(
        pilot_dir / "random_partner_v1" / "raw_scores" / "sprint_random_scores_v1.tsv",
        random_rows,
        aligned,
    )

    analog_rows = tsv_pairs(
        pilot_dir / "label_concordant_v1" / "label_concordant_analogs_v1.tsv",
        "anchor",
        "e_positive",
    )
    n_analog = write_directed_pair_id(
        pilot_dir / "label_concordant_v1" / "raw_scores" / "sprint_analog_scores_v1.tsv",
        analog_rows,
        aligned,
    )

    structure_pairs = args.structure_dir / "scoring_pairs_v1.tsv"
    n_structure = 0
    structure_out = args.structure_dir / "sprint_scores_v1.tsv"
    if structure_pairs.exists():
        with structure_pairs.open(encoding="utf-8", newline="") as handle, structure_out.open(
            "w", encoding="utf-8"
        ) as out:
            for row in csv.DictReader(handle, delimiter="\t"):
                left, right = row["protein_x"], row["protein_y"]
                key = (left, right) if left <= right else (right, left)
                if key not in aligned:
                    continue
                out.write(f"{left}\t{right}\t{aligned[key]:.10g}\n")
                n_structure += 1

    summary = {
        "n_sprint_scores": len(aligned),
        "n_pilot_undirected_scored": n_pilot,
        "n_random_scored": n_random,
        "n_analog_scored": n_analog,
        "n_structure_scored": n_structure,
        "pilot_scores": str(raw / "sprint_directed_scores_v1.tsv"),
        "random_scores": str(
            pilot_dir / "random_partner_v1" / "raw_scores" / "sprint_random_scores_v1.tsv"
        ),
        "analog_scores": str(
            pilot_dir / "label_concordant_v1" / "raw_scores" / "sprint_analog_scores_v1.tsv"
        ),
        "structure_scores": str(structure_out),
    }
    (eval_dir / "parse_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
