"""Summarize published PLM-interact scores on Negatome 2.0 manual pairs.

Negatome Manual is literature-supported non-interaction, not assembly
non-contact. This script only reports score distribution and a 0.5 false
positive rate; it does not merge the two negative semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-tsv", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--structure-eval-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "frac_ge_0.5": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "n": len(values),
        "mean": round(sum(values) / len(values), 4),
        "median": round(median, 4),
        "frac_ge_0.5": round(sum(v >= 0.5 for v in values) / len(values), 4),
    }


def main() -> None:
    args = parse_args()
    pairs = list(csv.DictReader(args.pairs_tsv.open(encoding="utf-8", newline=""), delimiter="\t"))
    scores = [float(row["score"]) for row in csv.DictReader(args.scores.open(encoding="utf-8", newline=""))]
    if len(pairs) != len(scores):
        raise SystemExit(f"pair/score mismatch {len(pairs)} vs {len(scores)}")

    all_scores = scores
    stringent = [score for pair, score in zip(pairs, scores) if pair["in_manual_stringent"] == "1"]
    structure = json.loads(args.structure_eval_json.read_text(encoding="utf-8"))
    plm = structure["models"]["plminteract_base"]["by_role"]

    result = {
        "status": "negatome2_manual_plminteract_zero_shot",
        "model": "PLM-interact-650M-humanV12 published",
        "negatome_manual": stats(all_scores),
        "negatome_manual_stringent": stats(stringent),
        "structure_ac_negative": plm["ac_negative"],
        "structure_ad_positive": plm["ad_positive"],
        "structure_random_unlabeled": plm["random_unlabeled"],
        "note": (
            "Negatome Manual is literature non-interaction among mostly mammalian "
            "proteins. Structure A-C is assembly-context noncontact. Do not merge labels."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
