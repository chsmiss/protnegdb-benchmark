#!/usr/bin/env python3
"""Aggregate model scores on structure triplets v1 into an evaluation report.

For each model the report covers:

- score distribution per pair role (ac_negative by stratum/grade,
  ad_positive by similarity bin/positive source, random_unlabeled reference);
- triplet ranking accuracy: fraction of counterfactual triplets with
  score(anchor,D+) > score(anchor,C-), overall and stratified by
  counterfactual grade, assembly-size stratum and D~C similarity bin;
- label-discordant false-positive rate on A-C structure negatives at 0.5.

The report is descriptive model evaluation on provisional structure labels;
it does not by itself freeze benchmark status.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--plminteract-pairs-csv", type=Path, required=True)
    parser.add_argument("--scores", nargs="*", default=[],
                        help="model=path; plminteract:*.csv one-column scores aligned to the pairs csv; "
                             "others: 'protA protB score' rows")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
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


def load_plminteract_scores(pairs_csv: Path, scores_csv: Path) -> dict[str, float]:
    rows = list(csv.DictReader(pairs_csv.open(encoding="utf-8", newline="")))
    scores = [float(r["score"]) for r in csv.DictReader(scores_csv.open(encoding="utf-8", newline=""))]
    if len(scores) != len(rows):
        raise SystemExit(f"plminteract score count {len(scores)} != pair rows {len(rows)}")
    return {row["pair_id"]: score for row, score in zip(rows, scores)}


def load_pair_scores(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for parts in csv.reader(fh, delimiter="\t"):
            if len(parts) < 3 or parts[0] == "protein_x":
                continue
            scores[tuple(sorted((parts[0], parts[1])))] = float(parts[2])
    return scores


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader(args.pairs.open(encoding="utf-8", newline=""), delimiter="\t"))
    links = list(csv.DictReader(args.links.open(encoding="utf-8", newline=""), delimiter="\t"))

    model_scores: dict[str, dict[str, float]] = {}
    for spec in args.scores:
        if "=" not in spec:
            raise SystemExit(f"--scores entries must be model=path, got {spec}")
        model, path = spec.split("=", 1)
        path = Path(path)
        if not path.exists():
            print(f"skip missing scores for {model}: {path}")
            continue
        if model.startswith("plminteract"):
            model_scores[model] = load_plminteract_scores(args.plminteract_pairs_csv, path)
        else:
            by_proteins = load_pair_scores(path)
            model_scores[model] = {
                row["pair_id"]: by_proteins[tuple(sorted((row["protein_x"], row["protein_y"])))]
                for row in pairs
                if tuple(sorted((row["protein_x"], row["protein_y"]))) in by_proteins
            }
    if not model_scores:
        raise SystemExit("no model scores loaded")

    report: dict[str, dict] = {}
    for model, scores in model_scores.items():
        by_role: dict[str, list[float]] = defaultdict(list)
        ac_by_stratum: dict[str, list[float]] = defaultdict(list)
        ac_by_grade: dict[str, list[float]] = defaultdict(list)
        ad_by_sim: dict[str, list[float]] = defaultdict(list)
        ad_by_source: dict[str, list[float]] = defaultdict(list)
        for row in pairs:
            score = scores.get(row["pair_id"])
            if score is None:
                continue
            by_role[row["role"]].append(score)
            if row["role"] == "ac_negative":
                ac_by_stratum[row["assembly_size_stratum"]].append(score)
                ac_by_grade[row["structure_grade_provisional"]].append(score)
            elif row["role"] == "ad_positive":
                ad_by_sim[row["sequence_similarity_bin"]].append(score)
                ad_by_source[row["positive_source"]].append(score)
        report[model] = {
            "pairs_scored": len(scores),
            "by_role": {k: stats(v) for k, v in sorted(by_role.items())},
            "ac_negative_by_stratum": {k: stats(v) for k, v in sorted(ac_by_stratum.items())},
            "ac_negative_by_structure_grade": {k: stats(v) for k, v in sorted(ac_by_grade.items())},
            "ad_positive_by_similarity_bin": {k: stats(v) for k, v in sorted(ad_by_sim.items())},
            "ad_positive_by_source": {k: stats(v) for k, v in sorted(ad_by_source.items())},
        }

    # Triplet ranking accuracy per model.
    for model, scores in model_scores.items():
        buckets: dict[tuple[str, str], Counter] = defaultdict(Counter)
        overall = Counter()
        margins: list[float] = []
        for link in links:
            s_ac = scores.get(link["ac_pair_id"])
            s_ad = scores.get(link["ad_pair_id"])
            if s_ac is None or s_ad is None:
                continue
            win = s_ad > s_ac
            overall["n"] += 1
            overall["win"] += int(win)
            margins.append(s_ad - s_ac)
            for axis in ("counterfactual_grade_provisional", "assembly_size_stratum",
                         "sequence_similarity_bin"):
                key = (axis, link[axis])
                buckets[key]["n"] += 1
                buckets[key]["win"] += int(win)
        stratified = {}
        for (axis, value), cnt in sorted(buckets.items()):
            stratified.setdefault(axis, {})[value] = {
                "n": cnt["n"], "ranking_accuracy": round(cnt["win"] / cnt["n"], 4) if cnt["n"] else None,
            }
        report[model]["triplet_ranking"] = {
            "n": overall["n"],
            "ranking_accuracy": round(overall["win"] / overall["n"], 4) if overall["n"] else None,
            "mean_margin_ad_minus_ac": round(sum(margins) / len(margins), 4) if margins else None,
            "stratified": stratified,
        }

    # Per-triplet score table carries every model side by side.
    wide: dict[str, dict[str, str]] = {}
    for model, scores in model_scores.items():
        for link in links:
            s_ac = scores.get(link["ac_pair_id"])
            s_ad = scores.get(link["ad_pair_id"])
            if s_ac is None or s_ad is None:
                continue
            row = wide.setdefault(link["triplet_id"], {
                "triplet_id": link["triplet_id"], "anchor": link["anchor"],
                "negative_partner_c": link["negative_partner_c"],
                "positive_partner_d": link["positive_partner_d"],
                "counterfactual_grade_provisional": link["counterfactual_grade_provisional"],
                "assembly_size_stratum": link["assembly_size_stratum"],
                "sequence_similarity_bin": link["sequence_similarity_bin"],
            })
            row[f"{model}_ac"] = f"{s_ac:.6f}"
            row[f"{model}_ad"] = f"{s_ad:.6f}"
    model_cols = [f"{m}_{suffix}" for m in model_scores for suffix in ("ac", "ad")]
    fields = ["triplet_id", "anchor", "negative_partner_c", "positive_partner_d",
              "counterfactual_grade_provisional", "assembly_size_stratum",
              "sequence_similarity_bin", *model_cols]
    with (args.output_dir / "triplet_model_scores_v1.tsv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for triplet_id in sorted(wide):
            writer.writerow(wide[triplet_id])

    (args.output_dir / "structure_triplet_model_report_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
