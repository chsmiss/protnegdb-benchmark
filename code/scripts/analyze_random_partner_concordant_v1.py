#!/usr/bin/env python3
"""Compare s(A,D+)/s(A,C−)/s(A,R−) and A-D+/A-E+ label concordance.

Uses the 1000-triplet pilot scores for A-D and A-C, same-anchor random-partner
scores for A-R, and analog scores (falling back to pilot) for A-E. The
biological anchor is always ``AC ∩ AD``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT_DIR = ROOT / "data/interim/counterfactual_1000_triplet_pilot_v1"
ROUTES = {
    "protnegdb_literature_noninteraction": "literature",
    "structure_assembly_context": "structure",
}
MODEL_LABELS = {
    "plminteract_base": "PLM-interact",
    "mint_bernett": "MINT",
    "dscript": "D-SCRIPT",
    "topsy_turvy": "Topsy-Turvy",
    "sprint": "SPRINT",
}
DATASET_LABELS = {
    "literature": "Literature",
    "literature_conflict_free": "Literature conflict-free",
    "structure": "Structure",
}
DATASET_ORDER = ["literature", "literature_conflict_free", "structure"]
DEFAULT_CONFLICTS = (
    ROOT / "data/interim/counterfactual_conflict_audit_v1/undirected_conflict_pairs_v1.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--score", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--random-score", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--analog-score", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PILOT_DIR / "random_concordant_audit_v1",
    )
    parser.add_argument("--conflicts", type=Path, default=DEFAULT_CONFLICTS)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_score_spec(spec: str) -> tuple[str, Path]:
    name, raw = spec.split("=", 1)
    return name, Path(raw)


def pair_members(pair_key: str) -> set[str]:
    left, right = pair_key.split("|")
    return {left, right}


def triplet_anchor(row: dict[str, str]) -> str:
    shared = pair_members(row["ac_pair_key"]) & pair_members(row["ad_pair_key"])
    if len(shared) != 1:
        raise ValueError(f"no unique anchor in {row.get('unified_triplet_id', row)}")
    return next(iter(shared))


def sorted_pair_key(left: str, right: str) -> str:
    return f"{left}|{right}" if left <= right else f"{right}|{left}"


def load_directed_scores(path: Path, directed_by_protein: dict[tuple[str, str], str]):
    scores: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        handle.seek(0)
        if "directed_pair_id" in sample:
            delimiter = "\t" if "\t" in sample else ","
            for row in csv.DictReader(handle, delimiter=delimiter):
                value = row.get("score", row.get("score_mean", ""))
                if value != "":
                    scores[row["directed_pair_id"]] = float(value)
        else:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                ident = directed_by_protein.get((parts[0], parts[1]))
                if ident is not None:
                    scores[ident] = float(parts[2])
    return scores


def reciprocal_from_directions(directions: dict[str, dict[str, str]], scores: dict[str, float]):
    out: dict[str, float] = {}
    for key, dirs in directions.items():
        s_f = scores.get(dirs["forward"])
        s_r = scores.get(dirs["reverse"])
        if s_f is not None and s_r is not None:
            out[key] = (s_f + s_r) / 2.0
    return out


def partner_directions(rows: list[dict[str, str]], left_field: str, right_field: str):
    directions: dict[str, dict[str, str]] = {}
    by_protein: dict[tuple[str, str], str] = {}
    for row in rows:
        left, right = row[left_field], row[right_field]
        key = f"{left}|{right}"
        directions[key] = {
            "forward": f"{left}|{right}:forward",
            "reverse": f"{left}|{right}:reverse",
        }
        by_protein[(left, right)] = f"{left}|{right}:forward"
        by_protein[(right, left)] = f"{left}|{right}:reverse"
    return directions, by_protein


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def ranking_rate(deltas: list[float]) -> float:
    if not deltas:
        return float("nan")
    return sum(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in deltas) / len(deltas)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


def summarize_three_way(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    s_ad = [r["s_ad"] for r in rows]
    s_ac = [r["s_ac"] for r in rows]
    s_ar = [r["s_ar"] for r in rows]
    d_minus_c = [a - c for a, c in zip(s_ad, s_ac)]
    d_minus_r = [a - r for a, r in zip(s_ad, s_ar)]
    c_minus_r = [c - r for c, r in zip(s_ac, s_ar)]
    return {
        "n": len(rows),
        "n_anchors": len({r["anchor"] for r in rows}),
        "mean_s_ad": mean(s_ad),
        "mean_s_ac": mean(s_ac),
        "mean_s_ar": mean(s_ar),
        "mean_d_minus_c": mean(d_minus_c),
        "mean_d_minus_r": mean(d_minus_r),
        "mean_c_minus_r": mean(c_minus_r),
        "p_d_gt_c": ranking_rate(d_minus_c),
        "p_d_gt_r": ranking_rate(d_minus_r),
        "p_c_gt_r": ranking_rate(c_minus_r),
        "p_d_gt_c_and_d_gt_r": sum(
            1 for a, c, r in zip(s_ad, s_ac, s_ar) if a > c and a > r
        ) / len(rows),
    }


def summarize_concordant(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    s_ad = [r["s_ad"] for r in rows]
    s_ae = [r["s_ae"] for r in rows]
    abs_diff = [abs(a - e) for a, e in zip(s_ad, s_ae)]
    out = {
        "n": len(rows),
        "n_anchors": len({r["anchor"] for r in rows}),
        "mean_s_ad": mean(s_ad),
        "mean_s_ae": mean(s_ae),
        "mean_abs_d_minus_e": mean(abs_diff),
        "spearman_ad_ae": spearman(s_ad, s_ae),
        "pearson_ad_ae": pearson(s_ad, s_ae),
        "p_e_gt_d": ranking_rate([e - d for d, e in zip(s_ad, s_ae)]),
    }
    with_c = [r for r in rows if r.get("s_ac") is not None]
    if with_c:
        closer = [
            abs(r["s_ad"] - r["s_ae"]) < abs(r["s_ad"] - r["s_ac"])
            for r in with_c
        ]
        out["n_with_negative"] = len(with_c)
        out["p_ae_closer_to_ad_than_ac"] = sum(closer) / len(closer)
    return out


def load_conflict_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["pair_key"] for row in read_tsv(path)}


def triplet_uses_dual_label(trip: dict[str, str], dual_keys: set[str]) -> bool:
    return trip["ac_pair_key"] in dual_keys or trip["ad_pair_key"] in dual_keys


def load_clusters(path: Path) -> dict[str, str]:
    clusters = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                clusters[parts[1]] = parts[0]
    return clusters


def anchor_block_bootstrap_rate(rows: list[dict], field: str, replicates: int, seed: int) -> dict:
    """Resample unique anchors with replacement; all triplets of an A travel together.

    Ties in ranking_rate are counted as 0.5. This is not a row-iid bootstrap.
    """
    import numpy as np

    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["anchor"]].append(row[field])
    anchors = sorted(groups)
    if not anchors:
        return {"n_blocks": 0, "ci_low": float("nan"), "ci_high": float("nan")}
    ranking_fields = {"d_minus_c", "d_minus_r", "c_minus_r"}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(replicates):
        chosen = rng.choice(anchors, size=len(anchors), replace=True)
        vals = [v for a in chosen for v in groups[a]]
        boots.append(ranking_rate(vals) if field in ranking_fields else mean(vals))
    low, high = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return {"n_blocks": len(anchors), "ci_low": low, "ci_high": high}


def attach_three_way_ci(summary: dict, rows: list[dict], replicates: int, seed: int) -> dict:
    if not rows:
        return summary
    for row in rows:
        row["d_minus_c"] = row["s_ad"] - row["s_ac"]
        row["d_minus_r"] = row["s_ad"] - row["s_ar"]
        row["c_minus_r"] = row["s_ac"] - row["s_ar"]
    summary["p_d_gt_c_ci"] = anchor_block_bootstrap_rate(rows, "d_minus_c", replicates, seed)
    summary["p_d_gt_r_ci"] = anchor_block_bootstrap_rate(rows, "d_minus_r", replicates, seed)
    summary["p_c_gt_r_ci"] = anchor_block_bootstrap_rate(rows, "c_minus_r", replicates, seed)
    return summary


def build_figure(model: str, rows: list[dict], output_dir: Path) -> str | None:
    if not rows:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    s_ad = np.asarray([r["s_ad"] for r in rows])
    s_ac = np.asarray([r["s_ac"] for r in rows])
    s_ar = np.asarray([r["s_ar"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    lo = float(min(s_ad.min(), s_ac.min(), s_ar.min()))
    hi = float(max(s_ad.max(), s_ac.max(), s_ar.max()))
    bins = np.linspace(lo, hi, 40)
    axes[0].hist(s_ad, bins=bins, alpha=0.5, label="A-D+", density=True)
    axes[0].hist(s_ac, bins=bins, alpha=0.5, label="A-C−", density=True)
    axes[0].hist(s_ar, bins=bins, alpha=0.5, label="A-R−", density=True)
    axes[0].set_title(f"{model}: score histogram")
    axes[0].set_xlabel("pair score")
    axes[0].legend()
    for values, label in [(s_ad, "A-D+"), (s_ac, "A-C−"), (s_ar, "A-R−")]:
        ordered = np.sort(values)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)
    axes[1].set_title(f"{model}: ECDF")
    axes[1].set_xlabel("pair score")
    axes[1].legend()
    fig.tight_layout()
    png = output_dir / f"{model}_three_way.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return str(png)


def fmt_prob(value: float, ci: dict | None) -> str:
    if ci is None:
        return f"{value:.3f}"
    return f"{value:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def stratified_table_rows(results: dict) -> list[dict]:
    rows = []
    for model, payload in results.items():
        three_way = payload.get("three_way", {})
        for dataset in DATASET_ORDER:
            summary = three_way.get(dataset)
            if not summary or summary.get("n", 0) == 0:
                continue
            rows.append({
                "model": MODEL_LABELS.get(model, model),
                "model_key": model,
                "dataset": DATASET_LABELS[dataset],
                "dataset_key": dataset,
                "n_triplets": summary["n"],
                "n_anchors": summary["n_anchors"],
                "p_d_gt_c": summary["p_d_gt_c"],
                "p_d_gt_c_ci_low": summary["p_d_gt_c_ci"]["ci_low"],
                "p_d_gt_c_ci_high": summary["p_d_gt_c_ci"]["ci_high"],
                "p_d_gt_r": summary["p_d_gt_r"],
                "p_d_gt_r_ci_low": summary["p_d_gt_r_ci"]["ci_low"],
                "p_d_gt_r_ci_high": summary["p_d_gt_r_ci"]["ci_high"],
                "p_c_gt_r": summary["p_c_gt_r"],
                "p_c_gt_r_ci_low": summary["p_c_gt_r_ci"]["ci_low"],
                "p_c_gt_r_ci_high": summary["p_c_gt_r_ci"]["ci_high"],
                "n_blocks": summary["p_d_gt_c_ci"]["n_blocks"],
            })
    return rows


def build_stratified_markdown(table_rows: list[dict]) -> str:
    lines = [
        "# Stratified three-way ranking (anchor-block bootstrap)",
        "",
        "Datasets: literature assay negatives, the conflict-free literature "
        "core (drop a literature triplet if undirected A-C or A-D is labelled "
        "both positive and negative), and structure assembly-context "
        "noncontacts. P(D>C), P(D>R) and P(C>R) use reciprocal pair scores; "
        "ties count as 0.5. 95% CIs resample unique anchors with replacement "
        "(1,000 replicates, seed 20260814); all triplets of the same A travel "
        "together.",
        "",
        "| Model | Dataset | n (triplets / anchors) | D>C | D>R | C>R |",
        "|---|---|---|---|---|---|",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['model']} | {row['dataset']} | "
            f"{row['n_triplets']} / {row['n_anchors']} | "
            f"{fmt_prob(row['p_d_gt_c'], {'ci_low': row['p_d_gt_c_ci_low'], 'ci_high': row['p_d_gt_c_ci_high']})} | "
            f"{fmt_prob(row['p_d_gt_r'], {'ci_low': row['p_d_gt_r_ci_low'], 'ci_high': row['p_d_gt_r_ci_high']})} | "
            f"{fmt_prob(row['p_c_gt_r'], {'ci_low': row['p_c_gt_r_ci_low'], 'ci_high': row['p_c_gt_r_ci_high']})} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_stratified_tsv(path: Path, table_rows: list[dict]) -> None:
    fields = [
        "model", "dataset", "n_triplets", "n_anchors", "n_blocks",
        "p_d_gt_c", "p_d_gt_c_ci_low", "p_d_gt_c_ci_high",
        "p_d_gt_r", "p_d_gt_r_ci_low", "p_d_gt_r_ci_high",
        "p_c_gt_r", "p_c_gt_r_ci_low", "p_c_gt_r_ci_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(table_rows)


def build_markdown(results: dict) -> str:
    lines = ["# same-anchor random partner and label-concordant analog audit", ""]
    for model, payload in results.items():
        lines.append(f"## {model}")
        lines.append("")
        for route, summary in payload.get("three_way", {}).items():
            lines.append(f"### three-way {route}")
            if summary.get("n", 0) == 0:
                lines.append("- no complete triplets")
                lines.append("")
                continue
            lines.append(
                f"- n={summary['n']} triplets / {summary['n_anchors']} anchors; "
                f"mean s(A,D+)={summary['mean_s_ad']:.4f}, "
                f"s(A,C−)={summary['mean_s_ac']:.4f}, "
                f"s(A,R−)={summary['mean_s_ar']:.4f}"
            )
            lines.append(
                f"- P(D>C)={summary['p_d_gt_c']:.3f}; "
                f"P(D>R)={summary['p_d_gt_r']:.3f}; "
                f"P(C>R)={summary['p_c_gt_r']:.3f}; "
                f"P(D>C and D>R)={summary['p_d_gt_c_and_d_gt_r']:.3f}"
            )
            lines.append("")
        conc = payload.get("concordant", {})
        lines.append("### A-D+ vs A-E+ concordance")
        if conc.get("n", 0) == 0:
            lines.append("- no complete analog rows")
        else:
            lines.append(
                f"- n={conc['n']} analog triplets / {conc['n_anchors']} anchors; "
                f"mean |s(A,D)−s(A,E)|={conc['mean_abs_d_minus_e']:.4f}; "
                f"Spearman={conc['spearman_ad_ae']:.3f}; Pearson={conc['pearson_ad_ae']:.3f}"
            )
            if "p_ae_closer_to_ad_than_ac" in conc:
                lines.append(
                    f"- P(|D−E| < |D−C|)={conc['p_ae_closer_to_ad_than_ac']:.3f} "
                    f"(n={conc['n_with_negative']})"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if not args.score:
        raise SystemExit("supply at least one --score MODEL=PATH")
    triplets = read_tsv(args.pilot_dir / "pilot_triplets_v1.tsv")
    directed = read_tsv(args.pilot_dir / "pilot_directed_pairs_v1.tsv")
    random_rows = read_tsv(args.pilot_dir / "random_partner_v1" / "random_partners_v1.tsv")
    analog_path = args.pilot_dir / "label_concordant_v1" / "label_concordant_analogs_v1.tsv"
    analog_rows = read_tsv(analog_path) if analog_path.exists() else []
    clusters = load_clusters(args.pilot_dir / "clusters_pilot_v1_cluster.tsv")
    dual_keys = load_conflict_keys(args.conflicts)

    directions: dict[str, dict[str, str]] = defaultdict(dict)
    directed_by_protein: dict[tuple[str, str], str] = {}
    for row in directed:
        directions[row["pair_key"]][row["direction"]] = row["directed_pair_id"]
        directed_by_protein[(row["protein_a"], row["protein_b"])] = row["directed_pair_id"]

    random_of = {row["anchor"]: row["random_partner"] for row in random_rows}
    random_dirs, random_by_protein = partner_directions(random_rows, "anchor", "random_partner")
    analog_dirs, analog_by_protein = partner_directions(analog_rows, "anchor", "e_positive")

    random_specs = dict(parse_score_spec(spec) for spec in args.random_score)
    analog_specs = dict(parse_score_spec(spec) for spec in args.analog_score)

    ac_of_anchor: dict[str, list[str]] = defaultdict(list)
    for trip in triplets:
        ac_of_anchor[triplet_anchor(trip)].append(trip["ac_pair_key"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.score:
        model, path = parse_score_spec(spec)
        scores = load_directed_scores(path, directed_by_protein)
        pair_scores = reciprocal_from_directions(directions, scores)

        random_pair_scores: dict[str, float] = {}
        if model in random_specs:
            random_scores = load_directed_scores(random_specs[model], random_by_protein)
            random_pair_scores = reciprocal_from_directions(random_dirs, random_scores)

        analog_pair_scores: dict[str, float] = {}
        if model in analog_specs:
            analog_scores = load_directed_scores(analog_specs[model], analog_by_protein)
            analog_pair_scores = reciprocal_from_directions(analog_dirs, analog_scores)

        three_rows: dict[str, list[dict]] = defaultdict(list)
        for trip in triplets:
            a = triplet_anchor(trip)
            r = random_of.get(a)
            if r is None:
                continue
            s_ad = pair_scores.get(trip["ad_pair_key"])
            s_ac = pair_scores.get(trip["ac_pair_key"])
            s_ar = random_pair_scores.get(f"{a}|{r}")
            if s_ad is None or s_ac is None or s_ar is None:
                continue
            row = {
                "unified_triplet_id": trip["unified_triplet_id"],
                "anchor": a,
                "cluster": clusters.get(a, a),
                "route": trip["evidence_route"],
                "s_ad": s_ad,
                "s_ac": s_ac,
                "s_ar": s_ar,
            }
            three_rows[trip["evidence_route"]].append(row)
            three_rows["all"].append(row)
            if (
                trip["evidence_route"] == "protnegdb_literature_noninteraction"
                and not triplet_uses_dual_label(trip, dual_keys)
            ):
                three_rows["literature_conflict_free"].append(row)

        three_way = {}
        for route_key, rows in three_rows.items():
            label = ROUTES.get(route_key, route_key)
            summary = summarize_three_way(rows)
            three_way[label] = attach_three_way_ci(summary, rows, args.replicates, args.seed)

        conc_rows = []
        for analog in analog_rows:
            a, d, e = analog["anchor"], analog["d_positive"], analog["e_positive"]
            # Prefer the pilot score table for both A-D and A-E so concordance
            # is not mixed across runtimes (MINT analog was scored in esmc5090).
            s_ad = pair_scores.get(sorted_pair_key(a, d))
            s_ae = pair_scores.get(sorted_pair_key(a, e))
            if s_ae is None:
                s_ae = analog_pair_scores.get(f"{a}|{e}")
            if s_ad is None or s_ae is None:
                continue
            s_ac = None
            for ac_key in ac_of_anchor.get(a, []):
                if ac_key in pair_scores:
                    s_ac = pair_scores[ac_key]
                    break
            conc_rows.append(
                {
                    "anchor": a,
                    "s_ad": s_ad,
                    "s_ae": s_ae,
                    "s_ac": s_ac,
                }
            )

        results[model] = {
            "three_way": three_way,
            "concordant": summarize_concordant(conc_rows),
            "figure": build_figure(model, three_rows.get("all", []), args.output_dir),
        }

    (args.output_dir / "random_concordant_audit_v1.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    md = build_markdown(results)
    table_rows = stratified_table_rows(results)
    table_md = build_stratified_markdown(table_rows)
    write_stratified_tsv(args.output_dir / "stratified_three_way_v1.tsv", table_rows)
    (args.output_dir / "stratified_three_way_v1.md").write_text(table_md, encoding="utf-8")
    (args.output_dir / "stratified_three_way_v1.json").write_text(
        json.dumps(table_rows, indent=2), encoding="utf-8"
    )
    (args.output_dir / "random_concordant_audit_v1.md").write_text(
        table_md + md, encoding="utf-8"
    )
    print(table_md)
    print(md)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
