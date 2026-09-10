#!/usr/bin/env python3
"""Pre-registered Figure 3 metrics for the family-transfer split.

Primary: triplet P(s(A,D)>s(A,C)) with ties 0.5, cluster bootstrap by anchor
family. Local fit is reported but is not held-out. Sealed family-unseen is
never used for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--partner-hits", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scores", action="append", required=True,
                        help="name=path to score CSV with pair_id,score")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260816)
    return parser.parse_args()


def load_scores(spec: str) -> tuple[str, dict[str, float]]:
    name, path = spec.split("=", 1)
    rows = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("pair_id") or row.get("directed_pair_id")
            if not key:
                continue
            rows[key] = float(row["score"])
    return name, rows


def load_assignments(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_partner_s(path: Path) -> dict[str, float]:
    best: dict[str, float] = defaultdict(float)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            query, _target, pident = fields[0], fields[1], float(fields[2])
            if pident > best[query]:
                best[query] = pident
    return dict(best)


def ranking_row(trip: dict[str, str], scores: dict[str, float]) -> dict[str, float] | None:
    s_ac = scores.get(f"{trip['triplet_id']}|AC")
    s_ad = scores.get(f"{trip['triplet_id']}|AD")
    if s_ac is None or s_ad is None:
        return None
    if s_ad > s_ac:
        p = 1.0
    elif s_ad < s_ac:
        p = 0.0
    else:
        p = 0.5
    return {"p": p, "m": s_ad - s_ac, "s_ac": s_ac, "s_ad": s_ad}


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {"n": 0, "p": math.nan, "m": math.nan}
    return {
        "n": len(rows),
        "p": sum(r["p"] for r in rows) / len(rows),
        "m": sum(r["m"] for r in rows) / len(rows),
    }


def cluster_bootstrap(
    trips: list[dict[str, str]],
    scored: list[dict[str, float]],
    n_boot: int,
    seed: int,
) -> dict[str, list[float]]:
    by_fam: dict[str, list[dict[str, float]]] = defaultdict(list)
    for trip, row in zip(trips, scored):
        by_fam[trip["fam_a"]].append(row)
    fams = sorted(by_fam)
    rng = random.Random(seed)
    ps, ms = [], []
    for _ in range(n_boot):
        draw = []
        for fam in (rng.choice(fams) for _ in fams):
            draw.extend(by_fam[fam])
        stats = summarize(draw)
        ps.append(stats["p"])
        ms.append(stats["m"])
    ps.sort()
    ms.sort()
    lo = int(0.025 * (n_boot - 1))
    hi = int(0.975 * (n_boot - 1))
    return {"p_lo": ps[lo], "p_hi": ps[hi], "m_lo": ms[lo], "m_hi": ms[hi]}


def bin_partner(value: float) -> str:
    if value < 20:
        return "0-20"
    if value < 40:
        return "20-40"
    if value < 50:
        return "40-50"
    if value < 80:
        return "50-80"
    return "80-100"


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    assignments = load_assignments(args.assignments)
    partner_s = load_partner_s(args.partner_hits)
    loaded = dict(load_scores(spec) for spec in args.scores)
    arms: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for name, scores in loaded.items():
        if "_seed" in name:
            grouped[name.split("_seed")[0]].append(scores)
        else:
            arms[name] = scores
    for prefix, maps in grouped.items():
        keys = set.intersection(*(set(m) for m in maps))
        arms[prefix] = {
            key: sum(m[key] for m in maps) / len(maps) for key in keys
        }
    bins = {
        "local_fit": [r for r in assignments if r["split"] == "train_eligible"],
        "family_seen": [r for r in assignments if r["split"] == "protein_unseen_family_seen"],
        "family_unseen": [r for r in assignments if r["split"] == "family_unseen"],
    }
    dominant = protocol.get("dominant_family_seen_cluster", "")
    report: dict[str, object] = {"protocol": protocol, "arms": {}}
    for arm, scores in arms.items():
        arm_out: dict[str, object] = {}
        for bname, trips in bins.items():
            scored_rows = []
            kept = []
            for trip in trips:
                row = ranking_row(trip, scores)
                if row is None:
                    continue
                s_c = partner_s.get(trip["c"], 0.0)
                s_d = partner_s.get(trip["d"], 0.0)
                row["s_partner"] = min(s_c, s_d)
                row["s_anchor"] = partner_s.get(trip["anchor"], 0.0)
                scored_rows.append(row)
                kept.append(trip)
            stats = summarize(scored_rows)
            boot = cluster_bootstrap(kept, scored_rows, args.n_boot, args.seed) if kept else {}
            arm_out[bname] = {**stats, **boot, "n_scored": len(scored_rows)}
            if bname == "family_seen" and dominant:
                sub_trips = [t for t in kept
                             if dominant not in {t["fam_a"], t["fam_c"], t["fam_d"]}]
                sub = [r for t, r in zip(kept, scored_rows)
                       if dominant not in {t["fam_a"], t["fam_c"], t["fam_d"]}]
                sub_boot = cluster_bootstrap(
                    sub_trips, sub, args.n_boot, args.seed) if sub else {}
                arm_out["family_seen_leave_dominant_out"] = {
                    **summarize(sub),
                    **sub_boot,
                    "n": len(sub),
                    "dropped_cluster": dominant,
                }
            if bname != "local_fit":
                repl_trips = [t for t in kept if t["replicated_noncontact"] == "1"]
                repl = [r for t, r in zip(kept, scored_rows)
                        if t["replicated_noncontact"] == "1"]
                repl_boot = cluster_bootstrap(
                    repl_trips, repl, args.n_boot, args.seed) if repl else {}
                arm_out[f"{bname}_replicated"] = {
                    **summarize(repl), **repl_boot, "n_scored": len(repl),
                }
            if bname in ("family_seen", "family_unseen"):
                by_bin: dict[str, list[dict[str, float]]] = defaultdict(list)
                for row in scored_rows:
                    by_bin[bin_partner(row["s_partner"])].append(row)
                arm_out[f"{bname}_by_s_partner"] = {
                    key: summarize(val) for key, val in sorted(by_bin.items())
                }
        report["arms"][arm] = arm_out

    if "original" in arms and "structure" in arms:
        deltas = {}
        for bname in ("family_seen", "family_unseen", "local_fit"):
            a = report["arms"]["structure"][bname]  # type: ignore[index]
            b = report["arms"]["original"][bname]  # type: ignore[index]
            deltas[bname] = {
                "delta_p": a["p"] - b["p"],
                "delta_m": a["m"] - b["m"],
            }
        report["structure_minus_original"] = deltas
    if "random" in arms and "structure" in arms:
        deltas = {}
        for bname in ("family_seen", "family_unseen"):
            a = report["arms"]["structure"][bname]  # type: ignore[index]
            b = report["arms"]["random"][bname]  # type: ignore[index]
            deltas[bname] = {
                "delta_p": a["p"] - b["p"],
                "delta_m": a["m"] - b["m"],
            }
        report["structure_minus_random"] = deltas
    if "original" in arms and "structure" in arms:
        partner_deltas = {}
        for bname in ("family_seen", "family_unseen"):
            s_bins = report["arms"]["structure"].get(f"{bname}_by_s_partner", {})  # type: ignore[union-attr]
            o_bins = report["arms"]["original"].get(f"{bname}_by_s_partner", {})  # type: ignore[union-attr]
            partner_deltas[bname] = {
                key: {
                    "n": s_bins[key]["n"],
                    "delta_p": s_bins[key]["p"] - o_bins[key]["p"],
                    "delta_m": s_bins[key]["m"] - o_bins[key]["m"],
                }
                for key in s_bins if key in o_bins
            }
        report["delta_m_vs_s_partner"] = partner_deltas
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "protocol"}, indent=2))


if __name__ == "__main__":
    main()
