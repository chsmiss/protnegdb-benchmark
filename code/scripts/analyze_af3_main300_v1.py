#!/usr/bin/env python3
"""Score AF3 partner ranking on the frozen main300 triplets.

Loads every non-empty dimer summary under af3_partner_ranking_v1 (pilot48
outputs are reused, not recomputed). Primary metric is P(D>C) on unique A-C
triplets, ties counted as 0.5, Wilson 95% CI. Sequence-model scores come from
figure2_triplet_master_v1 so AF3 and PLM-interact / D-SCRIPT / Topsy-Turvy
share the same 300 triplets.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esmfold_linker_pilot_lib_v1 import mean, ranking_hit, read_tsv, wilson_interval, write_tsv

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/af3_partner_ranking_v1"
MASTER = ROOT / "data/interim/figure12_master_v1/figure2_triplet_master_v1.tsv"
SEQ_MODELS = (
    ("PLMInteract", "score_AD_PLMInteract", "score_AC_PLMInteract"),
    ("DSCRIPT", "score_AD_DSCRIPT", "score_AC_DSCRIPT"),
    ("TopsyTurvy", "score_AD_TopsyTurvy", "score_AC_TopsyTurvy"),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--links", type=Path, default=BASE / "main300_pair_links_v1.tsv")
    p.add_argument("--manifest", type=Path, default=BASE / "main300_manifest_v1.tsv")
    p.add_argument("--pilot-manifest", type=Path, default=BASE / "pilot48_manifest_v1.tsv")
    p.add_argument("--master", type=Path, default=MASTER)
    p.add_argument("--output-dir", type=Path, default=BASE / "main300_eval_v1")
    return p.parse_args()


def as_float(value: str) -> float:
    if value is None or str(value).strip() in {"", "NA", "nan", "None"}:
        return float("nan")
    return float(value)


def load_af3_scores(base: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for summary in base.glob("dimer_outputs_*/*/*_summary_confidences.json"):
        if not summary.is_file() or summary.stat().st_size <= 0:
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        scores[summary.parent.name] = {
            "iptm": float(payload["iptm"]),
            "ptm": float(payload["ptm"]),
            "ranking_score": float(payload["ranking_score"]),
        }
    return scores


def metric_block(rows: list[dict], hit_key: str, ad_key: str, ac_key: str) -> dict:
    usable = [row for row in rows if not math.isnan(row[hit_key])]
    hits = [row[hit_key] for row in usable]
    p, lo, hi = wilson_interval(sum(hits), len(usable))
    return {
        "n": len(usable),
        "P_D_gt_C": p,
        "wilson95_lo": lo,
        "wilson95_hi": hi,
        "n_strict": sum(1 for hit in hits if hit == 1.0),
        "n_tie": sum(1 for hit in hits if hit == 0.5),
        "mean_AD": mean([row[ad_key] for row in usable]),
        "mean_AC": mean([row[ac_key] for row in usable]),
        "mean_delta": mean([row[ad_key] - row[ac_key] for row in usable]),
    }


def by_stratum(rows: list[dict], key: str, hit_key: str, ad_key: str, ac_key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, ""))].append(row)
    return {label: metric_block(values, hit_key, ad_key, ac_key)
            for label, values in sorted(groups.items(), key=lambda item: item[0])}


def iptm_thresholds(rows: list[dict], cuts: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)) -> list[dict]:
    return [{
        "threshold": cut,
        "n_ad": sum(1 for row in rows if row["af3_iptm_ad"] >= cut),
        "n_ac": sum(1 for row in rows if row["af3_iptm_ac"] >= cut),
    } for cut in cuts]


def main() -> None:
    args = parse_args()
    links = read_tsv(args.links)
    manifest = {row["triplet_id"]: row for row in read_tsv(args.manifest)}
    pilot_ids = {row["triplet_id"] for row in read_tsv(args.pilot_manifest)}
    master = {row["triplet_id"]: row for row in read_tsv(args.master)}
    af3 = load_af3_scores(BASE)
    by_trip: dict[str, dict] = defaultdict(dict)
    for link in links:
        by_trip[link["triplet_id"]][link["role"]] = link

    rows = []; missing = []
    for triplet_id, roles in sorted(by_trip.items()):
        ad, ac = roles.get("ad"), roles.get("ac")
        if not ad or not ac:
            missing.append({"triplet_id": triplet_id, "reason": "missing_role"}); continue
        if ad["job_id"] not in af3 or ac["job_id"] not in af3:
            missing.append({"triplet_id": triplet_id, "reason": "missing_af3"}); continue
        meta = manifest.get(triplet_id, {})
        seq = master.get(triplet_id, {})
        row = {
            "triplet_id": triplet_id,
            "ac_id": ad["ac_id"],
            "anchor": ad["anchor"],
            "D": ad["partner"],
            "C": ac["partner"],
            "in_pilot48": int(triplet_id in pilot_ids),
            "positive_source": meta.get("positive_source", ""),
            "seqid_CD_bin": meta.get("seqid_CD_bin", ""),
            "family_split": meta.get("family_split", ""),
            "complex_size_bin": meta.get("complex_size_bin", ""),
            "af3_iptm_ad": af3[ad["job_id"]]["iptm"],
            "af3_iptm_ac": af3[ac["job_id"]]["iptm"],
            "af3_rank_ad": af3[ad["job_id"]]["ranking_score"],
            "af3_rank_ac": af3[ac["job_id"]]["ranking_score"],
            "af3_ptm_ad": af3[ad["job_id"]]["ptm"],
            "af3_ptm_ac": af3[ac["job_id"]]["ptm"],
        }
        row["hit_af3_iptm"] = ranking_hit(row["af3_iptm_ad"], row["af3_iptm_ac"])
        row["hit_af3_rank"] = ranking_hit(row["af3_rank_ad"], row["af3_rank_ac"])
        row["hit_af3_ptm"] = ranking_hit(row["af3_ptm_ad"], row["af3_ptm_ac"])
        for name, ad_col, ac_col in SEQ_MODELS:
            row[f"seq_{name}_ad"] = as_float(seq.get(ad_col, ""))
            row[f"seq_{name}_ac"] = as_float(seq.get(ac_col, ""))
            row[f"hit_seq_{name}"] = ranking_hit(row[f"seq_{name}_ad"], row[f"seq_{name}_ac"])
        rows.append(row)

    if len({row["ac_id"] for row in rows}) != len(rows):
        raise SystemExit("main300 is not unique A-C")

    nested48 = [row for row in rows if row["in_pilot48"] == 1]
    metrics = {
        "af3_iptm": metric_block(rows, "hit_af3_iptm", "af3_iptm_ad", "af3_iptm_ac"),
        "af3_ranking_score": metric_block(rows, "hit_af3_rank", "af3_rank_ad", "af3_rank_ac"),
        "af3_ptm": metric_block(rows, "hit_af3_ptm", "af3_ptm_ad", "af3_ptm_ac"),
    }
    for name, _, _ in SEQ_MODELS:
        metrics[f"seq_{name}"] = metric_block(rows, f"hit_seq_{name}", f"seq_{name}_ad", f"seq_{name}_ac")

    report = {
        "n_complete": len(rows),
        "n_missing": len(missing),
        "n_pilot48_nested": len(nested48),
        "missing": missing,
        "overall": metrics,
        "nested_pilot48": {
            "af3_iptm": metric_block(nested48, "hit_af3_iptm", "af3_iptm_ad", "af3_iptm_ac"),
            "af3_ranking_score": metric_block(nested48, "hit_af3_rank", "af3_rank_ad", "af3_rank_ac"),
        },
        "af3_iptm_by": {
            "positive_source": by_stratum(rows, "positive_source", "hit_af3_iptm", "af3_iptm_ad", "af3_iptm_ac"),
            "family_split": by_stratum(rows, "family_split", "hit_af3_iptm", "af3_iptm_ad", "af3_iptm_ac"),
            "seqid_CD_bin": by_stratum(rows, "seqid_CD_bin", "hit_af3_iptm", "af3_iptm_ad", "af3_iptm_ac"),
            "complex_size_bin": by_stratum(rows, "complex_size_bin", "hit_af3_iptm", "af3_iptm_ad", "af3_iptm_ac"),
        },
        "iptm_thresholds": iptm_thresholds(rows),
        "protocol": {
            "af3": "3.0.3, 1 seed x 5 diffusion samples, max_template_date=2021-09-30",
            "set": "main300 unique A-C; exact-pair pre-cutoff history audit not done",
            "primary_score": "ipTM",
            "tie": 0.5,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(args.output_dir / "triplet_scores_v1.tsv", rows)
    (args.output_dir / "summary_v1.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_complete": report["n_complete"],
        "n_missing": report["n_missing"],
        "overall": report["overall"],
        "nested_pilot48": report["nested_pilot48"],
        "af3_iptm_by": report["af3_iptm_by"],
        "iptm_thresholds": report["iptm_thresholds"],
    }, indent=2))


if __name__ == "__main__":
    main()
