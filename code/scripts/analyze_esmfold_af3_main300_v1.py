#!/usr/bin/env python3
"""Compare ESMFold linker-dimer scores with AF3 on the main300 triplets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_af3_main300_v1 import load_af3_scores, metric_block
from esmfold_linker_pilot_lib_v1 import ranking_hit, read_tsv, write_tsv

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/af3_partner_ranking_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--links", type=Path, default=BASE / "main300_pair_links_v1.tsv")
    p.add_argument("--manifest", type=Path, default=BASE / "main300_manifest_v1.tsv")
    p.add_argument("--pilot-manifest", type=Path, default=BASE / "pilot48_manifest_v1.tsv")
    p.add_argument("--esmfold-dir", type=Path, default=BASE / "esmfold_linker_main300_v1")
    p.add_argument("--output-dir", type=Path, default=BASE / "esmfold_linker_main300_v1")
    return p.parse_args()


def load_esmfold(directory: Path) -> dict[str, dict]:
    scores = {}
    pred = directory / "predictions"
    paths = list(pred.glob("*.json")) if pred.is_dir() else []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            continue
        scores[payload["job_id"]] = payload
    return scores


def qc_block(records: list[dict]) -> dict:
    plddt = [float(row["mean_chain_plddt"]) for row in records]
    pae = [float(row["mean_inter_pae"]) for row in records]
    return {
        "n_ok_jobs": len(records),
        "mean_chain_plddt_max": max(plddt) if plddt else float("nan"),
        "mean_chain_plddt_median": sorted(plddt)[len(plddt) // 2] if plddt else float("nan"),
        "mean_inter_pae_median": sorted(pae)[len(pae) // 2] if pae else float("nan"),
        "n_mean_inter_pae_lt_15": sum(1 for value in pae if value < 15.0),
        "n_plddt_ge_50": sum(1 for value in plddt if value >= 50.0),
    }


def main() -> None:
    args = parse_args()
    links = read_tsv(args.links)
    manifest = {row["triplet_id"]: row for row in read_tsv(args.manifest)}
    pilot_ids = {row["triplet_id"] for row in read_tsv(args.pilot_manifest)}
    af3 = load_af3_scores(BASE)
    esm = load_esmfold(args.esmfold_dir)
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
        if ad["job_id"] not in esm or ac["job_id"] not in esm:
            missing.append({"triplet_id": triplet_id, "reason": "missing_esmfold"}); continue
        meta = manifest.get(triplet_id, {})
        row = {
            "triplet_id": triplet_id,
            "ac_id": ad["ac_id"],
            "in_pilot48": int(triplet_id in pilot_ids),
            "positive_source": meta.get("positive_source", ""),
            "seqid_CD_bin": meta.get("seqid_CD_bin", ""),
            "family_split": meta.get("family_split", ""),
            "af3_iptm_ad": af3[ad["job_id"]]["iptm"],
            "af3_iptm_ac": af3[ac["job_id"]]["iptm"],
            "esm_score_ad": esm[ad["job_id"]]["score_neg_mean_inter_pae"],
            "esm_score_ac": esm[ac["job_id"]]["score_neg_mean_inter_pae"],
            "esm_pae_ad": esm[ad["job_id"]]["mean_inter_pae"],
            "esm_pae_ac": esm[ac["job_id"]]["mean_inter_pae"],
            "esm_contacts_ad": esm[ad["job_id"]]["n_ca_contacts_8A"],
            "esm_contacts_ac": esm[ac["job_id"]]["n_ca_contacts_8A"],
            "esm_plddt_ad": esm[ad["job_id"]]["mean_chain_plddt"],
            "esm_plddt_ac": esm[ac["job_id"]]["mean_chain_plddt"],
        }
        row["hit_af3_iptm"] = ranking_hit(row["af3_iptm_ad"], row["af3_iptm_ac"])
        row["hit_esm_pae"] = ranking_hit(row["esm_score_ad"], row["esm_score_ac"])
        row["hit_esm_contacts"] = ranking_hit(row["esm_contacts_ad"], row["esm_contacts_ac"])
        rows.append(row)

    nested48 = [row for row in rows if row["in_pilot48"] == 1]
    both_correct = sum(1 for row in rows if row["hit_af3_iptm"] == 1 and row["hit_esm_pae"] == 1)
    af3_only = sum(1 for row in rows if row["hit_af3_iptm"] == 1 and row["hit_esm_pae"] == 0)
    esm_only = sum(1 for row in rows if row["hit_af3_iptm"] == 0 and row["hit_esm_pae"] == 1)
    both_wrong = sum(1 for row in rows if row["hit_af3_iptm"] == 0 and row["hit_esm_pae"] == 0)
    report = {
        "n_complete": len(rows),
        "n_missing": len(missing),
        "n_pilot48_nested": len(nested48),
        "missing_reasons": dict(sorted(
            {reason: sum(1 for row in missing if row["reason"] == reason)
             for reason in {row["reason"] for row in missing}}.items()
        )),
        "overall": {
            "af3_iptm": metric_block(rows, "hit_af3_iptm", "af3_iptm_ad", "af3_iptm_ac"),
            "esmfold_neg_mean_inter_pae": metric_block(rows, "hit_esm_pae", "esm_score_ad", "esm_score_ac"),
            "esmfold_n_ca_contacts_8A": metric_block(rows, "hit_esm_contacts", "esm_contacts_ad", "esm_contacts_ac"),
        },
        "nested_pilot48": {
            "af3_iptm": metric_block(nested48, "hit_af3_iptm", "af3_iptm_ad", "af3_iptm_ac"),
            "esmfold_neg_mean_inter_pae": metric_block(nested48, "hit_esm_pae", "esm_score_ad", "esm_score_ac"),
        } if nested48 else {},
        "strict_discordance_excluding_ties": {
            "both_D": both_correct, "AF3_only": af3_only,
            "ESMFold_only": esm_only, "both_C": both_wrong,
        },
        "esmfold_qc": qc_block(list(esm.values())),
        "protocol": {
            "esmfold": "facebook/esmfold_v1 via transformers EsmForProteinFolding; seqA+G25+seqB",
            "af3": "existing main300 seed1 triton summaries",
            "primary_esm_score": "-mean inter-chain PAE after dropping linker residues",
            "note": "ESMFold linker-PAE is not AF3 ipTM; compare only D vs C order.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        write_tsv(args.output_dir / "triplet_comparison_v1.tsv", rows)
    (args.output_dir / "comparison_summary_v1.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "n_complete": report["n_complete"],
        "n_missing": report["n_missing"],
        "missing_reasons": report["missing_reasons"],
        "overall": report["overall"],
        "nested_pilot48": report["nested_pilot48"],
        "strict_discordance_excluding_ties": report["strict_discordance_excluding_ties"],
        "esmfold_qc": report["esmfold_qc"],
    }, indent=2))


if __name__ == "__main__":
    main()
