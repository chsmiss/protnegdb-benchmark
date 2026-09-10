#!/usr/bin/env python3
"""Aggregate five full-core MINT crop seeds using Figure 2's macro endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/interim/structure_core_mint_full_v1"
SEEDS = [13, 29, 47, 71, 97]


def read(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def macro(links: list[dict[str, str]], pair_scores: dict[str, float]) -> tuple[float, list[float], list[float]]:
    by_ac: dict[str, list[float]] = defaultdict(list); deltas = []
    for row in links:
        ac = pair_scores[row["ac_pair_id"]]; ad = pair_scores[row["ad_pair_id"]]
        win = 1.0 if ad > ac else 0.0 if ad < ac else .5
        by_ac[row["ac_id"]].append(win); deltas.append(ad - ac)
    ac_values = [statistics.mean(values) for values in by_ac.values()]
    return statistics.mean(ac_values), ac_values, deltas


def bootstrap(values: list[float], seed: int = 20260816, n: int = 10000) -> list[float]:
    rng = np.random.RandomState(seed); x = np.asarray(values, dtype=float); out=[]
    remaining=n
    while remaining:
        batch=min(500,remaining); idx=rng.randint(0,len(x),size=(batch,len(x)))
        out.extend(x[idx].mean(axis=1).tolist()); remaining-=batch
    return [float(v) for v in np.percentile(out,[2.5,97.5])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    return parser.parse_args()


def main() -> None:
    args=parse_args(); links=read(args.base/"core_triplet_links_v1.tsv")
    directed=read(args.base/"mint_directed_pairs_v1.tsv")
    did_meta={row["directed_pair_id"]:row for row in directed}
    seed_maps={}
    for seed in SEEDS:
        rows=read(args.base/f"scores_seed{seed}_v1.tsv")
        if len(rows)!=len(directed) or any(row["n_crop_seeds"]!="1" for row in rows):
            raise SystemExit(f"seed {seed} score contract failed")
        seed_maps[seed]={row["directed_pair_id"]:float(row["score_mean"]) for row in rows}
        if set(seed_maps[seed])!=set(did_meta):
            raise SystemExit(f"seed {seed} directed IDs differ")

    pair_ids=sorted({row["pair_id"] for row in directed})
    pair_rows=[]; seed_pair_maps={seed:{} for seed in SEEDS}
    for pair_id in pair_ids:
        values=[]
        for seed in SEEDS:
            score=(seed_maps[seed][f"{pair_id}:forward"]+seed_maps[seed][f"{pair_id}:reverse"])/2
            seed_pair_maps[seed][pair_id]=score; values.append(score)
        pair_rows.append({"pair_id":pair_id,"score_mean":f"{statistics.mean(values):.8f}",
                          "score_sd_across_crop_seeds":f"{statistics.pstdev(values):.8f}",
                          **{f"score_seed{seed}":f"{value:.8f}" for seed,value in zip(SEEDS,values)}})
    mean_pair={row["pair_id"]:float(row["score_mean"]) for row in pair_rows}
    rate,ac_values,deltas=macro(links,mean_pair); ci=bootstrap(ac_values)
    per_seed=[]
    for seed in SEEDS:
        seed_rate,_values,seed_deltas=macro(links,seed_pair_maps[seed])
        per_seed.append({"seed":seed,"p_d_gt_c_macro":seed_rate,
                         "median_delta":statistics.median(seed_deltas)})
    triplet_rows=[]
    for row in links:
        ac=mean_pair[row["ac_pair_id"]]; ad=mean_pair[row["ad_pair_id"]]
        triplet_rows.append({"triplet_id":row["triplet_id"],"ac_id":row["ac_id"],
                             "score_ac":f"{ac:.8f}","score_ad":f"{ad:.8f}",
                             "delta":f"{ad-ac:.8f}","rank_value":1 if ad>ac else 0 if ad<ac else .5})
    report={
        "model":"MINT Bernett","coverage":"full Figure 2 structural core",
        "crop_seeds":SEEDS,"direction_aggregation":"mean reciprocal direction per seed",
        "seed_aggregation":"mean pair score across five fixed crop seeds before ranking",
        "n_triplets":len(links),"n_unique_ac":len(ac_values),"n_unique_pairs":len(pair_ids),
        "p_d_gt_c_macro":rate,"ci95_unique_ac_bootstrap":ci,
        "median_delta":statistics.median(deltas),"mean_delta":statistics.mean(deltas),
        "per_seed":per_seed,
        "pair_crop_sd":{"median":statistics.median(float(row["score_sd_across_crop_seeds"]) for row in pair_rows),
                        "mean":statistics.mean(float(row["score_sd_across_crop_seeds"]) for row in pair_rows)},
    }
    write(args.base/"mint_pair_scores_v1.tsv",pair_rows,list(pair_rows[0]))
    write(args.base/"mint_triplet_scores_v1.tsv",triplet_rows,list(triplet_rows[0]))
    (args.base/"mint_full_core_eval_v1.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
