#!/usr/bin/env python3
"""Score directed protein-sequence pairs with the official MINT Bernett head.

MINT's supplied ``CollateFn(512)`` randomly crops a chain above 512 tokens.
For a reproducible audit we run fixed independent crop seeds, retaining both
the mean score and crop sensitivity.  The caller supplies directed rows, then
the evaluation layer averages reciprocal scores into an undirected PPI score.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-tsv", type=Path, required=True)
    parser.add_argument("--out-tsv", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, default=ROOT / "data/external/model_training_sets/mint/repo"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "data/external/model_training_sets/mint/checkpoints/mint.ckpt"
    )
    parser.add_argument(
        "--mlp-checkpoint", type=Path,
        default=ROOT / "data/external/model_training_sets/mint/checkpoints/bernett_mlp.pth",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-length", type=int, default=512)
    parser.add_argument("--seeds", default="13,29,47,71,97")
    return parser.parse_args()


def batches(rows: list[dict[str, str]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    import torch

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested ({args.device}) but is unavailable")
    sys.path.insert(0, str(args.repo.resolve()))
    import mint.helpers.extract as mint_extract
    from mint.helpers.extract import CollateFn, MINTWrapper, load_config
    from mint.helpers.predict import SimpleMLP

    # The upstream CollateFn calls random.randint without importing random in
    # its own module.  Inject this module's random object so crop seeds remain
    # controllable while preserving the upstream inference path.
    mint_extract.random = random

    rows = list(csv.DictReader(args.pairs_tsv.open(encoding="utf-8"), delimiter="\t"))
    required = {"directed_pair_id", "query", "text"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"pairs must have {sorted(required)}")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise SystemExit("at least one crop seed is required")

    # MINT source now passes weights_only=, but the pinned torch 1.12 install
    # predates that keyword. Preserve official semantics after dropping only it.
    original_load = torch.load

    def compatible_load(*load_args, **load_kwargs):
        if "weights_only" not in inspect.signature(original_load).parameters:
            load_kwargs.pop("weights_only", None)
        return original_load(*load_args, **load_kwargs)

    cfg = load_config(str(args.repo / "data/esm2_t33_650M_UR50D.json"))
    torch.load = compatible_load
    try:
        wrapper = MINTWrapper(cfg, str(args.checkpoint), sep_chains=True, device=args.device)
    finally:
        torch.load = original_load
    wrapper.eval()
    mlp = SimpleMLP()
    mlp.load_state_dict(original_load(str(args.mlp_checkpoint), map_location=args.device))
    mlp.eval().to(args.device)

    all_scores: list[list[float]] = [[] for _ in rows]
    with torch.no_grad():
        for seed in seeds:
            # CollateFn uses the stdlib random module for crop start positions.
            random.seed(seed)
            torch.manual_seed(seed)
            collate = CollateFn(args.crop_length)
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start : start + args.batch_size]
                chains, chain_ids = collate([(x["query"], x["text"]) for x in batch])
                embeddings = wrapper(chains.to(args.device), chain_ids.to(args.device))
                probs = torch.sigmoid(mlp(embeddings)).reshape(-1).detach().cpu().tolist()
                for offset, score in enumerate(probs):
                    all_scores[start + offset].append(float(score))
                if (start // args.batch_size) % 250 == 0:
                    print(f"seed={seed} scored {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["directed_pair_id", "score_mean", "score_sd", "n_crop_seeds"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row, scores in zip(rows, all_scores):
            mean = sum(scores) / len(scores)
            variance = sum((x - mean) ** 2 for x in scores) / len(scores)
            writer.writerow({
                "directed_pair_id": row["directed_pair_id"],
                "score_mean": f"{mean:.8f}",
                "score_sd": f"{variance ** 0.5:.8f}",
                "n_crop_seeds": len(scores),
            })
    print(f"scored {len(rows)} directed pairs across {len(seeds)} crop seeds -> {args.out_tsv}")


if __name__ == "__main__":
    main()
