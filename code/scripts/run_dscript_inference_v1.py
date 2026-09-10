#!/usr/bin/env python3
"""Score protein pairs with D-SCRIPT / Topsy-Turvy directly (v1).

Why not `dscript predict`: dscript 0.3.1's blocked predict worker is broken
twice over — the error handler itself crashes
(`log(e, file=None, printAlso=True)`; utils.log has no `printAlso` kwarg) and
the fallback `output_queue.put(i0, i1, -1)` passes three args to a one-arg
method. Any per-pair RuntimeError therefore hangs the run silently.

The underlying RuntimeError is also real: ModelInteraction registers the
position-weight vector `xx` with only 2000 entries, so `map_predict` fails for
any pair where either protein is longer than 2000 residues. That is a model
limitation, not something to patch around by truncating embeddings — such
pairs are left unscored and counted in the summary sidecar.

This script mirrors run_plminteract_inference_v1.py's pattern: load the
published checkpoint locally, stream precomputed embeddings from the h5
written by `dscript embed`, score each pair, write `protA\\tprotB\\tscore`
rows (the format build_structure_triplet_model_report_v1.py parses).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ModelInteraction.xx has exactly 2000 entries; map_predict broadcasts it
# against the N x M contact map, so N or M above this raises RuntimeError.
MODEL_MAX_LEN = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True,
                        help="whitespace-separated protA protB rows (no header)")
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="h5 from `dscript embed`: one (1, N, 6165) dataset per accession")
    parser.add_argument("--model", type=Path, required=True,
                        help="local checkpoint dir (config.json + model.safetensors)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output TSV: protA protB score")
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=MODEL_MAX_LEN)
    return parser.parse_args()


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts and parts[:2] != ["protein_a", "protein_b"]:
                pairs.append((parts[0], parts[1]))
    return pairs


def iter_pair_scores(pairs, get_len, predict_pair, max_len):
    """Yield (protA, protB, score); count pairs that cannot be scored.

    get_len(prot) -> residue count; predict_pair(protA, protB) -> float and
    may raise RuntimeError (e.g. CUDA OOM). Both failure modes skip the pair
    and accumulate counters instead of aborting the run.
    """
    stats = {"scored": 0, "skipped_too_long": 0, "runtime_errors": 0}
    for prot_a, prot_b in pairs:
        if get_len(prot_a) > max_len or get_len(prot_b) > max_len:
            stats["skipped_too_long"] += 1
            continue
        try:
            score = predict_pair(prot_a, prot_b)
        except RuntimeError:
            stats["runtime_errors"] += 1
            continue
        stats["scored"] += 1
        yield prot_a, prot_b, score, dict(stats)


def main() -> None:
    args = parse_args()
    import h5py
    import torch
    from dscript.models.interaction import DSCRIPTModel

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model = DSCRIPTModel.from_pretrained(args.model)
    model.to(device).eval()

    pairs = read_pairs(args.pairs)
    h5 = h5py.File(args.embeddings, "r")

    def get_len(prot: str) -> int:
        return h5[prot].shape[1]

    @torch.no_grad()
    def predict_pair(prot_a: str, prot_b: str) -> float:
        z0 = torch.from_numpy(h5[prot_a][:]).to(device)
        z1 = torch.from_numpy(h5[prot_b][:]).to(device)
        _, phat = model.map_predict(z0, z1)
        return phat.item()

    stats = {"scored": 0, "skipped_too_long": 0, "runtime_errors": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out:
        for prot_a, prot_b, score, stats in iter_pair_scores(
            pairs, get_len, predict_pair, args.max_len
        ):
            out.write(f"{prot_a}\t{prot_b}\t{score:.6f}\n")
            if stats["scored"] % 500 == 0:
                print(f"scored {stats['scored']}/{len(pairs)}", flush=True)
    h5.close()

    summary = {
        "model": str(args.model),
        "pairs_total": len(pairs),
        **stats,
        "max_len": args.max_len,
        "note": "pairs over max_len are a dscript model limitation "
                "(position-weight vector capped at 2000 residues), left unscored",
    }
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
