#!/usr/bin/env python3
"""Score protein pairs with PLM-interact (650M human-V12) directly.

Loads:
- ESM-2 650M base (facebook/esm2_t33_650M_UR50D) from the local dir
  (config.json + model.safetensors)
- the PLM-interact fine-tuned checkpoint (pytorch_model.bin) whose state dict
  holds `esm_mask.*` (the fine-tuned ESM-2 weights) + `classifier.*` (1x1280).

The pipeline is exactly the repo's forward: ESM-2 CLS token -> ReLU -> Linear
-> sigmoid. Input is a CSV with `query,text` (the two protein sequences).

Usage (in esmc5090 env; torch must support sm_120 for RTX 5090 GPUs):
  python scripts/run_plminteract_inference_v1.py --pairs-csv ... --out-csv ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--esm2-dir", type=Path,
                        default=ROOT / "data/external/model_training_sets/esm2_t33_650M_UR50D")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "data/external/model_training_sets/checkpoints/PLM-interact-650M-humanV12/pytorch_model.bin")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    return parser.parse_args()


def load_plminteract(esm2_dir: Path, checkpoint: Path, device):
    """Load ESM-2 650M + PLM-interact fine-tuned weights and the pair head.

    Returns (esm, classifier_w, classifier_b, tokenizer) where the forward is
    ESM-2 CLS -> ReLU -> Linear -> sigmoid, matching the PLM-interact repo.
    Shared by inference and fine-tuning so both see an identical model.
    """
    import torch
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

    config = AutoConfig.from_pretrained(esm2_dir)
    tokenizer = AutoTokenizer.from_pretrained(esm2_dir, use_fast=True)

    # load fine-tuned ESM-2 weights into the ESM model
    esm = AutoModelForMaskedLM.from_pretrained(esm2_dir, config=config)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    esm_state = {k.removeprefix("esm_mask."): v for k, v in ckpt.items() if k.startswith("esm_mask.")}
    # transformers 4.x ESM registers a vestigial absolute position_embeddings
    # parameter; 5.x drops it (ESM-2 uses rotary attention, so it carries no
    # signal either way) and moves rotary inv_freq from per-layer buffers to one
    # shared buffer. inv_freq is deterministic from the config, so start from
    # the freshly loaded base state and let the fine-tuned weights override.
    merged = dict(esm.state_dict())
    model_keys = set(merged)
    filtered = {k: v for k, v in esm_state.items() if k in model_keys}
    merged.update(filtered)
    dropped = sorted(set(esm_state) - model_keys)
    if dropped:
        print(f"note: dropped {len(dropped)} checkpoint keys absent in this "
              f"transformers build (e.g. {dropped[:2]})", flush=True)
    esm.load_state_dict(merged, strict=True)
    esm.to(device).eval()

    classifier_w = ckpt["classifier.weight"].to(device)
    classifier_b = ckpt["classifier.bias"].to(device)
    return esm, classifier_w, classifier_b, tokenizer


def main() -> None:
    args = parse_args()
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    esm, classifier_w, classifier_b, tokenizer = load_plminteract(
        args.esm2_dir, args.checkpoint, device)

    rows = list(csv.DictReader(args.pairs_csv.open(encoding="utf-8")))
    if not rows:
        raise SystemExit("empty pairs csv")

    scores = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            texts_a = [r["query"] for r in batch]
            texts_b = [r["text"] for r in batch]
            tok = tokenizer(texts_a, texts_b, padding=True, truncation="longest_first",
                            return_tensors="pt", max_length=args.max_length).to(device)
            out = esm(**tok, output_hidden_states=True)
            cls_emb = out.hidden_states[-1][:, 0, :]  # [B, 1280]
            emb = torch.relu(cls_emb)
            logits = emb @ classifier_w.t() + classifier_b
            probs = torch.sigmoid(logits).squeeze(-1)
            scores.extend(p.cpu().item() for p in probs)

    # Existing callers use a score-only CSV.  The unified audit supplies a
    # directed_pair_id and needs a content key rather than fragile row order.
    include_id = "directed_pair_id" in rows[0]
    with args.out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["directed_pair_id", "score"] if include_id else ["score"])
        for row, s in zip(rows, scores):
            w.writerow([row["directed_pair_id"], f"{s:.6f}"] if include_id else [f"{s:.6f}"])
    print(f"scored {len(scores)} pairs -> {args.out_csv}")


if __name__ == "__main__":
    main()
