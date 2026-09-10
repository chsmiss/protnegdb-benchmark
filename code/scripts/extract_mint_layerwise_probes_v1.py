#!/usr/bin/env python3
"""Extract MINT pair embeddings (concat of per-chain mean-pools) at ESM layers.

MINT is not a pair-CLS model. Official Bernett scoring does:
  collate two chains (crop 512) -> ESM-2-multimer -> mean-pool each chain
  excluding special tokens -> concat 2560-d -> SimpleMLP.

This script freezes those pair vectors at selected layers. No backbone training.
Crop seed is fixed (default 13) so the geometry is reproducible.

--backbone esm2 encodes the same cropped amino acids independently with
original ESM-2 650M (no multimer attention) and concatenates mean-pools.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS = ROOT / "data/interim/plminteract_layerwise_probes_v1/unique_directed_pairs_v1.csv"
DEFAULT_OUT = ROOT / "data/interim/mint_layerwise_probes_v1"
DEFAULT_LAYERS = tuple(range(34))
MINT_REPO = ROOT / "data/external/model_training_sets/mint/repo"
MINT_CKPT = ROOT / "data/external/model_training_sets/mint/checkpoints/mint.ckpt"
MINT_MLP = ROOT / "data/external/model_training_sets/mint/checkpoints/bernett_mlp.pth"
ESM2_DIR = ROOT / "data/external/model_training_sets/esm2_t33_650M_UR50D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backbone", choices=("mint", "esm2"), default="mint")
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-length", type=int, default=512)
    parser.add_argument("--crop-seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repo", type=Path, default=MINT_REPO)
    parser.add_argument("--checkpoint", type=Path, default=MINT_CKPT)
    parser.add_argument("--mlp-checkpoint", type=Path, default=MINT_MLP)
    parser.add_argument("--esm2-dir", type=Path, default=ESM2_DIR)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def mean_pool_masked(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """hidden [T, D] or [B, T, D]; mask same rank minus D. Zeros if empty."""
    hidden = np.asarray(hidden, dtype=np.float64)
    mask = np.asarray(mask)
    if hidden.ndim == 2:
        m = mask.astype(np.float64)
        if float(m.sum()) <= 0:
            return np.zeros(hidden.shape[-1], dtype=np.float64)
        return (hidden * m[:, None]).sum(axis=0) / m.sum()
    m = mask.astype(np.float64)
    counts = m.sum(axis=1, keepdims=True)
    summed = (hidden * m[:, :, None]).sum(axis=1)
    out = np.zeros_like(summed)
    ok = counts.reshape(-1) > 0
    out[ok] = summed[ok] / counts[ok]
    return out


def concat_chain_means(hidden: np.ndarray, chain_ids: np.ndarray, token_mask: np.ndarray) -> np.ndarray:
    """hidden [B, T, D] -> [B, 2D] mean-pool of chain 0 then chain 1."""
    a = mean_pool_masked(hidden, token_mask & (chain_ids == 0))
    b = mean_pool_masked(hidden, token_mask & (chain_ids == 1))
    return np.concatenate([a, b], axis=-1)


def mlp_hidden(pair_vec: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """z = ReLU(x @ W.T + b) for torch.nn.Linear."""
    x = np.asarray(pair_vec, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64).reshape(-1)
    pre = x @ w.T + b
    return np.maximum(pre, 0.0)


def compatible_torch_load(torch_mod):
    original = torch_mod.load

    def wrapped(*args, **kwargs):
        if "weights_only" not in inspect.signature(original).parameters:
            kwargs.pop("weights_only", None)
        return original(*args, **kwargs)

    return wrapped


def tokens_to_aa(token_row, alphabet, pad_idx: int, cls_idx: int, eos_idx: int) -> str:
    aas = []
    for tok_id in token_row.tolist():
        ident = int(tok_id)
        if ident in (pad_idx, cls_idx, eos_idx):
            continue
        tok = alphabet.get_tok(ident)
        if len(tok) == 1:
            aas.append(tok)
    return "".join(aas)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or (DEFAULT_OUT / args.backbone)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(csv.DictReader(args.pairs_csv.open(encoding="utf-8", newline="")))
    required = {"directed_pair_id", "query", "text"}
    if not pairs or not required.issubset(pairs[0]):
        raise SystemExit(f"pairs csv needs {sorted(required)}")
    if args.limit > 0:
        pairs = pairs[: args.limit]

    import torch

    sys.path.insert(0, str(args.repo.resolve()))
    import mint.helpers.extract as mint_extract
    from mint.helpers.extract import CollateFn, MINTWrapper, load_config
    from mint.helpers.predict import SimpleMLP

    mint_extract.random = random
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"loading {args.backbone} on {device}", flush=True)
    layers = list(args.layers)

    mlp_w1 = mlp_b1 = mlp_w2 = mlp_b2 = None
    if args.backbone == "mint":
        cfg = load_config(str(args.repo / "data/esm2_t33_650M_UR50D.json"))
        original_load = torch.load
        torch.load = compatible_torch_load(torch)
        try:
            wrapper = MINTWrapper(cfg, str(args.checkpoint), sep_chains=True, device=str(device))
        finally:
            torch.load = original_load
        wrapper.eval()
        mlp = SimpleMLP()
        mlp_kwargs = {"map_location": device}
        if "weights_only" in inspect.signature(original_load).parameters:
            mlp_kwargs["weights_only"] = False
        mlp.load_state_dict(original_load(str(args.mlp_checkpoint), **mlp_kwargs))
        mlp.eval()
        linear0 = mlp.project[0]
        linear1 = mlp.project[3]
        mlp_w1 = linear0.weight.detach().float().cpu().numpy()
        mlp_b1 = linear0.bias.detach().float().cpu().numpy()
        mlp_w2 = linear1.weight.detach().float().cpu().numpy()
        mlp_b2 = linear1.bias.detach().float().cpu().numpy()
        np.save(out_dir / "mlp_w1_v1.npy", mlp_w1)
        np.save(out_dir / "mlp_b1_v1.npy", mlp_b1)
        np.save(out_dir / "mlp_w2_v1.npy", mlp_w2)
        np.save(out_dir / "mlp_b2_v1.npy", mlp_b2)
        encoder = wrapper.model
        hf_model = tokenizer = None
    else:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.esm2_dir, use_fast=True)
        hf_model = AutoModelForMaskedLM.from_pretrained(args.esm2_dir).to(device).eval()
        encoder = None
        wrapper = None

    alphabet = CollateFn(args.crop_length).alphabet
    ids = []
    blocks = []
    random.seed(args.crop_seed)
    torch.manual_seed(args.crop_seed)
    collate = CollateFn(args.crop_length)

    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            seq_pairs = [(row["query"], row["text"]) for row in batch]
            if args.backbone == "mint":
                chains, chain_ids = collate(seq_pairs)
                chains = chains.to(device)
                chain_ids = chain_ids.to(device)
                mask = (
                    (~chains.eq(encoder.cls_idx))
                    & (~chains.eq(encoder.eos_idx))
                    & (~chains.eq(encoder.padding_idx))
                )
                out = encoder(chains, chain_ids, repr_layers=layers)
                chain_np = chain_ids.detach().cpu().numpy()
                mask_np = mask.detach().cpu().numpy().astype(bool)
                stacked = []
                for layer in layers:
                    hidden = out["representations"][layer].detach().float().cpu().numpy()
                    stacked.append(concat_chain_means(hidden, chain_np, mask_np))
                pair_stack = np.stack(stacked, axis=1)
            else:
                tok_a = collate.convert([p[0] for p in seq_pairs])
                tok_b = collate.convert([p[1] for p in seq_pairs])
                texts_a = [
                    tokens_to_aa(row, alphabet, alphabet.padding_idx, alphabet.cls_idx, alphabet.eos_idx)
                    for row in tok_a
                ]
                texts_b = [
                    tokens_to_aa(row, alphabet, alphabet.padding_idx, alphabet.cls_idx, alphabet.eos_idx)
                    for row in tok_b
                ]
                stacked_a = _hf_mean_layers(hf_model, tokenizer, texts_a, layers, device)
                stacked_b = _hf_mean_layers(hf_model, tokenizer, texts_b, layers, device)
                pair_stack = np.concatenate([stacked_a, stacked_b], axis=-1)
            blocks.append(pair_stack.astype(np.float16))
            ids.extend(row["directed_pair_id"] for row in batch)
            print(f"scored {min(start + len(batch), len(pairs))}/{len(pairs)}", flush=True)

    np.savez_compressed(
        out_dir / "layer_cls_v1.npz",
        directed_pair_id=np.array(ids),
        layers=np.array(layers, dtype=np.int16),
        cls=np.concatenate(blocks, axis=0),
    )
    manifest = {
        "backbone": args.backbone,
        "n_pairs": len(ids),
        "layers": layers,
        "crop_length": args.crop_length,
        "crop_seed": args.crop_seed,
        "pair_dim": int(blocks[0].shape[-1]) if blocks else 0,
        "note": "pair vector = concat(mean_pool chain A, mean_pool chain B); not pair CLS",
    }
    (out_dir / "extract_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


def _hf_mean_layers(model, tokenizer, texts, layers, device):
    import torch
    tok = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    hidden_states = model(**tok, output_hidden_states=True).hidden_states
    special = set()
    for name in ("cls_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            special.add(int(value))
    ids = tok["input_ids"].detach().cpu().numpy()
    mask = np.ones(ids.shape, dtype=bool)
    for sid in special:
        mask &= ids != sid
    stacked = []
    for layer in layers:
        hidden = hidden_states[layer].detach().float().cpu().numpy()
        stacked.append(mean_pool_masked(hidden, mask))
    return np.stack(stacked, axis=1)


if __name__ == "__main__":
    main()
