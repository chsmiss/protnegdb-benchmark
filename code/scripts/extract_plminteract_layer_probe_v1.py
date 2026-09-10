#!/usr/bin/env python3
"""Extract PLM-interact pair CLS probes from selected ESM-2 layers.

No training. Frozen published human-V12 checkpoint. The official decision is
last-layer CLS -> ReLU -> Linear(1x1280) -> sigmoid. Intermediate layers use
the same linear head as a diagnostic probe only.

Input is the 1000-triplet pilot (A-D, A-C) plus same-anchor random partners
(A-R). Literature assay negatives and structural noncontacts stay separate
roles; they are not merged into one negative label.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = ROOT / "data/interim/counterfactual_1000_triplet_pilot_v1"
DEFAULT_OUT = DEFAULT_PILOT / "plminteract_layer_probe_v1"
DEFAULT_LAYERS = (0, 8, 16, 24, 33)
LAST_LAYER = 33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N directed pairs")
    parser.add_argument("--pairs-csv", type=Path, default=None,
                        help="prebuilt unique directed pairs; skips the 1000-pilot builder")
    parser.add_argument("--backbone", choices=("plminteract", "esm2"), default="plminteract")
    parser.add_argument("--cls-only", action="store_true",
                        help="skip A/B token pooling; only save pair CLS")
    parser.add_argument("--save-layer-cls", action="store_true",
                        help="write layer_cls_v1.npz with CLS for every requested layer")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--esm2-dir", type=Path,
        default=ROOT / "data/external/model_training_sets/esm2_t33_650M_UR50D",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "data/external/model_training_sets/checkpoints/PLM-interact-650M-humanV12/pytorch_model.bin",
    )
    return parser.parse_args()


def load_inference_module():
    path = ROOT / "code/scripts" / "run_plminteract_inference_v1.py"
    spec = importlib.util.spec_from_file_location("run_plminteract_inference_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def probe_score_from_cls(cls: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Official head on an arbitrary CLS: sigmoid(ReLU(cls) @ W.T + b).

    cls: [..., 1280], weight: [1, 1280], bias: [1] or scalar.
    """
    hidden = relu(cls)
    logits = hidden @ np.asarray(weight, dtype=np.float64).reshape(-1)
    logits = logits + float(np.asarray(bias).reshape(-1)[0])
    return sigmoid(logits)


def pair_token_spans(input_ids: list[int], eos_id: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """ESM pair format: <cls> seqA <eos> seqB <eos> [pad...].

    Returns half-open (start, end) spans for the amino-acid tokens of A and B.
    """
    eos = [i for i, token in enumerate(input_ids) if token == eos_id]
    if len(eos) < 2:
        return None
    a0, a1 = 1, eos[0]
    b0, b1 = eos[0] + 1, eos[1]
    if a1 <= a0 or b1 <= b0:
        return None
    return (a0, a1), (b0, b1)


def cosine_vec(left: np.ndarray, right: np.ndarray) -> float:
    num = float(np.dot(left, right))
    den = float(np.linalg.norm(left) * np.linalg.norm(right))
    if den == 0.0:
        return float("nan")
    return num / den


def layer_name(index: int, last_index: int) -> str:
    if index == 0:
        return "L0_embed"
    if index == last_index:
        return f"L{index}_last"
    return f"L{index}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_unique_pairs(pilot_dir: Path) -> list[dict[str, str]]:
    ad_ac = read_csv(pilot_dir / "pilot_plminteract_directed_pairs_v1.csv")
    random_path = pilot_dir / "random_partner_v1" / "random_partners_plminteract_pairs.csv"
    random_rows = read_csv(random_path) if random_path.exists() else []
    seen: dict[str, dict[str, str]] = {}
    for source, rows in (("ad_ac", ad_ac), ("random", random_rows)):
        for row in rows:
            ident = row["directed_pair_id"]
            if ident in seen:
                continue
            seen[ident] = {
                "directed_pair_id": ident,
                "query": row["query"],
                "text": row["text"],
                "source": source,
            }
    return list(seen.values())


def write_pairs_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["directed_pair_id", "query", "text", "source"]
    extra = []
    for row in rows:
        for key in row:
            if key not in fields and key not in extra:
                extra.append(key)
    fields.extend(extra)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean_pool(hidden, start: int, end: int, mask) -> np.ndarray:
    sl = hidden[start:end]
    m = mask[start:end]
    if int(m.sum()) == 0:
        return np.zeros(hidden.shape[-1], dtype=np.float64)
    return (sl * m[:, None]).sum(axis=0) / m.sum()


def extract_batch(esm, tokenizer, classifier_w, classifier_b, batch, layers, max_length, device,
                  cls_only: bool = False):
    import torch

    texts_a = [row["query"] for row in batch]
    texts_b = [row["text"] for row in batch]
    tok = tokenizer(
        texts_a, texts_b, padding=True, truncation="longest_first",
        return_tensors="pt", max_length=max_length,
    ).to(device)
    with torch.no_grad():
        out = esm(**tok, output_hidden_states=True)
        hidden_states = out.hidden_states
        last_index = len(hidden_states) - 1
        ids = tok["input_ids"].detach().cpu().numpy()
        attn = tok["attention_mask"].detach().cpu().numpy()
        eos_id = tokenizer.eos_token_id
        rows = []
        last_cls = []
        layer_cls = []
        for i, row in enumerate(batch):
            record = {"directed_pair_id": row["directed_pair_id"]}
            spans = None if cls_only else pair_token_spans(ids[i].tolist(), eos_id)
            layer_vecs = []
            for layer in layers:
                if layer > last_index:
                    raise SystemExit(f"layer {layer} > last hidden index {last_index}")
                cls_t = hidden_states[layer][i, 0].detach().float()
                cls = cls_t.cpu().numpy()
                score = float(probe_score_from_cls(cls, classifier_w, classifier_b))
                ab = float("nan")
                if not cls_only:
                    h = hidden_states[layer][i].detach().float().cpu().numpy()
                    if spans is not None:
                        (a0, a1), (b0, b1) = spans
                        a_mean = mean_pool(h, a0, a1, attn[i])
                        b_mean = mean_pool(h, b0, b1, attn[i])
                        ab = cosine_vec(a_mean, b_mean)
                record[f"probe_L{layer}"] = score
                record[f"cls_norm_L{layer}"] = float(np.linalg.norm(cls))
                record[f"ab_cosine_L{layer}"] = ab
                layer_vecs.append(cls.astype(np.float16))
            last = hidden_states[last_index][i, 0].detach().float().cpu().numpy()
            record["official_score"] = float(probe_score_from_cls(last, classifier_w, classifier_b))
            record["last_hidden_index"] = last_index
            rows.append(record)
            last_cls.append(last.astype(np.float16))
            layer_cls.append(np.stack(layer_vecs, axis=0))
        del out, hidden_states
    return rows, np.stack(last_cls, axis=0), last_index, np.stack(layer_cls, axis=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pairs_csv is not None:
        pairs = read_csv(args.pairs_csv)
        if not pairs or "directed_pair_id" not in pairs[0] or "query" not in pairs[0]:
            raise SystemExit("pairs csv needs directed_pair_id,query,text")
        for row in pairs:
            row.setdefault("source", row.get("cohort", "supplied"))
    else:
        pairs = build_unique_pairs(args.pilot_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    pairs_csv = args.output_dir / "unique_directed_pairs_v1.csv"
    write_pairs_csv(pairs_csv, pairs)
    manifest = {
        "n_directed_pairs": len(pairs),
        "n_ad_ac": sum(1 for r in pairs if r["source"] == "ad_ac"),
        "n_random": sum(1 for r in pairs if r["source"] == "random"),
        "layers": list(args.layers),
        "max_length": args.max_length,
        "backbone": args.backbone,
        "cls_only": bool(args.cls_only),
        "save_layer_cls": bool(args.save_layer_cls),
        "prepare_only": bool(args.prepare_only),
    }
    (args.output_dir / "extract_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)
    if args.prepare_only:
        return

    import torch

    inf = load_inference_module()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.backbone == "esm2":
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.esm2_dir, use_fast=True)
        esm = AutoModelForMaskedLM.from_pretrained(args.esm2_dir).to(device).eval()
        w = np.zeros((1, 1280), dtype=np.float64)
        b = np.zeros((1,), dtype=np.float64)
    else:
        esm, classifier_w, classifier_b, tokenizer = inf.load_plminteract(
            args.esm2_dir, args.checkpoint, device)
        w = classifier_w.detach().float().cpu().numpy()
        b = classifier_b.detach().float().cpu().numpy()
        np.save(args.output_dir / "classifier_weight_v1.npy", w)
        np.save(args.output_dir / "classifier_bias_v1.npy", b)

    score_rows = []
    cls_blocks = []
    layer_blocks = []
    ids = []
    last_index = None
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start:start + args.batch_size]
        batch_rows, batch_cls, last_index, batch_layers = extract_batch(
            esm, tokenizer, w, b, batch, args.layers, args.max_length, device,
            cls_only=bool(args.cls_only))
        score_rows.extend(batch_rows)
        cls_blocks.append(batch_cls)
        if args.save_layer_cls:
            layer_blocks.append(batch_layers)
        ids.extend(row["directed_pair_id"] for row in batch)
        done = min(start + args.batch_size, len(pairs))
        print(f"scored {done}/{len(pairs)}", flush=True)

    scores_path = args.output_dir / "layer_probe_scores_v1.tsv"
    fieldnames = ["directed_pair_id", "official_score", "last_hidden_index"]
    for layer in args.layers:
        fieldnames.extend([f"probe_L{layer}", f"cls_norm_L{layer}", f"ab_cosine_L{layer}"])
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in score_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    np.savez_compressed(
        args.output_dir / "last_cls_v1.npz",
        directed_pair_id=np.array(ids),
        cls=np.concatenate(cls_blocks, axis=0),
    )
    if args.save_layer_cls:
        np.savez_compressed(
            args.output_dir / "layer_cls_v1.npz",
            directed_pair_id=np.array(ids),
            layers=np.array(list(args.layers), dtype=np.int16),
            cls=np.concatenate(layer_blocks, axis=0),
        )
    manifest.update({
        "backbone": args.backbone,
        "cls_only": bool(args.cls_only),
        "save_layer_cls": bool(args.save_layer_cls),
        "last_hidden_index": last_index,
        "n_scored": len(score_rows),
        "scores_tsv": str(scores_path.resolve().relative_to(ROOT.resolve())),
        "cls_npz": "last_cls_v1.npz",
        "layer_cls_npz": "layer_cls_v1.npz" if args.save_layer_cls else "",
        "layer_names": {str(i): layer_name(i, last_index or LAST_LAYER) for i in args.layers},
    })
    (args.output_dir / "extract_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(score_rows)} rows -> {scores_path}", flush=True)


if __name__ == "__main__":
    main()
