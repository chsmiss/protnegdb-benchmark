#!/usr/bin/env python3
"""Fine-tune PLM-interact 650M on structure-triplet pairs (exploratory).

Trains the exact PLM-interact inference graph (ESM-2 CLS -> ReLU -> Linear ->
sigmoid) with BCE on the PDB-grouped train/dev CSVs from
prepare_structure_triplet_finetune_v1.py. This keeps the trained model
drop-in compatible with run_plminteract_inference_v1.py: the saved checkpoint
uses the published esm_mask.*/classifier.* state-dict format.

Why not the repo's train_binary.py: it needs sentence_transformers (absent in
the sm_120-capable env) and its CrossEncoder head (EsmForSequenceClassification
dense+out_proj) does not match the published PLM-interact head, so the
published humanV12 weights cannot be resumed faithfully through it.

Exploratory status: split controls PDB-group leakage only; homology-level
leakage control is still a v1 audit gate. Labels keep their v1 semantics
(A-C = assembly-context direct noncontact, not universal non-binder).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

from run_plminteract_inference_v1 import load_plminteract  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--dev-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--esm2-dir", type=Path,
                        default=ROOT / "data/external/model_training_sets/esm2_t33_650M_UR50D")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "data/external/model_training_sets/checkpoints/PLM-interact-650M-humanV12/pytorch_model.bin")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr-scale", type=float, default=10.0,
                        help="classifier head LR = lr * this scale")
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--freeze-esm", action="store_true",
                        help="ablation: train the classifier head only")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--fixed-epochs",
        action="store_true",
        help="train all requested epochs and save the last checkpoint; no early stop",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="do not evaluate a dev set; implies --fixed-epochs",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def average_precision(labels: list[int], scores: list[float]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = 0
    area = 0.0
    total_pos = sum(labels)
    if total_pos == 0:
        return float("nan")
    for rank, idx in enumerate(order, 1):
        if labels[idx]:
            hits += 1
            area += hits / rank
    return area / total_pos


def main() -> None:
    args = parse_args()
    import torch

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    esm, classifier_w, classifier_b, tokenizer = load_plminteract(
        args.esm2_dir, args.checkpoint, device)
    esm.train()
    esm.gradient_checkpointing_enable()

    # Single-Linear head as a proper module so the optimizer sees it.
    head = torch.nn.Linear(classifier_w.shape[1], classifier_w.shape[0], bias=True).to(device)
    with torch.no_grad():
        head.weight.copy_(classifier_w)
        head.bias.copy_(classifier_b)

    if args.freeze_esm:
        for p in esm.parameters():
            p.requires_grad_(False)
    esm_params = [p for p in esm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": esm_params, "lr": args.lr},
        {"params": head.parameters(), "lr": args.lr * args.head_lr_scale},
    ], weight_decay=0.01)

    train_rows = read_pairs(args.train_csv)
    if args.skip_eval:
        args.fixed_epochs = True
    if args.dev_csv is None:
        if not args.skip_eval:
            raise SystemExit("--dev-csv is required unless --skip-eval")
        dev_rows = []
    else:
        dev_rows = read_pairs(args.dev_csv)
    steps_per_epoch = math.ceil(len(train_rows) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward_probs(batch: list[dict[str, str]]):
        tok = tokenizer([r["query"] for r in batch], [r["text"] for r in batch],
                        padding=True, truncation="longest_first",
                        return_tensors="pt", max_length=args.max_length).to(device)
        out = esm(**tok, output_hidden_states=True)
        cls_emb = out.hidden_states[-1][:, 0, :]
        logits = head(torch.relu(cls_emb)).squeeze(-1)
        return logits

    @torch.no_grad()
    def evaluate() -> dict[str, float]:
        esm.eval()
        scores, labels = [], []
        total_loss = 0.0
        n = 0
        for i in range(0, len(dev_rows), args.batch_size):
            batch = dev_rows[i:i + args.batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = forward_probs(batch)
            y = torch.tensor([float(r["label"]) for r in batch], device=device)
            total_loss += loss_fn(logits.float(), y).item() * len(batch)
            n += len(batch)
            scores.extend(torch.sigmoid(logits.float()).cpu().tolist())
            labels.extend(int(r["label"]) for r in batch)
        esm.train()
        return {"dev_loss": total_loss / n, "dev_auprc": average_precision(labels, scores)}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_auprc = -1.0
    bad_epochs = 0
    step = 0
    rng = torch.Generator().manual_seed(args.seed)

    for epoch in range(args.epochs):
        order = torch.randperm(len(train_rows), generator=rng).tolist()
        running = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, len(order), args.batch_size):
            batch = [train_rows[j] for j in order[i:i + args.batch_size]]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = forward_probs(batch)
            y = torch.tensor([float(r["label"]) for r in batch], device=device)
            loss = loss_fn(logits.float(), y) / args.grad_accum
            loss.backward()
            running += loss.item() * len(batch) * args.grad_accum
            seen += len(batch)
            if (i // args.batch_size + 1) % args.grad_accum == 0 or i + args.batch_size >= len(order):
                torch.nn.utils.clip_grad_norm_(esm_params + list(head.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        if args.skip_eval:
            metrics = {}
        else:
            metrics = evaluate()
        record = {"epoch": epoch + 1, "train_loss": round(running / max(1, seen), 4),
                  **{k: round(v, 4) for k, v in metrics.items()}, "steps": step}
        history.append(record)
        print(json.dumps(record), flush=True)

        def save_checkpoint() -> None:
            esm_state = {f"esm_mask.{k}": v.detach().cpu() for k, v in esm.state_dict().items()}
            esm_state["classifier.weight"] = head.weight.detach().cpu()
            esm_state["classifier.bias"] = head.bias.detach().cpu()
            torch.save(esm_state, args.output_dir / "pytorch_model.bin")

        if args.fixed_epochs:
            save_checkpoint()
            continue
        if metrics["dev_auprc"] > best_auprc:
            best_auprc = metrics["dev_auprc"]
            bad_epochs = 0
            # Save in the published PLM-interact state-dict format so the
            # inference script can consume it unchanged.
            save_checkpoint()
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch + 1}", flush=True)
                break

    summary = {"best_dev_auprc": round(best_auprc, 4), "history": history,
               "train_rows": len(train_rows), "dev_rows": len(dev_rows),
               "freeze_esm": args.freeze_esm, "lr": args.lr,
               "head_lr_scale": args.head_lr_scale, "epochs_requested": args.epochs,
               "fixed_epochs": args.fixed_epochs, "skip_eval": args.skip_eval}
    (args.output_dir / "training_summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_dev_auprc": summary["best_dev_auprc"]}), flush=True)


if __name__ == "__main__":
    main()
