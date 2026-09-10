#!/usr/bin/env python3
"""Four diagnostics on frozen MINT pair embeddings vs independent ESM-2 concat.

Reuse the Figure 3 family-disjoint triplets prepared for PLM-interact.
Pair vector is concat(mean-pool A, mean-pool B), not pair CLS.
No backbone training.
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
DEFAULT = ROOT / "data/interim/mint_layerwise_probes_v1"
TRIPLETS = ROOT / "data/interim/plminteract_layerwise_probes_v1/structure_triplets_v1.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", type=Path, default=TRIPLETS)
    parser.add_argument("--mint-dir", type=Path, default=DEFAULT / "mint")
    parser.add_argument("--esm2-dir", type=Path, default=DEFAULT / "esm2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def load(name: str):
    path = ROOT / "code/scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def layer_name(index: int, last: int) -> str:
    if index == 0:
        return "L0_embed"
    if index == last:
        return f"L{index}_last"
    return f"L{index}"


def orthogonal_mlp(lib, ext, train, test, w1, b1, w2, lam, n_boot, seed) -> dict:
    w2 = np.asarray(w2, dtype=np.float64).reshape(-1)
    u = w2 / np.linalg.norm(w2)

    def pack(trips, space: str):
        xs, ys, gs = [], [], []
        for row in trips:
            for vec, lab in ((row["ad"], 1), (row["ac"], 0)):
                z = ext.mlp_hidden(vec, w1, b1)
                if space == "raw":
                    feat = np.asarray(vec, dtype=np.float64).reshape(-1)
                elif space == "full":
                    feat = z
                elif space == "parallel":
                    feat = np.asarray([float(np.dot(z, u))], dtype=np.float64)
                else:
                    _par, orth = lib.decompose_relu_cls(z, w2)
                    feat = orth
                xs.append(feat.reshape(-1))
                ys.append(lab)
                gs.append(row["triplet_id"])
        return np.stack(xs), np.asarray(ys), np.asarray(gs)

    report = {}
    for space in ("raw", "full", "parallel", "orthogonal"):
        xtr, ytr, _ = pack(train, space)
        xte, yte, gte = pack(test, space)
        scores = lib.fit_ridge_scores(xtr, ytr, xte, lam=lam)
        block = lib.bootstrap_auroc(yte, scores, gte, n_boot=n_boot, seed=seed)
        block["dim"] = int(xtr.shape[1])
        report[space] = block
    return report


def main() -> None:
    args = parse_args()
    lib = load("plminteract_layerwise_probe_lib_v1.py")
    an = load("analyze_plminteract_layerwise_probes_v1.py")
    ext = load("extract_mint_layerwise_probes_v1.py")
    trips = list(csv.DictReader(args.triplets.open(encoding="utf-8", newline=""), delimiter="\t"))
    train_meta = [r for r in trips if r["split"] == "train_eligible"]
    test_meta = [r for r in trips if r["split"] == "family_unseen"]
    if not lib.families_disjoint(train_meta, test_meta):
        raise SystemExit("family leakage in triplets")

    models = {}
    for name, directory in (("mint", args.mint_dir), ("esm2", args.esm2_dir)):
        lookup, layers = an.load_cls(directory / "layer_cls_v1.npz")
        models[name] = {"lookup": lookup, "layers": layers}
    if models["mint"]["layers"] != models["esm2"]["layers"]:
        raise SystemExit("MINT and ESM-2 layer lists differ")
    layers = models["mint"]["layers"]
    last = max(layers)

    w1 = np.load(args.mint_dir / "mlp_w1_v1.npy")
    b1 = np.load(args.mint_dir / "mlp_b1_v1.npy")
    w2 = np.load(args.mint_dir / "mlp_w2_v1.npy")

    by_model = {}
    for model_name, payload in models.items():
        lookup = payload["lookup"]
        layer_rows = []
        for li, layer in enumerate(layers):
            train = an.collect_triplets(train_meta, lookup, li)
            test = an.collect_triplets(test_meta, lookup, li)
            rec = {
                "layer": layer,
                "layer_name": layer_name(layer, last),
                "n_train": len(train),
                "n_test": len(test),
                "context": an.probe_split(lib, train, test, "context", args.lam, args.n_boot, args.seed),
                "contact": an.probe_split(lib, train, test, "contact", args.lam, args.n_boot, args.seed + 1),
                "rho_family_unseen": an.rho_table(lib, test),
            }
            if model_name == "mint" and layer == last:
                rec["orthogonal_contact"] = orthogonal_mlp(
                    lib, ext, train, test, w1, b1, w2, args.lam, args.n_boot, args.seed + 2)
            layer_rows.append(rec)
        by_model[model_name] = layer_rows

    mint_of = {r["layer"]: r for r in by_model["mint"]}
    esm_of = {r["layer"]: r for r in by_model["esm2"]}
    delta = []
    for layer in layers:
        m, e = mint_of[layer], esm_of[layer]
        delta.append({
            "layer": layer,
            "layer_name": m["layer_name"],
            "delta_context": m["context"]["auroc"] - e["context"]["auroc"],
            "delta_contact": m["contact"]["auroc"] - e["contact"]["auroc"],
            "mint_context": m["context"]["auroc"],
            "esm2_context": e["context"]["auroc"],
            "mint_contact": m["contact"]["auroc"],
            "esm2_contact": e["contact"]["auroc"],
            "mint_cos_ad_ac": m["rho_family_unseen"]["cosine_ad_ac"]["mean"],
            "esm2_cos_ad_ac": e["rho_family_unseen"]["cosine_ad_ac"]["mean"],
            "mint_cos_ad_ar": m["rho_family_unseen"]["cosine_ad_ar"]["mean"],
            "esm2_cos_ad_ar": e["rho_family_unseen"]["cosine_ad_ar"]["mean"],
            "mint_rho": m["rho_family_unseen"]["mean"],
            "esm2_rho": e["rho_family_unseen"]["mean"],
        })

    summary = {
        "status": "mint_layerwise_probes_v1",
        "pair_vector": "concat(mean-pool chain A, mean-pool chain B) after MINT crop-512 seed 13",
        "probe": "standardized ridge on frozen pair vectors; train_eligible vs family_unseen",
        "n_train_triplets": len(train_meta),
        "n_test_triplets": len(test_meta),
        "layers": layers,
        "models": by_model,
        "delta_mint_minus_esm2": delta,
        "literature_footnote": (
            "Codex literature-negative FPR uses Negatome 443 and ProtNeg-104 from "
            "480 papers. This cohort is structural AD/AC/AR, not those assay negatives."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "layerwise_probes_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    tsv_fields = [
        "layer", "layer_name",
        "mint_context", "mint_context_lo", "mint_context_hi",
        "esm2_context", "delta_context",
        "mint_contact", "mint_contact_lo", "mint_contact_hi",
        "esm2_contact", "delta_contact",
        "mint_rho", "esm2_rho",
        "mint_cos_ad_ac", "esm2_cos_ad_ac",
        "mint_cos_ad_ar", "esm2_cos_ad_ar",
        "n_test",
    ]
    with (args.output_dir / "layerwise_probes_metrics_v1.tsv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tsv_fields, delimiter="\t")
        writer.writeheader()
        for row in delta:
            m = mint_of[row["layer"]]
            e = esm_of[row["layer"]]
            writer.writerow({
                "layer": row["layer"],
                "layer_name": row["layer_name"],
                "mint_context": f"{m['context']['auroc']:.6f}",
                "mint_context_lo": f"{m['context']['ci95_lo']:.6f}",
                "mint_context_hi": f"{m['context']['ci95_hi']:.6f}",
                "esm2_context": f"{e['context']['auroc']:.6f}",
                "delta_context": f"{row['delta_context']:.6f}",
                "mint_contact": f"{m['contact']['auroc']:.6f}",
                "mint_contact_lo": f"{m['contact']['ci95_lo']:.6f}",
                "mint_contact_hi": f"{m['contact']['ci95_hi']:.6f}",
                "esm2_contact": f"{e['contact']['auroc']:.6f}",
                "delta_contact": f"{row['delta_contact']:.6f}",
                "mint_rho": f"{row['mint_rho']:.6f}",
                "esm2_rho": f"{row['esm2_rho']:.6f}",
                "mint_cos_ad_ac": f"{row['mint_cos_ad_ac']:.6f}",
                "esm2_cos_ad_ac": f"{row['esm2_cos_ad_ac']:.6f}",
                "mint_cos_ad_ar": f"{row['mint_cos_ad_ar']:.6f}",
                "esm2_cos_ad_ar": f"{row['esm2_cos_ad_ar']:.6f}",
                "n_test": m["n_test"],
            })

    last_mint = mint_of[last]
    orth = last_mint.get("orthogonal_contact", {})
    lines = [
        "# MINT 层间冻结探针 v1",
        "",
        "同一套 Figure 3 结构三元组（train_eligible / family_unseen）。",
        "pair 向量是两条链 mean-pool 拼接，不是 pair CLS。",
        "",
        f"n_train={len(train_meta)}  n_test={len(test_meta)}  ridge λ={args.lam}",
        "",
        "| layer | MINT context | ESM-2 context | Δctx | MINT contact | ESM-2 contact | Δcon | MINT ρ | ESM-2 ρ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in delta:
        lines.append(
            f"| {row['layer_name']} | {row['mint_context']:.3f} | {row['esm2_context']:.3f} | "
            f"{row['delta_context']:+.3f} | {row['mint_contact']:.3f} | {row['esm2_contact']:.3f} | "
            f"{row['delta_contact']:+.3f} | {row['mint_rho']:.3f} | {row['esm2_rho']:.3f} |"
        )
    if orth:
        lines += [
            "",
            "## 分析2 MLP 隐空间正交补（family-unseen AD vs AC）",
            "",
            f"- raw concat AUROC {orth['raw']['auroc']:.3f} "
            f"[{orth['raw']['ci95_lo']:.3f}, {orth['raw']['ci95_hi']:.3f}]",
            f"- full z=ReLU(W1h+b1) AUROC {orth['full']['auroc']:.3f} "
            f"[{orth['full']['ci95_lo']:.3f}, {orth['full']['ci95_hi']:.3f}]",
            f"- parallel 1D AUROC {orth['parallel']['auroc']:.3f} "
            f"[{orth['parallel']['ci95_lo']:.3f}, {orth['parallel']['ci95_hi']:.3f}]",
            f"- orthogonal AUROC {orth['orthogonal']['auroc']:.3f} "
            f"[{orth['orthogonal']['ci95_lo']:.3f}, {orth['orthogonal']['ci95_hi']:.3f}]",
        ]
        if orth["orthogonal"]["auroc"] <= 0.55 and orth["raw"]["auroc"] <= 0.55:
            lines.append(
                "原始拼接向量和正交补都分不开：可迁移接触轴在 MINT pair embedding 里没有形成。"
            )
        elif orth["orthogonal"]["auroc"] > 0.55:
            lines.append("正交空间仍能分 AD/AC：表征有接触信息，Bernett 头没用上。")
        else:
            lines.append("原始 concat 与正交补结论不一致，见 JSON。")
    lines += ["", "## 脚注", "", summary["literature_footnote"]]
    (args.output_dir / "layerwise_probes_summary_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'layerwise_probes_summary_v1.json'}", flush=True)


if __name__ == "__main__":
    main()
