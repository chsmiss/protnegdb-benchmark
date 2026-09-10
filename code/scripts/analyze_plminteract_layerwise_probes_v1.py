#!/usr/bin/env python3
"""Four diagnostics on frozen pair-CLS: context/contact probes, orthogonal
complement, rho geometry, ESM-2 vs PLM-interact. No backbone training.
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
DEFAULT = ROOT / "data/interim/plminteract_layerwise_probes_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT)
    parser.add_argument("--plm-dir", type=Path, default=DEFAULT / "plminteract")
    parser.add_argument("--esm2-dir", type=Path, default=DEFAULT / "esm2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def load_lib():
    path = ROOT / "code/scripts" / "plminteract_layerwise_probe_lib_v1.py"
    spec = importlib.util.spec_from_file_location("plminteract_layerwise_probe_lib_v1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_cls(path: Path) -> tuple[dict[str, np.ndarray], list[int]]:
    data = np.load(path, allow_pickle=True)
    ids = [str(x) for x in data["directed_pair_id"]]
    layers = [int(x) for x in data["layers"]]
    cls = data["cls"]
    lookup = {ident: cls[i] for i, ident in enumerate(ids)}
    return lookup, layers


def vec(lookup: dict[str, np.ndarray], ident: str, layer_index: int) -> np.ndarray | None:
    block = lookup.get(ident)
    if block is None:
        return None
    return np.asarray(block[layer_index], dtype=np.float64)


def collect_triplets(rows: list[dict[str, str]], lookup, layer_index: int):
    kept = []
    for row in rows:
        ad = vec(lookup, row["ad_id"], layer_index)
        ac = vec(lookup, row["ac_id"], layer_index)
        ar = vec(lookup, row["ar_id"], layer_index)
        if ad is None or ac is None or ar is None:
            continue
        kept.append({**row, "ad": ad, "ac": ac, "ar": ar})
    return kept


def context_xy(trips):
    x, y, g = [], [], []
    for row in trips:
        x.extend([row["ad"], row["ac"], row["ar"]])
        y.extend([1, 1, 0])
        g.extend([row["triplet_id"]] * 3)
    return np.stack(x), np.asarray(y), np.asarray(g)


def contact_xy(trips):
    x, y, g = [], [], []
    for row in trips:
        x.extend([row["ad"], row["ac"]])
        y.extend([1, 0])
        g.extend([row["triplet_id"]] * 2)
    return np.stack(x), np.asarray(y), np.asarray(g)


def probe_split(lib, train, test, kind: str, lam: float, n_boot: int, seed: int) -> dict:
    if kind == "context":
        xtr, ytr, _ = context_xy(train)
        xte, yte, gte = context_xy(test)
    else:
        xtr, ytr, _ = contact_xy(train)
        xte, yte, gte = contact_xy(test)
    scores = lib.fit_ridge_scores(xtr, ytr, xte, lam=lam)
    out = lib.bootstrap_auroc(yte, scores, gte, n_boot=n_boot, seed=seed)
    out["task"] = kind
    out["n_train"] = int(len(train))
    out["n_test"] = int(len(test))
    return out


def rho_table(lib, trips) -> dict:
    values = [lib.rho_triplet(r["ad"], r["ac"], r["ar"]) for r in trips]
    cos_dc = [float(lib.cosine(r["ad"], r["ac"])) for r in trips]
    cos_dr = [float(lib.cosine(r["ad"], r["ar"])) for r in trips]
    out = lib.summarize(values)
    out["cosine_ad_ac"] = lib.summarize(cos_dc)
    out["cosine_ad_ar"] = lib.summarize(cos_dr)
    return out


def orthogonal_contact(lib, train, test, weight, lam, n_boot, seed) -> dict:
    w = np.asarray(weight, dtype=np.float64).reshape(-1)
    u = w / np.linalg.norm(w)

    def pack(trips, space: str):
        xs, ys, gs = [], [], []
        for row in trips:
            for vec, lab in ((row["ad"], 1), (row["ac"], 0)):
                _z_par, z_orth = lib.decompose_relu_cls(vec, w)
                if space == "full":
                    feat = lib.relu(vec)
                elif space == "parallel":
                    feat = np.asarray([float(np.dot(lib.relu(vec), u))], dtype=np.float64)
                else:
                    feat = z_orth
                xs.append(np.asarray(feat, dtype=np.float64).reshape(-1))
                ys.append(lab)
                gs.append(row["triplet_id"])
        return np.stack(xs), np.asarray(ys), np.asarray(gs)

    report = {}
    for space in ("full", "parallel", "orthogonal"):
        xtr, ytr, _ = pack(train, space)
        xte, yte, gte = pack(test, space)
        scores = lib.fit_ridge_scores(xtr, ytr, xte, lam=lam)
        block = lib.bootstrap_auroc(yte, scores, gte, n_boot=n_boot, seed=seed)
        block["dim"] = int(xtr.shape[1])
        report[space] = block
    return report


def layer_name(index: int, last: int) -> str:
    if index == 0:
        return "L0_embed"
    if index == last:
        return f"L{index}_last"
    return f"L{index}"


def main() -> None:
    args = parse_args()
    lib = load_lib()
    trips = list(csv.DictReader((args.prepare_dir / "structure_triplets_v1.tsv").open(
        encoding="utf-8", newline=""), delimiter="\t"))
    train_meta = [r for r in trips if r["split"] == "train_eligible"]
    test_meta = [r for r in trips if r["split"] == "family_unseen"]
    if not lib.families_disjoint(train_meta, test_meta):
        raise SystemExit("family leakage in prepared triplets")

    models = {}
    for name, directory in (("plm", args.plm_dir), ("esm2", args.esm2_dir)):
        lookup, layers = load_cls(directory / "layer_cls_v1.npz")
        models[name] = {"lookup": lookup, "layers": layers}

    if models["plm"]["layers"] != models["esm2"]["layers"]:
        raise SystemExit("PLM and ESM-2 layer lists differ")
    layers = models["plm"]["layers"]
    last = max(layers)

    weight_path = args.plm_dir / "classifier_weight_v1.npy"
    weight = np.load(weight_path) if weight_path.exists() else None

    by_model = {}
    for model_name, payload in models.items():
        lookup = payload["lookup"]
        layer_rows = []
        for li, layer in enumerate(layers):
            train = collect_triplets(train_meta, lookup, li)
            test = collect_triplets(test_meta, lookup, li)
            context = probe_split(lib, train, test, "context", args.lam, args.n_boot, args.seed)
            contact = probe_split(lib, train, test, "contact", args.lam, args.n_boot, args.seed + 1)
            rho_test = rho_table(lib, test)
            rho_train = rho_table(lib, train)
            rec = {
                "layer": layer,
                "layer_name": layer_name(layer, last),
                "n_train": len(train),
                "n_test": len(test),
                "context": context,
                "contact": contact,
                "rho_family_unseen": rho_test,
                "rho_train_eligible": rho_train,
            }
            if model_name == "plm" and weight is not None and layer == last:
                rec["orthogonal_contact"] = orthogonal_contact(
                    lib, train, test, weight, args.lam, args.n_boot, args.seed + 2)
            layer_rows.append(rec)
        by_model[model_name] = layer_rows

    delta = []
    plm_of = {r["layer"]: r for r in by_model["plm"]}
    esm_of = {r["layer"]: r for r in by_model["esm2"]}
    for layer in layers:
        p, e = plm_of[layer], esm_of[layer]
        delta.append({
            "layer": layer,
            "layer_name": p["layer_name"],
            "delta_context": p["context"]["auroc"] - e["context"]["auroc"],
            "delta_contact": p["contact"]["auroc"] - e["contact"]["auroc"],
            "plm_context": p["context"]["auroc"],
            "esm2_context": e["context"]["auroc"],
            "plm_contact": p["contact"]["auroc"],
            "esm2_contact": e["contact"]["auroc"],
            "plm_cos_ad_ac": p["rho_family_unseen"]["cosine_ad_ac"]["mean"],
            "esm2_cos_ad_ac": e["rho_family_unseen"]["cosine_ad_ac"]["mean"],
            "plm_cos_ad_ar": p["rho_family_unseen"]["cosine_ad_ar"]["mean"],
            "esm2_cos_ad_ar": e["rho_family_unseen"]["cosine_ad_ar"]["mean"],
            "plm_rho": p["rho_family_unseen"]["mean"],
            "esm2_rho": e["rho_family_unseen"]["mean"],
        })

    summary = {
        "status": "plminteract_layerwise_probes_v1",
        "distance": "cosine_distance = 1 - cos on L2-normalized pair CLS",
        "probe": "standardized ridge (lambda=%.3g) on frozen CLS; train_eligible vs family_unseen" % args.lam,
        "n_train_triplets": len(train_meta),
        "n_test_triplets": len(test_meta),
        "layers": layers,
        "models": by_model,
        "delta_plm_minus_esm2": delta,
        "literature_footnote": (
            "Codex literature-negative FPR uses Negatome cleaned direct n=443 "
            "(median 0.0006, FPR 24.6%) and ProtNeg-104 direct-assay pairs from "
            "480 papers (median 0.0079, FPR 25.0%). It is not the 1000-pilot "
            "700 matched-neighbor C partners (FPR 0.733)."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "layerwise_probes_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    tsv_path = args.output_dir / "layerwise_probes_metrics_v1.tsv"
    tsv_fields = [
        "layer", "layer_name",
        "plm_context", "plm_context_lo", "plm_context_hi",
        "esm2_context", "delta_context",
        "plm_contact", "plm_contact_lo", "plm_contact_hi",
        "esm2_contact", "delta_contact",
        "plm_rho", "esm2_rho",
        "plm_cos_ad_ac", "esm2_cos_ad_ac",
        "plm_cos_ad_ar", "esm2_cos_ad_ar",
        "n_test",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tsv_fields, delimiter="\t")
        writer.writeheader()
        for row in delta:
            p = plm_of[row["layer"]]
            e = esm_of[row["layer"]]
            writer.writerow({
                "layer": row["layer"],
                "layer_name": row["layer_name"],
                "plm_context": f"{p['context']['auroc']:.6f}",
                "plm_context_lo": f"{p['context']['ci95_lo']:.6f}",
                "plm_context_hi": f"{p['context']['ci95_hi']:.6f}",
                "esm2_context": f"{e['context']['auroc']:.6f}",
                "delta_context": f"{row['delta_context']:.6f}",
                "plm_contact": f"{p['contact']['auroc']:.6f}",
                "plm_contact_lo": f"{p['contact']['ci95_lo']:.6f}",
                "plm_contact_hi": f"{p['contact']['ci95_hi']:.6f}",
                "esm2_contact": f"{e['contact']['auroc']:.6f}",
                "delta_contact": f"{row['delta_contact']:.6f}",
                "plm_rho": f"{row['plm_rho']:.6f}",
                "esm2_rho": f"{row['esm2_rho']:.6f}",
                "plm_cos_ad_ac": f"{row['plm_cos_ad_ac']:.6f}",
                "esm2_cos_ad_ac": f"{row['esm2_cos_ad_ac']:.6f}",
                "plm_cos_ad_ar": f"{row['plm_cos_ad_ar']:.6f}",
                "esm2_cos_ad_ar": f"{row['esm2_cos_ad_ar']:.6f}",
                "n_test": p["n_test"],
            })

    last_plm = plm_of[last]
    orth = last_plm.get("orthogonal_contact", {})
    lines = [
        "# PLM-interact 层间冻结探针 v1",
        "",
        "结构 Figure 3：train_eligible 训练线性探针，family_unseen 测试。",
        "不是 1000-pilot 文献匹配 C。Codex 文献阴性 FPR 见文末脚注。",
        "",
        f"n_train={len(train_meta)}  n_test={len(test_meta)}  ridge λ={args.lam}",
        "",
        "| layer | PLM context | ESM-2 context | Δctx | PLM contact | ESM-2 contact | Δcon | PLM ρ | ESM-2 ρ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in delta:
        lines.append(
            f"| {row['layer_name']} | {row['plm_context']:.3f} | {row['esm2_context']:.3f} | "
            f"{row['delta_context']:+.3f} | {row['plm_contact']:.3f} | {row['esm2_contact']:.3f} | "
            f"{row['delta_contact']:+.3f} | {row['plm_rho']:.3f} | {row['esm2_rho']:.3f} |"
        )
    if orth:
        lines += [
            "",
            "## 分析2 最后一层正交补（family-unseen AD vs AC）",
            "",
            f"- full z=ReLU(CLS) AUROC {orth['full']['auroc']:.3f} "
            f"[{orth['full']['ci95_lo']:.3f}, {orth['full']['ci95_hi']:.3f}]",
            f"- parallel 1D AUROC {orth['parallel']['auroc']:.3f} "
            f"[{orth['parallel']['ci95_lo']:.3f}, {orth['parallel']['ci95_hi']:.3f}]",
            f"- orthogonal AUROC {orth['orthogonal']['auroc']:.3f} "
            f"[{orth['orthogonal']['ci95_lo']:.3f}, {orth['orthogonal']['ci95_hi']:.3f}]",
        ]
        if orth["orthogonal"]["auroc"] <= 0.55:
            lines.append(
                "正交空间也分不开：可迁移的直接接触轴在表征里基本没形成，不只是分类头没用。"
            )
        else:
            lines.append(
                "正交空间仍能分 AD/AC：表征里有接触信息，官方头没用上。"
            )
    lines += [
        "",
        "## 脚注：文献阴性口径",
        "",
        summary["literature_footnote"],
    ]
    (args.output_dir / "layerwise_probes_summary_v1.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir / 'layerwise_probes_summary_v1.json'}", flush=True)


if __name__ == "__main__":
    main()
