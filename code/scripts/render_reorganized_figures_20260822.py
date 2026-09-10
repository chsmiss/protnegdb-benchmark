#!/usr/bin/env python3
"""Render the 2026-08-22 reorganized ProtNegDB main and supplementary figures.

This module is deliberately downstream-only.  It reads locked local tables and
JSON ledgers, performs no model inference or training, and writes PDF/PNG/SVG
figures plus a compact statistics ledger.  Figure 1 is intentionally excluded;
the author-supplied Figure 1 is copied verbatim during packaging.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "deliverables/manuscript_revision_20260822/figures"

COL = {
    "ink": "#17212B",
    "muted": "#66727E",
    "grid": "#D7DEE5",
    "blue": "#2F6B9A",
    "blue_light": "#DCEAF4",
    "green": "#14866D",
    "green_light": "#D8EEE8",
    "orange": "#B26D16",
    "orange_light": "#F4E7CF",
    "red": "#D55E45",
    "red_light": "#F7E1DB",
    "purple": "#6C5AA7",
    "purple_light": "#E8E3F4",
    "grey": "#7B8792",
    "grey_light": "#E7EAED",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--only",
        choices=("all", "Figure3", "Figure4", "Figure5", "S12"),
        default="all",
    )
    return p.parse_args()


def style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.titlesize": 8.3,
        "axes.titleweight": "normal",
        "axes.labelsize": 7.2,
        "axes.edgecolor": COL["ink"],
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "text.color": COL["ink"],
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def panel(ax, label: str, title: str) -> None:
    ax.text(-0.10, 1.08, label, transform=ax.transAxes, fontsize=11,
            fontweight="normal", va="top", ha="left", clip_on=False)
    ax.set_title(title, loc="left", pad=8, fontsize=7.2)


def clean(ax, grid: str | None = None) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=COL["grid"], linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)


def save_all(fig, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(outdir / f"{stem}.{suffix}", dpi=400)
    plt.close(fig)


def point_ci(ax, x, lo, hi, y, color, marker="o", size=36, zorder=3):
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], fmt="none",
                ecolor=COL["ink"], elinewidth=0.85, capsize=2, zorder=zorder - 1)
    ax.scatter([x], [y], s=size, color=color, marker=marker,
               edgecolor="white", linewidth=0.6, zorder=zorder)


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)])
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def figure3(outdir: Path) -> dict:
    """Representation probes and common-set structure-model comparison."""
    plm = pd.read_csv(
        ROOT / "data/interim/plminteract_layerwise_probes_v1/layerwise_probes_metrics_v1.tsv",
        sep="\t",
    )
    mint = pd.read_csv(
        ROOT / "data/interim/mint_layerwise_probes_v1/layerwise_probes_metrics_v1.tsv",
        sep="\t",
    )
    missing = read_json(
        ROOT / "data/interim/manuscript_missing_numbers_v1/manuscript_missing_numbers_v1.json"
    )["figure3_final_layer_probes"]
    locked = read_json(
        ROOT / "data/interim/af3_partner_ranking_v1/locked300_all_models_v1/summary_v1.json"
    )["all_model_complete"]
    trip = pd.read_csv(
        ROOT / "data/interim/af3_partner_ranking_v1/locked300_all_models_v1/triplet_scores_v1.tsv",
        sep="\t",
    )

    fig = plt.figure(figsize=(7.2, 7.45))
    gs = fig.add_gridspec(2, 6, left=0.07, right=0.985, bottom=0.07,
                          top=0.88, wspace=0.9, hspace=0.50)
    fig.text(0.055, 0.965,
             "Figure 3 | PPI adaptation preserves context ambiguity, whereas AlphaFold 3 adds partial partner resolution",
             fontsize=9.7, fontweight="normal", va="top")
    fig.text(0.055, 0.938,
             "Frozen family-unseen probes (n=540) • locked seven-model comparison (300 triplets; 256 anchors)",
             fontsize=6.9, color=COL["muted"], va="top")

    # a: PLM-Interact layer-wise probes
    ax = fig.add_subplot(gs[0, :2])
    panel(ax, "a", "PLM-Interact: context rises; contact does not")
    ax.plot(plm.layer, plm.plm_context, color=COL["blue"], lw=1.8, label="Context, PLM-Interact")
    ax.fill_between(plm.layer, plm.plm_context_lo, plm.plm_context_hi,
                    color=COL["blue"], alpha=0.13, linewidth=0)
    ax.plot(plm.layer, plm.plm_contact, color=COL["red"], lw=1.6, label="Contact, PLM-Interact")
    ax.fill_between(plm.layer, plm.plm_contact_lo, plm.plm_contact_hi,
                    color=COL["red"], alpha=0.12, linewidth=0)
    ax.plot(plm.layer, plm.esm2_context, color=COL["blue"], lw=1.1, ls="--", label="Context, ESM-2")
    ax.plot(plm.layer, plm.esm2_contact, color=COL["red"], lw=1.1, ls="--", label="Contact, ESM-2")
    ax.axhline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set(xlabel="Transformer layer", ylabel="Family-unseen AUROC", ylim=(0.44, 0.76), xlim=(0, 33))
    clean(ax, "y")
    ax.legend(loc="upper left", ncol=1, handlelength=2.2)

    # b: MINT layer-wise probes
    ax = fig.add_subplot(gs[0, 2:4])
    panel(ax, "b", "MINT: contact remains near chance")
    ax.plot(mint.layer, mint.mint_context, color=COL["blue"], lw=1.8, label="Context, MINT")
    ax.fill_between(mint.layer, mint.mint_context_lo, mint.mint_context_hi,
                    color=COL["blue"], alpha=0.13, linewidth=0)
    ax.plot(mint.layer, mint.mint_contact, color=COL["red"], lw=1.6, label="Contact, MINT")
    ax.fill_between(mint.layer, mint.mint_contact_lo, mint.mint_contact_hi,
                    color=COL["red"], alpha=0.12, linewidth=0)
    ax.plot(mint.layer, mint.esm2_context, color=COL["blue"], lw=1.1, ls="--", label="Context, ESM-2")
    ax.plot(mint.layer, mint.esm2_contact, color=COL["red"], lw=1.1, ls="--", label="Contact, ESM-2")
    ax.axhline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set(xlabel="Transformer layer", ylabel="Family-unseen AUROC", ylim=(0.38, 0.64), xlim=(0, 33))
    clean(ax, "y")
    ax.legend(loc="upper right", ncol=1, handlelength=2.2)

    # c: confirmatory final-layer forest
    ax = fig.add_subplot(gs[0, 4:])
    panel(ax, "c", "Final-layer probes")
    probe_specs = [
        ("ESM-2\npair CLS", missing["ESM-2_pairCLS"]),
        ("PLM-Interact\npair CLS", missing["PLM-Interact_pairCLS"]),
        ("MINT\nconcat mean-pool", missing["MINT_concat_meanpool"]),
    ]
    y = np.arange(3)[::-1]
    for yi, (label, block) in zip(y, probe_specs):
        point_ci(ax, block["context_auroc"], *block["context_ci95"], yi + 0.13, COL["blue"], marker="o")
        point_ci(ax, block["contact_auroc"], *block["contact_ci95"], yi - 0.13, COL["red"], marker="s")
    ax.axvline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y); ax.set_yticklabels([x[0] for x in probe_specs])
    ax.set(xlabel="AUROC (triplet-bootstrap 95% CI)", xlim=(0.44, 0.75), ylim=(-0.6, 2.6))
    clean(ax, "x")
    ax.scatter([], [], color=COL["blue"], marker="o", label="Context")
    ax.scatter([], [], color=COL["red"], marker="s", label="Direct contact")
    ax.legend(loc="lower right")

    # d: seven-model common-set comparison
    ax = fig.add_subplot(gs[1, :2])
    panel(ax, "d", "Common-set partner ranking")
    order = ["AF3", "ESMFold", "PLM-Interact", "MINT", "D-SCRIPT", "Topsy-Turvy", "SPRINT"]
    yy = np.arange(len(order))[::-1]
    for yi, model in zip(yy, order):
        b = locked["models"][model]
        color = COL["green"] if model == "AF3" else COL["grey"]
        point_ci(ax, b["P_D_gt_C"], *b["anchor_block_ci95"], yi, color, size=42 if model == "AF3" else 30)
        ax.text(b["anchor_block_ci95"][1] + 0.006, yi, f"{100*b['P_D_gt_C']:.1f}%", va="center", fontsize=5.7,
                color=color, fontweight="normal")
    ax.axvline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(yy); ax.set_yticklabels(order)
    ax.set(xlabel="P[s(A,D) > s(A,C)]", xlim=(0.41, 0.72), ylim=(-0.7, len(order)-0.3))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    clean(ax, "x")

    # e: paired AF3 advantage
    ax = fig.add_subplot(gs[1, 2:4])
    panel(ax, "e", "AF3 paired advantage")
    comps = ["PLM-Interact", "MINT", "D-SCRIPT", "Topsy-Turvy", "SPRINT", "ESMFold"]
    yy = np.arange(len(comps))[::-1]
    for yi, model in zip(yy, comps):
        b = locked["paired_delta_P"][f"AF3_minus_{model}"]
        color = COL["green"] if b["excludes_zero"] else COL["orange"]
        point_ci(ax, b["delta_P"], *b["anchor_block_ci95"], yi, color, size=34)
    ax.axvline(0, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(yy); ax.set_yticklabels(comps)
    ax.set(xlabel="AF3 − comparator in P(D>C)", xlim=(-0.035, 0.235), ylim=(-0.7, len(comps)-0.3))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    clean(ax, "x")
    ax.text(0.98, 0.02, "green: 95% CI excludes 0", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.5, color=COL["muted"])

    # f: paired AF3 ipTM
    ax = fig.add_subplot(gs[1, 4:])
    panel(ax, "f", "Paired AlphaFold 3 ipTM")
    win = trip["AF3_ad"] > trip["AF3_ac"]
    ax.scatter(trip.loc[~win, "AF3_ac"], trip.loc[~win, "AF3_ad"], s=13,
               color=COL["red"], alpha=0.65, edgecolor="none", label="C at least as high")
    ax.scatter(trip.loc[win, "AF3_ac"], trip.loc[win, "AF3_ad"], s=13,
               color=COL["green"], alpha=0.72, edgecolor="none", label="D ranked above C")
    ax.plot([0, 1], [0, 1], color=COL["grey"], lw=0.8, ls=(0, (2, 2)))
    ax.axhline(0.6, color=COL["green"], alpha=0.55, lw=0.8, ls=(0, (3, 2)))
    ax.set(xlabel="ipTM, A–C assembly noncontact", ylabel="ipTM, A–D direct contact",
           xlim=(-0.02, 1.0), ylim=(-0.02, 1.0))
    clean(ax, "both")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.04, "P(D>C)=63.0%\n95% CI 57.2–68.6%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.1, fontweight="normal", color=COL["green"])

    stem = "Figure3_representation_and_structure_models_20260822"
    save_all(fig, outdir / "main", stem)
    return {
        "stem": stem,
        "probe_test_triplets": 540,
        "common_triplets": locked["n_triplets"],
        "common_anchors": locked["n_anchors"],
        "af3_p_d_gt_c": locked["models"]["AF3"]["P_D_gt_C"],
    }


def figure4(outdir: Path) -> dict:
    """Negative semantics, score drivers and familiarity-matched comparison."""
    f34 = read_json(ROOT / "data/reported_stats/figure34_render_stats_v1.json")["figure4"]
    drivers = read_json(ROOT / "data/interim/figure5_structure_score_drivers_v2/figure5_score_drivers_v2.json")
    matching = read_json(ROOT / "data/interim/revision_v2_matching_fig5/matching_fig5_robustness_v2.json")
    features = pd.read_csv(ROOT / "data/interim/figure5_structure_score_drivers_v2/figure5_ac_features_v2.tsv", sep="\t")

    fig = plt.figure(figsize=(7.2, 7.05))
    gs = fig.add_gridspec(2, 6, left=0.07, right=0.985, bottom=0.075,
                          top=0.875, wspace=0.95, hspace=0.48)
    fig.text(0.055, 0.965,
             "Figure 4 | Experimental non-binding and assembly noncontact remain distinct after accounting for familiarity",
             fontsize=9.6, fontweight="normal", va="top")
    fig.text(0.055, 0.938,
             "Published PLM-Interact checkpoint • harmonized assay ontology • structural clean core n=2,480",
             fontsize=6.9, color=COL["muted"], va="top")

    # a: funnel
    ax = fig.add_subplot(gs[0, :2])
    panel(ax, "a", "Negatome common-QC funnel")
    stages = f34["funnel"][:-1]
    names = ["Manual\nscored", "Stringent", "Human", "Canonical", "No\ntruncation", "Common\nQC"]
    vals = [x["n"] for x in stages]
    y = np.arange(len(vals))[::-1]
    widths = np.asarray(vals) / max(vals)
    for yi, w, n, name in zip(y, widths, vals, names):
        ax.barh(yi, w, height=0.62, color=COL["orange_light"], edgecolor=COL["orange"], lw=0.8)
        ax.text(0.02, yi, name, va="center", ha="left", fontsize=5.6)
        ax.text(w + 0.025, yi, f"{n:,}", va="center", fontsize=6.0, fontweight="normal")
    ax.set_xlim(0, 1.18); ax.set_ylim(-0.6, len(vals)-0.4); ax.axis("off")
    ax.text(0.02, -0.38, "Common QC: 298 reconstituted; 145 cell-binary;\n119 co-complex; 22 unresolved.",
            fontsize=5.0, color=COL["muted"], va="top")

    # b: score-positive fractions
    ax = fig.add_subplot(gs[0, 2:])
    panel(ax, "b", "Score regimes depend on evidence semantics")
    order = ["Random A–R", "Reconstituted direct", "Cell binary / proximity",
             "Co-complex association", "Structural clean non-contact"]
    colors = [COL["grey"], COL["orange"], COL["blue"], COL["purple"], COL["red"]]
    labels = ["Random unlabeled", "Reconstituted\ndirect", "Cell binary /\nproximity",
              "Co-complex\nassociation", "Assembly\nnoncontact"]
    x = np.arange(len(order))
    vals = [f34["groups"][k]["frac_ge_0.5"] for k in order]
    lo = [f34["groups"][k]["frac_ge_0.5_ci95"][0] for k in order]
    hi = [f34["groups"][k]["frac_ge_0.5_ci95"][1] for k in order]
    ax.bar(x, vals, color=colors, alpha=0.88, width=0.68, zorder=2)
    ax.errorbar(x, vals, yerr=[np.asarray(vals)-np.asarray(lo), np.asarray(hi)-np.asarray(vals)],
                fmt="none", ecolor=COL["ink"], capsize=2, lw=0.8, zorder=3)
    for xi, v, key in zip(x, vals, order):
        ax.text(xi, v + 0.045, f"{100*v:.1f}%\nn={f34['groups'][key]['n']:,}",
                ha="center", va="bottom", fontsize=5.5, fontweight="normal")
    ax.axhline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set(ylabel="Fraction with predicted score ≥0.5", ylim=(0, 0.9))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean(ax, "y")

    # c: familiarity bins
    ax = fig.add_subplot(gs[1, :2])
    panel(ax, "c", "Training familiarity")
    bins = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    block = drivers["5a_training_exposure"]["by_endpoint_train_familiarity"]
    vals = [block[b]["frac_ge_0.5"] for b in bins]
    los = [block[b]["ci95"][0] for b in bins]
    his = [block[b]["ci95"][1] for b in bins]
    x = np.arange(len(bins))
    ax.plot(x, vals, color=COL["blue"], marker="o", lw=1.5, ms=4.5)
    ax.errorbar(x, vals, yerr=[np.asarray(vals)-np.asarray(los), np.asarray(his)-np.asarray(vals)],
                fmt="none", ecolor=COL["blue"], capsize=2, lw=0.8)
    ax.axhline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels(["<20 /\nno hit", "20–40", "40–60", "60–80", "80–100"])
    ax.set(xlabel="Minimum endpoint identity to positive-training proteins (%)",
           ylabel="Fraction score ≥0.5", ylim=(0.48, 1.02))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean(ax, "y")
    ax.text(0.02, 0.02, "Descriptive bins are not strictly monotonic.", transform=ax.transAxes,
            fontsize=5.4, color=COL["muted"])

    # d: adjusted regression forest
    ax = fig.add_subplot(gs[1, 2:4])
    panel(ax, "d", "Adjusted score correlates")
    coef = [x for x in drivers["5e_multivariable"]["coefficients"] if x["term"] != "intercept"]
    display = {
        "train_familiarity": "Training familiarity",
        "ac_3mer_similarity": "A–C sequence similarity",
        "log_complex_size": "Complex size",
        "log1p_replicate_pdb": "Repeated-PDB support",
        "log_min_heavy": "Minimum heavy-atom distance",
        "log_pair_length": "Pair sequence length",
        "log1p_bridge_contacts": "Bridge-interface contacts",
        "human": "Human–human pair",
    }
    order_terms = list(display)
    by = {x["term"]: x for x in coef}
    y = np.arange(len(order_terms))[::-1]
    for yi, term in zip(y, order_terms):
        b = by[term]; color = COL["blue"] if b["coef_log_odds"] >= 0 else COL["red"]
        point_ci(ax, b["coef_log_odds"], *b["ci95"], yi, color, size=28)
    ax.axvline(0, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y); ax.set_yticklabels([display[t] for t in order_terms])
    ax.set(xlabel="Standardized log-odds coefficient (95% CI)", xlim=(-2.25, 1.45), ylim=(-0.7, len(y)-0.3))
    clean(ax, "x")
    ax.text(0.98, 0.02, "observational; not causal", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.4, color=COL["muted"])

    # e: matched comparison
    ax = fig.add_subplot(gs[1, 4:])
    panel(ax, "e", "Familiarity-matched comparison")
    mb = matching["training_familiarity_matched"]
    sources = ["Negatome", "ProtNeg"]
    y = np.arange(2)[::-1]
    for yi, source, color in zip(y, sources, [COL["orange"], COL["green"]]):
        b = mb[source]
        point_ci(ax, b["p_structural_gt_experimental"], *b["p_ci95"], yi, color, size=45)
        ax.text(b["p_ci95"][1] + 0.015, yi,
                f"{100*b['p_structural_gt_experimental']:.1f}%  n={b['matched_pairs_n']}",
                va="center", fontsize=6.0, fontweight="normal", color=color)
    ax.axvline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y); ax.set_yticklabels(["Negatome\nreconstituted direct", "ProtNeg\ndirect assay"])
    ax.set(xlabel="P(structural score > experimental score)", xlim=(0.45, 1.01), ylim=(-0.7, 1.7))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    clean(ax, "x")
    ax.text(0.98, 0.03, "Matched on endpoint familiarity, pair length\nand length asymmetry; max |SMD| ≤0.15.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.35, color=COL["muted"])

    stem = "Figure4_negative_semantics_and_familiarity_20260822"
    save_all(fig, outdir / "main", stem)
    return {
        "stem": stem,
        "structural_clean_n": drivers["cohort_counts"]["primary_clean_core"],
        "matched_negatome_n": mb["Negatome"]["matched_pairs_n"],
        "matched_protneg_n": mb["ProtNeg"]["matched_pairs_n"],
    }


def figure5(outdir: Path) -> dict:
    """Independent contemporary validation and cross-model semantic regimes."""
    a6 = read_json(ROOT / "data/interim/figure6_protneg_assay_v1/figure6_analysis_v1.json")
    missing = read_json(ROOT / "data/interim/manuscript_missing_numbers_v1/manuscript_missing_numbers_v1.json")
    cross3 = read_json(ROOT / "data/interim/revision_v2_matching_fig5/cross_model_semantic_gap_v2.json")
    values = pd.read_csv(ROOT / "data/interim/figure6_protneg_assay_v1/figure6_plot_values_v1.tsv", sep="\t")
    harmonized = pd.read_csv(
        ROOT / "data/interim/revision_v2_semantic_robustness/harmonized_negative_manifest_v2.tsv",
        sep="\t",
    )
    features = pd.read_csv(
        ROOT / "data/interim/figure5_structure_score_drivers_v2/figure5_ac_features_v2.tsv",
        sep="\t",
    )
    clean_mask = (
        features.core_flag.eq(1)
        & features.truncated.eq(0)
        & features.train_pair.eq(0)
        & features.association_class.ne("direct_conflict")
    )

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(2, 2, left=0.08, right=0.985, bottom=0.08,
                          top=0.875, wspace=0.42, hspace=0.48)
    fig.text(0.055, 0.965,
             "Figure 5 | Independent literature negatives reproduce a model-dependent experimental–structural gap",
             fontsize=9.6, fontweight="normal", va="top")
    fig.text(0.055, 0.938,
             "480 manually reviewed pairs from 362 publications • common structural clean core n=2,480",
             fontsize=6.9, color=COL["muted"], va="top")

    # a: contemporary funnel and method composition
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "Independent contemporary-negative cohort")
    funnel = a6["6a_funnel"]
    labels = ["Reviewed\nunique", "Taxon\nmapped", "Sequence\navailable", "Negatome\nindependent", "Direct assay\nprimary"]
    vals = [x["n"] for x in funnel[:5]]
    x = np.arange(len(vals))
    ax.plot(x, vals, color=COL["green"], lw=1.7, marker="o", ms=5)
    ax.fill_between(x, vals, color=COL["green_light"], alpha=0.65)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 18, f"{v}", ha="center", fontsize=6.2, fontweight="normal")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set(ylabel="Pairs retained", ylim=(0, 545))
    clean(ax, "y")
    counts = a6["6a_raw_method_counts"]
    ax.text(0.02, 0.03,
            f"Reported methods before mapping: biophysical direct {counts['biophysical_direct']}; "
            f"other direct binding {counts['other_direct_binding']}; association/proximity {counts['association_negative']}.",
            transform=ax.transAxes, fontsize=5.45, color=COL["muted"], va="bottom")

    # b: score ECDFs
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "PLM-Interact score distributions")
    arrays = {
        "Random unlabeled": values.loc[values.dataset == "Random negative", "score"].to_numpy(float),
        "Negatome reconstituted": harmonized.loc[
            (harmonized.source == "Negatome")
            & (harmonized.eligible_common_qc == 1)
            & (harmonized.assay_ontology == "reconstituted_direct"),
            "score",
        ].to_numpy(float),
        "ProtNeg direct": values.loc[values.dataset == "ProtNeg direct", "score"].to_numpy(float),
        "Assembly noncontact": features.loc[clean_mask, "score"].to_numpy(float),
    }
    colors = [COL["grey"], COL["orange"], COL["green"], COL["red"]]
    for (label, arr), color in zip(arrays.items(), colors):
        xx, yy = ecdf(arr)
        ax.plot(xx, yy, color=color, lw=1.5, label=f"{label} (n={len(arr):,})")
    ax.axvline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set(xlabel="Published PLM-Interact score", ylabel="Cumulative fraction", xlim=(0, 1), ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean(ax, "both")
    ax.legend(loc="lower right")

    # c: cross-model probability of superiority
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "c", "Cross-model semantic shift")
    models = ["PLM-Interact", "MINT", "D-SCRIPT", "Topsy-Turvy", "SPRINT"]
    y = np.arange(len(models))[::-1]
    blocks = {}
    for model in ("PLM-Interact", "D-SCRIPT", "Topsy-Turvy"):
        blocks[model] = {
            "Negatome": cross3["all_taxa"][model]["Negatome"],
            "ProtNeg": cross3["all_taxa"][model]["ProtNeg"],
        }
    ms = missing["figure5_cross_model"]["MINT_SPRINT_with_ci"]
    for model in ("MINT", "SPRINT"):
        blocks[model] = {
            "Negatome": {
                "p_structural_gt_experimental": ms[model]["Negatome_298_reconstituted_direct"]["P_structural_score_gt_experimental"],
                "bootstrap_ci95": ms[model]["Negatome_298_reconstituted_direct"]["bootstrap_ci95"],
            },
            "ProtNeg": {
                "p_structural_gt_experimental": ms[model]["ProtNeg_104_direct_assay"]["P_structural_score_gt_experimental"],
                "bootstrap_ci95": ms[model]["ProtNeg_104_direct_assay"]["bootstrap_ci95"],
            },
        }
    for yi, model in zip(y, models):
        for offset, source, color, marker in [(0.12, "Negatome", COL["orange"], "o"), (-0.12, "ProtNeg", COL["green"], "s")]:
            b = blocks[model][source]
            point_ci(ax, b["p_structural_gt_experimental"], *b["bootstrap_ci95"], yi + offset,
                     color, marker=marker, size=32)
    ax.axvline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y); ax.set_yticklabels(models)
    ax.set(xlabel="P(structural score > experimental-direct score)", xlim=(0.28, 0.94), ylim=(-0.7, len(models)-0.3))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    clean(ax, "x")
    ax.scatter([], [], color=COL["orange"], marker="o", label="Negatome reconstituted (n=298)")
    ax.scatter([], [], color=COL["green"], marker="s", label="ProtNeg direct (n=104; 98 for D-SCRIPT/Topsy)")
    ax.legend(loc="upper left")
    ax.text(0.98, 0.03, "SPRINT reverses the orientation; scores are not compared across models.",
            transform=ax.transAxes, ha="right", fontsize=5.35, color=COL["muted"])

    # d: supportive paired ranking and assay sensitivity
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "d", "Supportive direct-positive partner ranking")
    rank = a6["6c_paired_ranking"]
    specs = [
        ("All selected", rank["all_selected"]),
        ("Both pairs\nnon-truncated", rank["both_pairs_no_truncation"]),
        ("Biophysical direct", rank["by_negative_assay_semantics"]["biophysical_direct"]),
        ("Other direct binding", rank["by_negative_assay_semantics"]["other_direct_binding"]),
        ("Association/proximity", rank["by_negative_assay_semantics"]["association_negative"]),
        ("Structural\nfamily-unseen", rank["structure_family_unseen_original"]),
    ]
    y = np.arange(len(specs))[::-1]
    for yi, (name, b) in zip(y, specs):
        p = b.get("win_rate")
        lo, hi = b["ci95"]
        color = COL["grey"] if "Structural" in name else COL["green"]
        point_ci(ax, p, lo, hi, yi, color, size=35)
        n = b.get("n")
        ax.text(hi + 0.012, yi, f"{100*p:.1f}%  n={n}", va="center", fontsize=5.7)
    ax.axvline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_yticks(y); ax.set_yticklabels([x[0] for x in specs])
    ax.set(xlabel="P(positive partner D > negative partner C)", xlim=(0.42, 1.02), ylim=(-0.7, len(specs)-0.3))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    clean(ax, "x")
    ax.text(0.98, 0.98, "IntAct MI:0407 D candidates were not individually re-curated.",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.25, color=COL["muted"])

    stem = "Figure5_independent_cross_model_validation_20260822"
    save_all(fig, outdir / "main", stem)
    return {
        "stem": stem,
        "reviewed_pairs": 480,
        "direct_primary_n": 104,
        "paired_ranking_n": rank["all_selected"]["n"],
    }


def supplementary_s12(outdir: Path) -> dict:
    """Updated S12 using the completed locked 300-triplet comparison."""
    summary = read_json(
        ROOT / "data/interim/af3_partner_ranking_v1/locked300_all_models_v1/summary_v1.json"
    )["all_model_complete"]
    trip = pd.read_csv(
        ROOT / "data/interim/af3_partner_ranking_v1/locked300_all_models_v1/triplet_scores_v1.tsv",
        sep="\t",
    )
    af3detail = read_json(
        ROOT / "data/interim/af3_partner_ranking_v1/main300_eval_v1/main300_partner_ranking_v1.json"
    ) if (ROOT / "data/interim/af3_partner_ranking_v1/main300_eval_v1/main300_partner_ranking_v1.json").exists() else None

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.09,
                          top=0.84, wspace=0.48, hspace=0.48)
    fig.text(0.055, 0.955,
             "Supplementary Figure S12 | Locked 300-triplet structure-model comparison and AlphaFold 3 sensitivity",
             fontsize=9.8, fontweight="normal", va="top")
    fig.text(0.055, 0.920,
             "All seven methods complete on 300 triplets from 256 anchors; no formal out-of-training-distribution claim.",
             fontsize=6.8, color=COL["muted"], va="top")

    # a paired scatter
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "Paired AlphaFold 3 ipTM")
    hit = trip.AF3_ad > trip.AF3_ac
    ax.scatter(trip.loc[~hit, "AF3_ac"], trip.loc[~hit, "AF3_ad"], s=11, color=COL["red"], alpha=0.60, edgecolor="none")
    ax.scatter(trip.loc[hit, "AF3_ac"], trip.loc[hit, "AF3_ad"], s=11, color=COL["green"], alpha=0.68, edgecolor="none")
    ax.plot([0, 1], [0, 1], color=COL["grey"], lw=0.8, ls=(0, (2, 2)))
    ax.set(xlabel="ipTM, A–C", ylabel="ipTM, A–D", xlim=(-0.02, 1), ylim=(-0.02, 1))
    clean(ax, "both")

    # b distributions
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "ipTM distributions by role")
    for col, label, color in [("AF3_ad", "A–D direct contact", COL["green"]), ("AF3_ac", "A–C noncontact", COL["red"])]:
        xx, yy = ecdf(trip[col].to_numpy(float)); ax.plot(xx, yy, color=color, lw=1.5, label=label)
    ax.set(xlabel="AlphaFold 3 ipTM", ylabel="Cumulative fraction", xlim=(0, 1), ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0)); clean(ax, "both"); ax.legend(loc="lower right")

    # c thresholds
    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c", "High-confidence counts")
    th = np.array([0.2, 0.4, 0.6, 0.8])
    ad = [(trip.AF3_ad >= x).sum() for x in th]; ac = [(trip.AF3_ac >= x).sum() for x in th]
    x = np.arange(len(th)); w = 0.36
    ax.bar(x-w/2, ad, width=w, color=COL["green_light"], edgecolor=COL["green"], label="A–D")
    ax.bar(x+w/2, ac, width=w, color=COL["red_light"], edgecolor=COL["red"], label="A–C")
    ax.set_xticks(x); ax.set_xticklabels([f"≥{v:.1f}" for v in th])
    ax.set(xlabel="ipTM threshold", ylabel="Pairs (of 300)"); clean(ax, "y"); ax.legend(loc="upper right")

    # d common model comparison
    ax = fig.add_subplot(gs[1, :2])
    panel(ax, "d", "All-model partner ranking")
    order = ["PLM-Interact", "MINT", "D-SCRIPT", "Topsy-Turvy", "SPRINT", "ESMFold", "AF3"]
    x = np.arange(len(order))
    vals = [summary["models"][m]["P_D_gt_C"] for m in order]
    los = [summary["models"][m]["anchor_block_ci95"][0] for m in order]
    his = [summary["models"][m]["anchor_block_ci95"][1] for m in order]
    colors = [COL["grey"]] * 6 + [COL["green"]]
    ax.scatter(x, vals, c=colors, s=[30]*6+[48], zorder=3)
    ax.errorbar(x, vals, yerr=[np.asarray(vals)-np.asarray(los), np.asarray(his)-np.asarray(vals)],
                fmt="none", ecolor=COL["ink"], capsize=2, lw=0.8, zorder=2)
    ax.axhline(0.5, color=COL["grey"], lw=0.8, ls=(0, (3, 2)))
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set(ylabel="P(D>C)", ylim=(0.40, 0.72)); ax.yaxis.set_major_formatter(PercentFormatter(1.0)); clean(ax, "y")

    # e stratification and guardrails
    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "e", "Interpretation guardrails")
    ax.axis("off")
    notes = [
        "AF3: version 3.0.3; one seed × five diffusion samples; ipTM endpoint.",
        "ESMFold: facebook/esmfold_v1; A+G25+B linker; −mean inter-chain PAE endpoint.",
        "ESMFold chain pLDDT remained <16 for all 577 dimers; the linker protocol did not form a reliable interface.",
        "AF3 exact-pair pre-cutoff exposure was not reconstructed; this is a benchmark comparison, not a zero-shot claim.",
    ]
    y = 0.92
    for i, note in enumerate(notes, 1):
        ax.text(0.02, y, f"{i}. {note}", fontsize=6.2, va="top", wrap=True,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=COL["grey_light"], edgecolor="none"))
        y -= 0.22

    stem = "Supplementary_Figure_S12_locked300_structure_models_20260822"
    save_all(fig, outdir / "supplementary", stem)
    return {"stem": stem, "n_triplets": 300, "n_anchors": 256, "af3_p": 0.63}


def main() -> None:
    args = parse_args()
    style()
    stats = {}
    if args.only in ("all", "Figure3"):
        stats["Figure3"] = figure3(args.output_dir)
    if args.only in ("all", "Figure4"):
        stats["Figure4"] = figure4(args.output_dir)
    if args.only in ("all", "Figure5"):
        stats["Figure5"] = figure5(args.output_dir)
    if args.only in ("all", "S12"):
        stats["S12"] = supplementary_s12(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "reorganized_figure_stats_20260822.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
