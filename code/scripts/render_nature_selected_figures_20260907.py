#!/usr/bin/env python3
"""Render the selected Nature main-figure panels with a compact 2x2 layout.

This renderer reuses the locked local ledgers and the plotting helpers from
render_reorganized_figures_20260822.py, but keeps only the panels requested for
the Nature submission: Figure 3 old a,c,d,f -> new a,b,c,d; Figure 4 old
b,c,d,e -> new a,b,c,d.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

import render_reorganized_figures_20260822 as rr


ROOT = rr.ROOT
ADOBE_EDITABLE_TEXT = False


def render_figure3(outdir: Path) -> None:
    plm = pd.read_csv(ROOT / "data/interim/plminteract_layerwise_probes_v1/layerwise_probes_metrics_v1.tsv", sep="\t")
    mint = pd.read_csv(ROOT / "data/interim/mint_layerwise_probes_v1/layerwise_probes_metrics_v1.tsv", sep="\t")
    missing = rr.read_json(ROOT / "data/interim/manuscript_missing_numbers_v1/manuscript_missing_numbers_v1.json")["figure3_final_layer_probes"]
    locked = rr.read_json(ROOT / "data/interim/af3_partner_ranking_v1/locked300_all_models_v1/summary_v1.json")["all_model_complete"]
    trip = pd.read_csv(ROOT / "data/interim/af3_partner_ranking_v1/locked300_all_models_v1/triplet_scores_v1.tsv", sep="\t")

    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(2, 2, left=.08, right=.985, bottom=.09, top=.86, wspace=.50, hspace=.58)

    # New a = old a: PLM-Interact layer-wise probes.
    ax = fig.add_subplot(gs[0, 0]); rr.panel(ax, "a", "PLM-Interact: context rises; contact does not")
    ax.plot(plm.layer, plm.plm_context, color=rr.COL["blue"], lw=1.8, label="Context, PLM-Interact")
    ax.fill_between(plm.layer, plm.plm_context_lo, plm.plm_context_hi, color=rr.COL["blue"], alpha=.13, linewidth=0)
    ax.plot(plm.layer, plm.plm_contact, color=rr.COL["red"], lw=1.6, label="Contact, PLM-Interact")
    ax.fill_between(plm.layer, plm.plm_contact_lo, plm.plm_contact_hi, color=rr.COL["red"], alpha=.12, linewidth=0)
    ax.plot(plm.layer, plm.esm2_context, color=rr.COL["blue"], lw=1.1, ls="--", label="Context, ESM-2")
    ax.plot(plm.layer, plm.esm2_contact, color=rr.COL["red"], lw=1.1, ls="--", label="Contact, ESM-2")
    ax.axhline(.5, color=rr.COL["grey"], lw=.8, ls=(0,(3,2)))
    ax.set(xlabel="Transformer layer", ylabel="Family-unseen AUROC", ylim=(.44,.76), xlim=(0,33)); rr.clean(ax,"y"); ax.legend(loc="upper left", fontsize=5.8)

    # New b = old c: final-layer probes.
    ax = fig.add_subplot(gs[0, 1]); rr.panel(ax, "b", "Final-layer probes")
    specs=[("ESM-2\npair CLS",missing["ESM-2_pairCLS"]),("PLM-Interact\npair CLS",missing["PLM-Interact_pairCLS"]),("MINT\nconcat mean-pool",missing["MINT_concat_meanpool"])]
    yy=np.arange(3)[::-1]
    for y,(label,b) in zip(yy,specs):
        rr.point_ci(ax,b["context_auroc"],*b["context_ci95"],y+.13,rr.COL["blue"],marker="o")
        rr.point_ci(ax,b["contact_auroc"],*b["contact_ci95"],y-.13,rr.COL["red"],marker="s")
    ax.axvline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set_yticks(yy); ax.set_yticklabels([s[0] for s in specs]); ax.set(xlabel="AUROC (triplet-bootstrap 95% CI)",xlim=(.44,.75),ylim=(-.6,2.6)); rr.clean(ax,"x")
    ax.scatter([],[],color=rr.COL["blue"],marker="o",label="Context"); ax.scatter([],[],color=rr.COL["red"],marker="s",label="Direct contact"); ax.legend(loc="lower right",fontsize=5.8)

    # New c = old d: common-set ranking.
    ax = fig.add_subplot(gs[1, 0]); rr.panel(ax, "c", "Common-set partner ranking")
    order=["AF3","ESMFold","PLM-Interact","MINT","D-SCRIPT","Topsy-Turvy","SPRINT"]; yy=np.arange(len(order))[::-1]
    for y,model in zip(yy,order):
        b=locked["models"][model]; color=rr.COL["green"] if model=="AF3" else rr.COL["grey"]
        rr.point_ci(ax,b["P_D_gt_C"],*b["anchor_block_ci95"],y,color,size=42 if model=="AF3" else 30)
        ax.text(b["anchor_block_ci95"][1]+.006,y,f"{100*b['P_D_gt_C']:.1f}%",va="center",fontsize=5.4,color=color)
    ax.axvline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set_yticks(yy); ax.set_yticklabels(order); ax.set(xlabel="P[s(A,D) > s(A,C)]",xlim=(.41,.72),ylim=(-.7,len(order)-.3)); ax.xaxis.set_major_formatter(PercentFormatter(1)); rr.clean(ax,"x")

    # New d = old f: paired AlphaFold 3 ipTM.
    ax = fig.add_subplot(gs[1, 1]); rr.panel(ax, "d", "Paired AlphaFold 3 ipTM")
    win=trip["AF3_ad"]>trip["AF3_ac"]; ax.scatter(trip.loc[~win,"AF3_ac"],trip.loc[~win,"AF3_ad"],s=12,color=rr.COL["red"],alpha=.55); ax.scatter(trip.loc[win,"AF3_ac"],trip.loc[win,"AF3_ad"],s=12,color=rr.COL["green"],alpha=.65)
    ax.plot([0,1],[0,1],color=rr.COL["grey"],lw=.8,ls=(0,(2,2))); ax.set(xlabel="ipTM, A–C assembly noncontact",ylabel="ipTM, A–D direct contact",xlim=(0,1),ylim=(0,1)); rr.clean(ax,"both")
    ax.text(.97,.04,"P(D>C)=63.0%\n95% CI 57.2–68.6%",transform=ax.transAxes,ha="right",va="bottom",fontsize=5.8,color=rr.COL["green"])
    rr.save_all(fig,outdir/"main","Figure3_representation_and_structure_models_20260907")


def render_figure4(outdir: Path) -> None:
    f34=rr.read_json(ROOT/"data/reported_stats/figure34_render_stats_v1.json")["figure4"]
    drivers=rr.read_json(ROOT/"data/interim/figure5_structure_score_drivers_v2/figure5_score_drivers_v2.json")
    matching=rr.read_json(ROOT/"data/interim/revision_v2_matching_fig5/matching_fig5_robustness_v2.json")
    fig=plt.figure(figsize=(7.8,6.2)); gs=fig.add_gridspec(2,2,left=.075,right=.985,bottom=.10,top=.86,wspace=.55,hspace=.58)

    # New a = old b: score-positive fractions.
    ax=fig.add_subplot(gs[0,0]); rr.panel(ax,"a","Score regimes depend on evidence semantics")
    order=["Random A–R","Reconstituted direct","Cell binary / proximity","Co-complex association","Structural clean non-contact"]; colors=[rr.COL["grey"],rr.COL["orange"],rr.COL["blue"],rr.COL["purple"],rr.COL["red"]]; labels=["Random unlabeled","Reconstituted\ndirect","Cell binary /\nproximity","Co-complex\nassociation","Assembly\nnoncontact"]; x=np.arange(len(order)); vals=[f34["groups"][k]["frac_ge_0.5"] for k in order]; lo=[f34["groups"][k]["frac_ge_0.5_ci95"][0] for k in order]; hi=[f34["groups"][k]["frac_ge_0.5_ci95"][1] for k in order]
    ax.bar(x,vals,color=colors,alpha=.88,width=.68); ax.errorbar(x,vals,yerr=[np.asarray(vals)-np.asarray(lo),np.asarray(hi)-np.asarray(vals)],fmt="none",ecolor=rr.COL["ink"],capsize=2,lw=.8)
    for xi,v,key in zip(x,vals,order): ax.text(xi,v+.045,f"{100*v:.1f}%\nn={f34['groups'][key]['n']:,}",ha="center",va="bottom",fontsize=5.2)
    score_threshold_label = "Fraction with predicted score >=0.5" if ADOBE_EDITABLE_TEXT else "Fraction with predicted score ≥0.5"
    ax.axhline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set_xticks(x); ax.set_xticklabels(labels); ax.set(ylabel=score_threshold_label,ylim=(0,.9)); ax.yaxis.set_major_formatter(PercentFormatter(1)); rr.clean(ax,"y")

    # New b = old c: training familiarity.
    ax=fig.add_subplot(gs[0,1]); rr.panel(ax,"b","Training familiarity")
    bins=["0-20","20-40","40-60","60-80","80-100"]; block=drivers["5a_training_exposure"]["by_endpoint_train_familiarity"]; vals=[block[b]["frac_ge_0.5"] for b in bins]; los=[block[b]["ci95"][0] for b in bins]; his=[block[b]["ci95"][1] for b in bins]; x=np.arange(len(bins)); ax.plot(x,vals,color=rr.COL["blue"],marker="o",lw=1.5,ms=4); ax.errorbar(x,vals,yerr=[np.asarray(vals)-np.asarray(los),np.asarray(his)-np.asarray(vals)],fmt="none",ecolor=rr.COL["blue"],capsize=2,lw=.8); ax.axhline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set_xticks(x); ax.set_xticklabels(["<20 /\nno hit","20–40","40–60","60–80","80–100"]); ax.set(xlabel="Minimum endpoint identity to positive-training proteins (%)",ylabel=("Fraction score >=0.5" if ADOBE_EDITABLE_TEXT else "Fraction score ≥0.5"),ylim=(.48,1.02)); ax.yaxis.set_major_formatter(PercentFormatter(1)); rr.clean(ax,"y")

    # New c = old d: adjusted regression forest.
    ax=fig.add_subplot(gs[1,0]); rr.panel(ax,"c","Adjusted score correlates"); coef=[x for x in drivers["5e_multivariable"]["coefficients"] if x["term"]!="intercept"]; display={"train_familiarity":"Training familiarity","ac_3mer_similarity":"A–C sequence similarity","log_complex_size":"Complex size","log1p_replicate_pdb":"Repeated-PDB support","log_min_heavy":"Minimum heavy-atom distance","log_pair_length":"Pair sequence length","log1p_bridge_contacts":"Bridge-interface contacts","human":"Human–human pair"}; terms=list(display); by={x["term"]:x for x in coef}; y=np.arange(len(terms))[::-1]
    for yi,t in zip(y,terms): b=by[t]; rr.point_ci(ax,b["coef_log_odds"],*b["ci95"],yi,rr.COL["blue"] if b["coef_log_odds"]>=0 else rr.COL["red"],size=28)
    ax.axvline(0,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set_yticks(y); ax.set_yticklabels([display[t] for t in terms]); ax.set(xlabel="Standardized log-odds coefficient (95% CI)",xlim=(-2.25,1.45),ylim=(-.7,len(y)-.3)); rr.clean(ax,"x")

    # New d = old e: matched comparison.
    ax=fig.add_subplot(gs[1,1]); rr.panel(ax,"d","Familiarity-matched comparison"); mb=matching["training_familiarity_matched"]; sources=["Negatome","ProtNeg"]; y=np.arange(2)[::-1]
    for yi,source,color in zip(y,sources,[rr.COL["orange"],rr.COL["green"]]): b=mb[source]; rr.point_ci(ax,b["p_structural_gt_experimental"],*b["p_ci95"],yi,color,size=45); ax.text(b["p_ci95"][1]+.015,yi,f"{100*b['p_structural_gt_experimental']:.1f}%  n={b['matched_pairs_n']}",va="center",fontsize=5.8,color=color)
    ax.axvline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set_yticks(y); ax.set_yticklabels(["Negatome\nreconstituted direct","ProtNeg\ndirect assay"]); ax.set(xlabel="P(structural score > experimental score)",xlim=(.45,1.01),ylim=(-.7,1.7)); ax.xaxis.set_major_formatter(PercentFormatter(1)); rr.clean(ax,"x")
    rr.save_all(fig,outdir/"main","Figure4_negative_semantics_and_familiarity_20260907")


def render_figure5(outdir: Path) -> None:
    """Reference-style Figure 5: replication, ordering and support ranking."""
    a6 = rr.read_json(ROOT / "data/interim/figure6_protneg_assay_v1/figure6_analysis_v1.json")
    missing = rr.read_json(ROOT / "data/interim/manuscript_missing_numbers_v1/manuscript_missing_numbers_v1.json")
    cross3 = rr.read_json(ROOT / "data/interim/revision_v2_matching_fig5/cross_model_semantic_gap_v2.json")
    values = pd.read_csv(ROOT / "data/interim/figure6_protneg_assay_v1/figure6_plot_values_v1.tsv", sep="\t")
    harmonized = pd.read_csv(ROOT / "data/interim/revision_v2_semantic_robustness/harmonized_negative_manifest_v2.tsv", sep="\t")
    features = pd.read_csv(ROOT / "data/interim/figure5_structure_score_drivers_v2/figure5_ac_features_v2.tsv", sep="\t")
    clean_mask = features.core_flag.eq(1) & features.truncated.eq(0) & features.train_pair.eq(0) & features.association_class.ne("direct_conflict")
    fig=plt.figure(figsize=(7.2,6.2)); gs=fig.add_gridspec(2,2,left=.08,right=.985,bottom=.10,top=.86,wspace=.50,hspace=.58)
    # a: former b, independent PLM-Interact score distributions
    ax=fig.add_subplot(gs[0,0]); rr.panel(ax,"a","Independent replication in PLM-Interact")
    arrays={"Negatome reconstituted":harmonized.loc[(harmonized.source=="Negatome")&(harmonized.eligible_common_qc==1)&(harmonized.assay_ontology=="reconstituted_direct"),"score"].to_numpy(float),"ProtNeg direct":values.loc[values.dataset=="ProtNeg direct","score"].to_numpy(float),"Assembly noncontact":features.loc[clean_mask,"score"].to_numpy(float)}
    for j,((label,arr),color) in enumerate(zip(arrays.items(),[rr.COL["orange"],rr.COL["green"],rr.COL["red"]])):
        xx,yy=rr.ecdf(arr); ax.plot(xx,yy,color=color,lw=1.5,label=f"{label} (n={len(arr):,})"); ax.text(.50,.82-.10*j,f"median={np.median(arr):.4f}",transform=ax.transAxes,fontsize=5.5,color=color)
    ax.axvline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set(xlabel="Published PLM-Interact score",ylabel="Cumulative fraction",xlim=(0,1),ylim=(0,1)); ax.yaxis.set_major_formatter(PercentFormatter(1)); rr.clean(ax,"both")
    # b: former c, cross-model semantic ordering
    ax=fig.add_subplot(gs[0,1]); rr.panel(ax,"b","Experimental–structural ordering across models")
    models=["PLM-Interact","MINT","D-SCRIPT","Topsy-Turvy","SPRINT"]; blocks={}
    for model in ("PLM-Interact","D-SCRIPT","Topsy-Turvy"): blocks[model]={"Negatome":cross3["all_taxa"][model]["Negatome"],"ProtNeg":cross3["all_taxa"][model]["ProtNeg"]}
    ms=missing["figure5_cross_model"]["MINT_SPRINT_with_ci"]
    for model in ("MINT","SPRINT"):
        blocks[model]={"Negatome":{"p_structural_gt_experimental":ms[model]["Negatome_298_reconstituted_direct"]["P_structural_score_gt_experimental"],"bootstrap_ci95":ms[model]["Negatome_298_reconstituted_direct"]["bootstrap_ci95"]},"ProtNeg":{"p_structural_gt_experimental":ms[model]["ProtNeg_104_direct_assay"]["P_structural_score_gt_experimental"],"bootstrap_ci95":ms[model]["ProtNeg_104_direct_assay"]["bootstrap_ci95"]}}
    yy=np.arange(len(models))[::-1]
    for y,model in zip(yy,models):
        for off,source,color,marker in [(0.12,"Negatome",rr.COL["orange"],"o"),(-0.12,"ProtNeg",rr.COL["green"],"s")]:
            b=blocks[model][source]; rr.point_ci(ax,b["p_structural_gt_experimental"],*b["bootstrap_ci95"],y+off,color,marker=marker,size=28)
    ax.axvline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set_yticks(yy); ax.set_yticklabels(models); ax.set(xlabel="P(structural score > experimental-direct score)",xlim=(.28,.94),ylim=(-.7,len(models)-.3)); ax.xaxis.set_major_formatter(PercentFormatter(1)); rr.clean(ax,"x")
    # c: new cross-dataset concordance scatter
    ax=fig.add_subplot(gs[1,0]); rr.panel(ax,"c","Model-specific ordering reproduces across datasets")
    colors={"PLM-Interact":rr.COL["orange"],"MINT":rr.COL["green"],"D-SCRIPT":rr.COL["blue"],"Topsy-Turvy":rr.COL["red"],"SPRINT":rr.COL["grey"]}; xs=np.array([blocks[m]["Negatome"]["p_structural_gt_experimental"] for m in models]); ys=np.array([blocks[m]["ProtNeg"]["p_structural_gt_experimental"] for m in models]); rho=float(np.corrcoef(pd.Series(xs).rank(),pd.Series(ys).rank())[0,1])
    ax.plot([.35,.92],[.35,.92],color=rr.COL["grid"],lw=.8); ax.axvline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.axhline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2)))
    for x,y,model in zip(xs,ys,models): ax.scatter(x,y,s=42,color=colors[model],edgecolor="white",linewidth=.5); ax.text(x+.009,y+.008,model,fontsize=5.4,color=colors[model])
    rho_label = f"Spearman rho = {rho:.2f}" if ADOBE_EDITABLE_TEXT else f"Spearman ρ = {rho:.2f}"
    ax.text(.03,.94,rho_label,transform=ax.transAxes,fontsize=6.0,va="top"); ax.set(xlabel="Negatome P(structural > experimental)",ylabel="ProtNegDB P(structural > experimental)",xlim=(.35,.92),ylim=(.35,.90)); rr.clean(ax,"both")
    # d: former d, supportive partner ranking
    ax=fig.add_subplot(gs[1,1]); rr.panel(ax,"d","Supportive direct-positive partner ranking"); rank=a6["6c_paired_ranking"]; specs=[("All selected",rank["all_selected"]),("Both pairs\nnon-truncated",rank["both_pairs_no_truncation"]),("Biophysical direct",rank["by_negative_assay_semantics"]["biophysical_direct"]),("Other direct binding",rank["by_negative_assay_semantics"]["other_direct_binding"]),("Association/proximity",rank["by_negative_assay_semantics"]["association_negative"]),("Structural\nfamily-unseen",rank["structure_family_unseen_original"])]
    yy=np.arange(len(specs))[::-1]
    for y,(name,b) in zip(yy,specs): p=b["win_rate"]; lo,hi=b["ci95"]; color=rr.COL["grey"] if "Structural" in name else rr.COL["green"]; rr.point_ci(ax,p,lo,hi,y,color,size=32); ax.text(hi+.012,y,f"{100*p:.1f}%  n={b.get('n')}",va="center",fontsize=5.5)
    ax.axvline(.5,color=rr.COL["grey"],lw=.8,ls=(0,(3,2))); ax.set_yticks(yy); ax.set_yticklabels([x[0] for x in specs]); ax.set(xlabel="P(positive partner D > negative partner C)",xlim=(.42,1.02),ylim=(-.7,len(specs)-.3)); ax.xaxis.set_major_formatter(PercentFormatter(1)); rr.clean(ax,"x")
    rr.save_all(fig,outdir/"main","Figure5_independent_cross_model_validation_20260907")


def main() -> None:
    global ADOBE_EDITABLE_TEXT
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=ROOT/"deliverables/nature_submission_20260823/figures/regenerated")
    p.add_argument(
        "--adobe-editable-text",
        action="store_true",
        help=("Use the PDF standard Helvetica font instead of a subset-embedded "
              "TrueType font. This avoids Adobe converting text to outlines."),
    )
    args=p.parse_args()
    ADOBE_EDITABLE_TEXT = args.adobe_editable_text
    rr.style()
    if args.adobe_editable_text:
        # Matplotlib normally embeds a subset CID TrueType font. Although that is
        # valid searchable PDF text, Illustrator may outline it when the exact
        # embedded subset cannot be mapped back to a locally installed font.
        # Helvetica is one of PDF's standard 14 fonts, so it remains a regular,
        # selectable text object without font subsetting or embedding.
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Arial"],
            "pdf.use14corefonts": True,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        })
    (args.output_dir/"main").mkdir(parents=True,exist_ok=True)
    render_figure3(args.output_dir)
    render_figure4(args.output_dir)
    render_figure5(args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
