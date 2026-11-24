"""Recompute the Patterns benchmark from its frozen scores and evaluation splits."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


MODELS = (
    "PLM-Interact", "MINT", "D-SCRIPT", "Topsy-Turvy", "SPRINT", "PPLM-PPI",
    "RAPPPID", "TUnA", "AF3", "RoseTTAFold2-PPI", "ESMFold", "PPLM-Contact",
)
SHARED = {"common11": MODELS[:-1], "common12": MODELS}


def read(path):
    return pd.read_csv(path, sep="\t", float_precision="round_trip")


def discrimination(labels, scores):
    positive = int(labels.sum())
    negative = len(labels) - positive
    return {
        "n": len(labels), "positive": positive, "negative": negative,
        "prevalence": labels.mean(),
        "AUROC": roc_auc_score(labels, scores) if positive and negative else np.nan,
        "AP": average_precision_score(labels, scores) if positive else np.nan,
    }


def anchor_wins(anchor, first, second):
    scores = (first > second).astype(float) + 0.5 * (first == second)
    means = scores.groupby(anchor).mean()
    return len(scores), len(means), means.mean()


def structural(data):
    comparisons = read(data / "structural_ranking.tsv.gz")
    observations = read(data / "structural_binary.tsv.gz").query("split == 'test'")
    rows = []
    for scope, models in [*(("available", (m,)) for m in MODELS), *SHARED.items()]:
        rank_columns = [f"{model}_{role}" for model in models for role in ("D", "C")]
        ranked = comparisons.dropna(subset=rank_columns)
        binary = observations.dropna(subset=list(models))
        for model in models:
            n, anchors, win = anchor_wins(ranked.anchor, ranked[f"{model}_D"], ranked[f"{model}_C"])
            rows.append({
                "scope": scope, "model": model, "ranking_comparisons": n, "ranking_anchors": anchors,
                "win_rate": win, **discrimination(binary.label, binary[model]),
            })
    return rows


def matched_reference(data):
    comparisons = read(data / "matched_reference.tsv.gz")
    rows = []
    for scope, cohort in [("strict237", comparisons),
                          ("primary_all_copy", comparisons[comparisons.primary_admitted.astype(bool)])]:
        for model, panel in cohort.groupby("model"):
            for first, second in [("D", "C"), ("D", "R"), ("C", "R")]:
                n, anchors, win = anchor_wins(panel.anchor, panel[f"score_{first}"], panel[f"score_{second}"])
                rows.append({"scope": scope, "model": model, "metric": f"{first}_gt_{second}",
                             "comparisons": n, "anchors": anchors, "win_rate": win})
    return rows


def assay_discrimination(observations, scores):
    wide = scores.pivot(index="pair", columns="model", values="score")
    eligible = observations[observations.binary_eligible.astype(bool)].join(wide, on="pair")
    rows = []
    for method, panel in eligible.groupby("method"):
        reviews = {"all_sources": panel,
                   "without_context_conflicts": panel[~panel.same_method_positive_conflict.astype(bool)]}
        for review, cohort in reviews.items():
            for scope, models in {"available": MODELS, **SHARED}.items():
                shared = cohort if scope == "available" else cohort.dropna(subset=list(models))
                for model in models:
                    tested = shared.dropna(subset=[model])
                    rows.append({"method": method, "review_scope": review, "score_scope": scope,
                                 "model": model, **discrimination(tested.label, tested[model])})
    return rows


def assay_ranking(matches, scores):
    rows = []
    for model, bank in scores.groupby("model"):
        bank = bank.set_index("pair")
        paired = matches.join(bank, on="positive_pair").join(bank, on="negative_pair", rsuffix="_negative")
        paired = paired[paired.protocol.eq(paired.protocol_negative)]
        for method in matches.method.unique():
            panel = paired[paired.method.eq(method)]
            n, anchors, win = anchor_wins(panel.anchor, panel.score, panel.score_negative)
            rows.append({"method": method, "model": model, "comparisons": n, "anchors": anchors,
                         "anchor_macro_win": win})
    return rows


def fit_cutoff(labels, scores, rule):
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, (ends + 1) - tp[1:]]
    cutoffs = np.r_[np.inf, scores[ends]]
    if rule == "group_OOF_FPR10":
        eligible = np.flatnonzero(fp / (len(labels) - labels.sum()) <= 0.1 + 1e-12)
        best = min(eligible, key=lambda i: (-tp[i], fp[i], i))
    else:
        f1 = 2 * tp / (labels.sum() + tp + fp)
        best = min(range(len(cutoffs)), key=lambda i: (-f1[i], fp[i], i))
    return cutoffs[best]


def operating_points(observations, scores):
    eligible = observations[observations.binary_eligible.astype(bool)]
    scored = eligible.merge(scores, on="pair")
    calls = []
    for (_, _, _), panel in scored.groupby(["method", "model", "protocol"]):
        for fold, test in panel.dropna(subset=["fold"]).groupby("fold"):
            train = panel[panel.fold.ne(fold)]
            if train.label.nunique() != 2:
                continue
            if set(train.component) & set(test.component):
                raise ValueError("A dependency component crosses a development and test split.")
            for rule in ("group_OOF_FPR10", "group_OOF_maxF1"):
                cutoff = fit_cutoff(train.label.to_numpy(), train.score.to_numpy(), rule)
                calls.append(test.assign(rule=rule, predicted=test.score.ge(cutoff)))
    calls = pd.concat(calls)
    rows = []
    for (method, rule), panel in calls.groupby(["method", "rule"]):
        for scope, models in {"available": MODELS, **SHARED}.items():
            selected = panel[panel.model.isin(models)]
            if scope != "available":
                size = selected.groupby("sample").model.transform("nunique")
                selected = selected[size.eq(len(models))]
            for model, result in selected.groupby("model"):
                tp = int((result.label.eq(1) & result.predicted).sum())
                fp = int((result.label.eq(0) & result.predicted).sum())
                fn = int(result.label.sum()) - tp
                tn = len(result) - tp - fp - fn
                rows.append({
                    "method": method, "model": model, "rule": rule, "score_scope": scope, "n": len(result),
                    "TP": tp, "FP": fp, "TN": tn, "FN": fn, "precision": tp / (tp + fp) if tp + fp else np.nan,
                    "recall": tp / (tp + fn) if tp + fn else np.nan,
                    "FPR": fp / (fp + tn) if fp + tn else np.nan,
                    "F1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0,
                })
    return rows


def nominations(data):
    matrix = read(data / "dip_dpr.tsv.gz")
    rows = []
    for model, panel in matrix.groupby("model"):
        scored = panel.sort_values(["dip", "score", "dpr_number"], ascending=[True, False, True])
        picked = scored.groupby("dip").head(2)
        sizes = panel.groupby("dip").spr_under_200.agg(["sum", "size"])
        rows.append({
            "model": model, "nominated": len(picked), "hits": int(picked.spr_under_200.sum()),
            "hits_excluding_flagged": int(picked.loc[~picked.apparent_nonspecific.astype(bool), "spr_under_200"].sum()),
            "anchors_with_hit": int(picked.groupby("dip").spr_under_200.max().sum()),
            "random_expected_hits": (2 * sizes["sum"] / sizes["size"]).sum(),
            **discrimination(panel.spr_under_200, panel.score),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, default=Path("patterns-results"))
    args = parser.parse_args()
    data = args.data
    observations = read(data / "assay_observations.tsv.gz")
    scores = read(data / "assay_scores.tsv.gz")
    results = {
        "structural": structural(data),
        "matched_reference": matched_reference(data),
        "assay_discrimination": assay_discrimination(observations, scores),
        "assay_ranking": assay_ranking(read(data / "assay_matches.tsv.gz"), scores),
        "assay_operating_points": operating_points(observations, scores),
        "dip_dpr": nominations(data),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in results.items():
        pd.DataFrame(rows).to_csv(args.output / f"{name}.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
