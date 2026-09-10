#!/usr/bin/env python3
"""Training-familiarity matching and concise Figure 5 robustness analyses.

The script has two stages. ``--prepare-only`` writes the unique experimental
protein sequences that must be searched against the locked PLM-Interact
positive-training proteins.  The default stage consumes those MMseqs hits,
matches human experimental direct negatives to clean human structural
noncontacts, and evaluates the prespecified Figure 5 sensitivity analyses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEM = ROOT / "data/interim/revision_v2_semantic_robustness"
FIG5 = ROOT / "data/interim/figure5_structure_score_drivers_v2"
FIG6 = ROOT / "data/interim/figure6_protneg_assay_v1"
NEG = ROOT / "data/interim/negatome_manual_plminteract_v1"
OUT = ROOT / "data/interim/revision_v2_matching_fig5"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--hits", type=Path, default=OUT / "experimental_train_similarity_hits_v2.tsv")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260817)
    return p.parse_args()


def read(path: Path, delimiter: str = "\t") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h, delimiter=delimiter))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()


def make_qid(sequence: str) -> str:
    return "E" + sha1(sequence)


def prepare(outdir: Path) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    harmonized = read(SEM / "harmonized_negative_manifest_v2.tsv")
    hmap = {(r["source"], r["record_id"]): r for r in harmonized}

    neg_pairs = read(NEG / "negatome_manual_pairs_v1.tsv")
    neg_seqs = read(NEG / "negatome_manual_pairs_v1.csv", ",")
    if len(neg_pairs) != len(neg_seqs):
        raise SystemExit("Negatome sequence alignment mismatch")
    rows: list[dict[str, object]] = []
    sequences: dict[str, str] = {}
    for i, (pair, seq) in enumerate(zip(neg_pairs, neg_seqs), 1):
        rec = f"NEG:{i:05d}"
        h = hmap[("Negatome", rec)]
        qa, qb = make_qid(seq["query"]), make_qid(seq["text"])
        sequences[qa] = seq["query"]; sequences[qb] = seq["text"]
        rows.append({**h, "qid_a": qa, "qid_b": qb,
                     "len_a": len(seq["query"]), "len_b": len(seq["text"])})

    prot_scored = {r["pair2_key"]: r for r in read(FIG6 / "figure6_scored_manifest_v1.tsv")}
    prot_seqs = {r["directed_pair_id"]: r for r in read(
        FIG6 / "protneg_assay_negative_pairs_v1.csv", ",")}
    for h in harmonized:
        if h["source"] != "ProtNeg" or h["record_id"] not in prot_scored:
            continue
        scored = prot_scored[h["record_id"]]
        seq = prot_seqs[scored["negative_score_id"]]
        qa, qb = make_qid(seq["query"]), make_qid(seq["text"])
        sequences[qa] = seq["query"]; sequences[qb] = seq["text"]
        rows.append({**h, "qid_a": qa, "qid_b": qb,
                     "len_a": len(seq["query"]), "len_b": len(seq["text"])})

    write(outdir / "experimental_pair_manifest_v2.tsv", rows)
    with (outdir / "experimental_query_proteins_v2.fasta").open("w", encoding="utf-8") as h:
        for qid, seq in sorted(sequences.items()):
            h.write(f">{qid}\n{seq}\n")
    primary = [r for r in rows if r["eligible_common_qc"] == "1" and
               r["assay_ontology"] == "reconstituted_direct" and
               r["exact_string_positive_overlap"] == "0"]
    primary_qids = {str(r["qid_a"]) for r in primary} | {str(r["qid_b"]) for r in primary}
    with (outdir / "experimental_direct_dscript_pairs_v2.tsv").open("w", encoding="utf-8") as h:
        for row in primary:
            h.write(f"{row['qid_a']}\t{row['qid_b']}\n")
    with (outdir / "experimental_direct_dscript_proteins_v2.fasta").open("w", encoding="utf-8") as h:
        for qid in sorted(primary_qids):
            h.write(f">{qid}\n{sequences[qid]}\n")
    report = {"pairs": len(rows), "unique_query_sequences": len(sequences),
              "direct_scoring_pairs": len(primary),
              "direct_scoring_unique_sequences": len(primary_qids),
              "sources": {s: sum(r["source"] == s for r in rows)
                          for s in ("Negatome", "ProtNeg")}}
    (outdir / "prepare_summary_v2.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report


def load_hits(path: Path) -> dict[str, float]:
    best: dict[str, tuple[float, float]] = {}
    fields = ["query", "target", "pident", "alnlen", "qcov", "tcov", "evalue", "bits"]
    with path.open(encoding="utf-8", newline="") as h:
        for r in csv.DictReader(h, delimiter="\t", fieldnames=fields):
            bits, identity = float(r["bits"]), float(r["pident"])
            if r["query"] not in best or bits > best[r["query"]][0]:
                best[r["query"]] = (bits, identity)
    return {q: value[1] for q, value in best.items()}


def smd(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    vl, vr = np.var(left, ddof=1), np.var(right, ddof=1)
    den = math.sqrt((vl + vr) / 2)
    return (float(np.mean(left)) - float(np.mean(right))) / den if den else 0.0


MATCH_VARS = ("train_identity_min", "train_identity_max", "log_pair_tokens", "abs_log_len_ratio")


def decorate_experimental(rows: list[dict[str, str]], hits: dict[str, float]) -> list[dict[str, object]]:
    out = []
    for raw in rows:
        ia, ib = hits.get(raw["qid_a"], 0.0), hits.get(raw["qid_b"], 0.0)
        la, lb = int(raw["len_a"]), int(raw["len_b"])
        out.append({**raw, "score": float(raw["score"]), "len_a": la, "len_b": lb,
                    "train_identity_min": min(ia, ib), "train_identity_max": max(ia, ib),
                    "no_hit_any": int(ia == 0 or ib == 0),
                    "log_pair_tokens": math.log(la + lb + 3),
                    "abs_log_len_ratio": abs(math.log(la / lb))})
    return out


def structural_rows() -> list[dict[str, object]]:
    out = []
    for raw in read(FIG5 / "figure5_ac_features_v2.tsv"):
        if not (raw["core_flag"] == "1" and raw["truncated"] == "0" and
                raw["train_pair"] == "0" and raw["human"] == "1" and
                raw["association_class"] != "direct_conflict"):
            continue
        la, lb = int(raw["len_a"]), int(raw["len_c"])
        ia, ib = float(raw["train_identity_a"]), float(raw["train_identity_c"])
        out.append({**raw, "source": "Structural", "record_id": raw["ac_id"],
                    "score": float(raw["score"]), "len_a": la, "len_b": lb,
                    "train_identity_min": min(ia, ib), "train_identity_max": max(ia, ib),
                    "no_hit_any": int(ia == 0 or ib == 0),
                    "log_pair_tokens": math.log(la + lb + 3),
                    "abs_log_len_ratio": abs(math.log(la / lb))})
    return out


def balance(exp: list[dict[str, object]], struct: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for name in MATCH_VARS:
        a = [float(r[name]) for r in struct]; b = [float(r[name]) for r in exp]
        rows.append({"variable": name, "structural_mean": float(np.mean(a)),
                     "experimental_mean": float(np.mean(b)), "smd": smd(a, b)})
    a = [float(r["no_hit_any"]) for r in struct]; b = [float(r["no_hit_any"]) for r in exp]
    rows.append({"variable": "no_hit_any", "structural_mean": float(np.mean(a)),
                 "experimental_mean": float(np.mean(b)), "smd": smd(a, b)})
    return rows


def match_group(exp: list[dict[str, object]], struct: list[dict[str, object]],
                source: str, n_boot: int, seed: int) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    # Standardize on the pooled comparison population. Greedily retain the
    # closest non-reused pairs, then choose the largest prefix satisfying the
    # prespecified absolute-SMD balance gate (<=0.15 for every covariate).
    pool = exp + struct
    means = np.array([np.mean([float(r[v]) for r in pool]) for v in MATCH_VARS])
    sds = np.array([np.std([float(r[v]) for r in pool], ddof=1) for v in MATCH_VARS])
    sds[sds < 1e-12] = 1.0
    xe = (np.array([[float(r[v]) for v in MATCH_VARS] for r in exp]) - means) / sds
    xs = (np.array([[float(r[v]) for v in MATCH_VARS] for r in struct]) - means) / sds
    cost = np.sqrt(((xe[:, None, :] - xs[None, :, :]) ** 2).sum(axis=2))
    candidates = sorted((float(cost[i, j]), i, j)
                        for i in range(len(exp)) for j in range(len(struct)))
    used_e: set[int] = set(); used_s: set[int] = set(); candidate_pairs = []
    for distance, eidx, sidx in candidates:
        if eidx not in used_e and sidx not in used_s:
            used_e.add(eidx); used_s.add(sidx)
            candidate_pairs.append((distance, eidx, sidx))
    accepted_n = 0
    for n in range(2, len(candidate_pairs)+1):
        prefix = candidate_pairs[:n]
        current = balance([exp[e] for _,e,_ in prefix], [struct[s] for _,_,s in prefix])
        if max(abs(float(r["smd"])) for r in current) <= 0.15:
            accepted_n = n
    if accepted_n == 0:
        raise SystemExit(f"no balanced matches for {source}")
    chosen = candidate_pairs[:accepted_n]
    ie = np.array([e for _,e,_ in chosen], dtype=int)
    is_ = np.array([s for _,_,s in chosen], dtype=int)
    pairs = []
    for j, (eidx, sidx) in enumerate(zip(ie, is_), 1):
        e, s = exp[int(eidx)], struct[int(sidx)]
        pairs.append({"source": source, "match_id": f"{source}:{j:04d}",
                      "experimental_id": e["record_id"], "structural_id": s["record_id"],
                      "experimental_score": e["score"], "structural_score": s["score"],
                      "score_difference": float(s["score"])-float(e["score"]),
                      "structural_gt_experimental": int(float(s["score"]) > float(e["score"])),
                      "match_distance": float(cost[eidx, sidx])})
    me = [exp[int(i)] for i in ie]; ms = [struct[int(i)] for i in is_]
    rng = random.Random(seed)
    diffs = [float(p["score_difference"]) for p in pairs]
    wins = [float(p["structural_gt_experimental"]) for p in pairs]
    boot_d, boot_w = [], []
    for _ in range(n_boot):
        indices = [rng.randrange(len(pairs)) for _ in pairs]
        boot_d.append(float(np.mean([diffs[i] for i in indices])))
        boot_w.append(float(np.mean([wins[i] for i in indices])))
    q = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
    report = {
        "experimental_eligible_n": len(exp), "structural_pool_n": len(struct),
        "matched_pairs_n": len(pairs), "balance_gate_abs_smd": 0.15,
        "pre_match_balance": balance(exp, struct),
        "post_match_balance": balance(me, ms),
        "post_match_max_abs_smd": max(abs(r["smd"]) for r in balance(me, ms)),
        "experimental_median_score": float(np.median([r["score"] for r in me])),
        "structural_median_score": float(np.median([r["score"] for r in ms])),
        "mean_paired_score_difference": float(np.mean(diffs)), "difference_ci95": q(boot_d),
        "p_structural_gt_experimental": float(np.mean(wins)), "p_ci95": q(boot_w),
    }
    return report, pairs, balance(me, ms)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def irls(x: np.ndarray, y: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(x.shape[1])
    for _ in range(max_iter):
        p = sigmoid(x @ beta); w = np.clip(p*(1-p), 1e-7, None)
        nxt = np.linalg.pinv((x.T*w)@x) @ ((x.T*w)@(x@beta+(y-p)/w))
        if np.max(np.abs(nxt-beta)) < 1e-8:
            beta = nxt; break
        beta = nxt
    return beta, sigmoid(x@beta)


def cluster_cov(x: np.ndarray, residual: np.ndarray, bread: np.ndarray,
                groups: list[str]) -> np.ndarray:
    by: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups): by[g].append(i)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for idx in by.values():
        score = x[idx].T @ residual[idx]
        meat += np.outer(score, score)
    return bread @ meat @ bread


PREDICTORS = (
    "log_complex_size", "log_min_heavy", "train_identity_if_hit", "no_hit_any",
    "ac_3mer_similarity", "log_pair_length", "log1p_replicate_pdb", "log1p_bridge_contacts", "human",
)


def model_matrix(rows: list[dict[str, object]]) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    raw = []
    for r in rows:
        familiar = float(r["train_familiarity"])
        raw.append([math.log(float(r["num_protein_chains"])), math.log(float(r["min_heavy"])),
                    familiar/100 if familiar > 0 else 0.0, float(familiar == 0),
                    float(r["kmer_jaccard"]), math.log(float(r["pair_tokens"])),
                    math.log1p(float(r["n_pdb"])), math.log1p(float(r["bridge_contact_strength"])),
                    float(r["human"])])
    z = np.asarray(raw, dtype=float); scaling = {}
    # Standardize continuous columns, retain no-hit and human as binary.
    for j, name in enumerate(PREDICTORS):
        if name in {"no_hit_any", "human"}: continue
        mean, sd = float(z[:, j].mean()), float(z[:, j].std())
        scaling[name] = {"mean": mean, "sd": sd}
        z[:, j] = (z[:, j]-mean)/sd if sd > 1e-12 else 0
    return np.column_stack([np.ones(len(rows)), z]), scaling


def fit_threshold(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    x, _ = model_matrix(rows); y = np.array([float(float(r["score"]) >= threshold) for r in rows])
    beta, p = irls(x, y); w = np.clip(p*(1-p), 1e-7, None)
    bread = np.linalg.pinv((x.T*w)@x)
    tables = {}
    for label, key in (("anchor_accession", "protein_a"), ("pdb", "pdb_id")):
        cov = cluster_cov(x, y-p, bread, [str(r[key]) for r in rows])
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
        tables[label] = [{"term": name, "coefficient": float(b), "se": float(s),
                          "ci95": [float(b-1.96*s), float(b+1.96*s)]}
                         for name, b, s in zip(("intercept",)+PREDICTORS, beta, se)]
    return {"threshold": threshold, "event_fraction": float(y.mean()), "coefficients": tables}


def rank_ols(rows: list[dict[str, object]]) -> dict[str, object]:
    x, _ = model_matrix(rows); scores = np.array([float(r["score"]) for r in rows])
    order = np.argsort(scores, kind="mergesort"); ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores)); y = (ranks-ranks.mean())/ranks.std()
    beta = np.linalg.pinv(x.T@x)@(x.T@y); residual = y-x@beta; bread = np.linalg.pinv(x.T@x)
    cov = cluster_cov(x, residual, bread, [str(r["protein_a"]) for r in rows])
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"outcome": "standardized rank of published score",
            "coefficients_anchor_cluster": [
                {"term": n, "coefficient": float(b), "se": float(s),
                 "ci95": [float(b-1.96*s), float(b+1.96*s)]}
                for n,b,s in zip(("intercept",)+PREDICTORS,beta,se)]}


def bootstrap_familiarity(rows: list[dict[str, object]], group_key: str,
                          n_boot: int, seed: int) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows: grouped[str(r[group_key])].append(r)
    keys = sorted(grouped); rng = random.Random(seed); estimates = []
    for _ in range(n_boot):
        sample = []
        for key in (rng.choice(keys) for _ in keys): sample.extend(grouped[key])
        try:
            x, _ = model_matrix(sample)
            y = np.array([float(float(r["score"]) >= .5) for r in sample])
            estimates.append(float(irls(x, y)[0][3]))  # train_identity_if_hit
        except (ValueError, np.linalg.LinAlgError):
            continue
    return {"group": group_key, "replicates_requested": n_boot, "replicates_fit": len(estimates),
            "median_coefficient": float(np.median(estimates)),
            "ci95": [float(np.quantile(estimates,.025)), float(np.quantile(estimates,.975))]}


def fig5_robustness(n_boot: int, seed: int) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for raw in read(FIG5 / "figure5_ac_features_v2.tsv"):
        if not (raw["core_flag"] == "1" and raw["truncated"] == "0" and
                raw["train_pair"] == "0" and raw["association_class"] != "direct_conflict"):
            continue
        row: dict[str, object] = dict(raw)
        for k in ("score","num_protein_chains","min_heavy","train_familiarity","kmer_jaccard",
                  "pair_tokens","n_pdb","bridge_contact_strength","human"):
            row[k] = float(raw[k])
        rows.append(row)
    return {"n": len(rows), "threshold_sensitivity": [fit_threshold(rows,t) for t in (.25,.5,.75)],
            "continuous_rank_outcome": rank_ols(rows),
            "block_bootstrap": [bootstrap_familiarity(rows,"protein_a",n_boot,seed),
                                bootstrap_familiarity(rows,"pdb_id",n_boot,seed+1)],
            "note": "No-hit status is separated from conditional nearest-training identity."}


def analyze(a: argparse.Namespace) -> dict[str, object]:
    hits = load_hits(a.hits)
    exp_all = decorate_experimental(read(a.output_dir / "experimental_pair_manifest_v2.tsv"), hits)
    structural = structural_rows()
    matches, pair_rows, balance_rows = {}, [], []
    for i, source in enumerate(("Negatome", "ProtNeg")):
        eligible = [r for r in exp_all if r["source"] == source and
                    r["eligible_common_qc"] == "1" and
                    r["assay_ontology"] == "reconstituted_direct" and
                    r["taxon_a"] == "9606" and r["taxon_b"] == "9606" and
                    int(r["len_a"])+int(r["len_b"])+3 <= 1024 and
                    r["exact_string_positive_overlap"] == "0"]
        report, pairs, post = match_group(eligible, structural, source, a.bootstrap, a.seed+i)
        matches[source] = report; pair_rows.extend(pairs)
        for row in post: balance_rows.append({"source": source, **row})
    report = {"protocol": {"scope": "human, <=1024 tokens, no exact STRING-positive overlap",
                           "exposure": "harmonized reconstituted-direct negatives",
                           "matching": "maximum-cardinality closest-distance 1:1 matching without replacement; all post-match absolute SMD <=0.15",
                           "covariates": list(MATCH_VARS)},
              "training_familiarity_matched": matches,
              "figure5_robustness": fig5_robustness(a.bootstrap, a.seed+100)}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    write(a.output_dir / "matched_pairs_v2.tsv", pair_rows)
    write(a.output_dir / "postmatch_balance_v2.tsv", balance_rows)
    (a.output_dir / "matching_fig5_robustness_v2.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"matching": {k: {"n":v["matched_pairs_n"],
                                      "max_smd":v["post_match_max_abs_smd"],
                                      "p_struct_gt_exp":v["p_structural_gt_experimental"]}
                                   for k,v in matches.items()},
                      "figure5_n": report["figure5_robustness"]["n"]}, indent=2))
    return report


def main() -> None:
    a = args()
    if a.prepare_only: prepare(a.output_dir)
    else: analyze(a)


if __name__ == "__main__": main()
