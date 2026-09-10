#!/usr/bin/env python3
"""Locked clean-core analysis for paper Figure 5.

The analysis reuses published PLM-Interact scores.  It separates exact STRING
training-positive overlap from model behaviour on core structural noncontacts,
adds endpoint-to-training nearest-neighbour similarity, treats graph distance as
fixed by open-wedge construction, and uses two-way family-cluster inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/figure5_structure_score_drivers_v2"
MI_RE = re.compile(r"MI:\d+")
BINARY_METHOD_MI = {
    "MI:0009", "MI:0012", "MI:0018", "MI:0055", "MI:0065",
    "MI:0096", "MI:0114", "MI:0415",
}
ASSOCIATION_METHOD_MI = {
    "MI:0004", "MI:0006", "MI:0007", "MI:0019", "MI:0027",
    "MI:0071", "MI:0676",
}
ASSOCIATION_TYPE_MI = {"MI:0914", "MI:0915"}
DIRECT_TYPE_MI = {"MI:0407"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument(
        "--ac-table", type=Path,
        default=ROOT / "data/interim/structure_counterfactual_v1/structure_ac_base_v1.tsv",
    )
    p.add_argument(
        "--figure1-master", type=Path,
        default=ROOT / "data/interim/figure12_master_v1/figure1_ac_master_v1.tsv",
    )
    p.add_argument(
        "--scoring-pairs", type=Path,
        default=ROOT / "data/interim/structure_triplet_scoring_v1/scoring_pairs_v1.tsv",
    )
    p.add_argument(
        "--model-pairs", type=Path,
        default=ROOT / "data/interim/structure_triplet_scoring_v1/plminteract_pairs_v1.csv",
    )
    p.add_argument(
        "--model-scores", type=Path,
        default=ROOT / "data/interim/structure_triplet_scoring_v1/plminteract_scores_v1.csv",
    )
    p.add_argument(
        "--string-train", type=Path,
        default=ROOT / "data/external/model_training_sets/plm_interact/string_v12/protein.pairs_human_V12_train.tsv",
    )
    p.add_argument(
        "--similarity-hits", type=Path,
        default=OUT / "figure5_train_similarity_hits_v2.tsv",
    )
    p.add_argument(
        "--intact-cache", type=Path,
        default=OUT / "intact_clean_core_cache_v2.tsv",
    )
    return p.parse_args()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def seq_pair_hash(a: str, b: str) -> str:
    left, right = (a, b) if a <= b else (b, a)
    return sha1_text(f"{left}\t{right}")


def kmer_jaccard(a: str, b: str, k: int = 3) -> float:
    if len(a) < k or len(b) < k:
        return 0.0
    left = {a[i:i + k] for i in range(len(a) - k + 1)}
    right = {b[i:i + k] for i in range(len(b) - k + 1)}
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def load_score_map(pairs_path: Path, scores_path: Path) -> dict[str, tuple[str, str, float]]:
    pairs = list(csv.DictReader(pairs_path.open(encoding="utf-8", newline="")))
    scores = list(csv.DictReader(scores_path.open(encoding="utf-8", newline="")))
    if len(pairs) != len(scores):
        raise SystemExit(f"pair/score row mismatch: {len(pairs)} != {len(scores)}")
    return {
        pair["pair_id"]: (pair["query"], pair["text"], float(score["score"]))
        for pair, score in zip(pairs, scores)
    }


def write_fasta(path: Path, sequences: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, sequence in sorted(sequences.items()):
            handle.write(f">{key}\n{sequence}\n")


def prepare_inputs(args: argparse.Namespace) -> dict[str, object]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ac_rows = list(csv.DictReader(args.ac_table.open(encoding="utf-8", newline=""), delimiter="\t"))
    master = {
        row["ac_id"]: row
        for row in csv.DictReader(args.figure1_master.open(encoding="utf-8", newline=""), delimiter="\t")
    }
    negative_pair = {
        row["ac_id"]: row["pair_id"]
        for row in csv.DictReader(args.scoring_pairs.open(encoding="utf-8", newline=""), delimiter="\t")
        if row["role"] == "ac_negative"
    }
    score_map = load_score_map(args.model_pairs, args.model_scores)

    train_sequences: dict[str, str] = {}
    train_proteins: set[str] = set()
    train_pairs: set[str] = set()
    n_train_positive_rows = 0
    with args.string_train.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("label") != "1":
                continue
            n_train_positive_rows += 1
            seq_a, seq_b = row["query"], row["text"]
            hash_a, hash_b = sha1_text(seq_a), sha1_text(seq_b)
            train_proteins.update((hash_a, hash_b))
            train_pairs.add(seq_pair_hash(seq_a, seq_b))
            train_sequences.setdefault(f"T{hash_a}", seq_a)
            train_sequences.setdefault(f"T{hash_b}", seq_b)

    query_sequences: dict[str, str] = {}
    manifest: list[dict[str, object]] = []
    for ac in ac_rows:
        pair_id = negative_pair[ac["ac_id"]]
        seq_a, seq_c, score = score_map[pair_id]
        hash_a, hash_c = sha1_text(seq_a), sha1_text(seq_c)
        qid_a, qid_c = f"Q{hash_a}", f"Q{hash_c}"
        query_sequences.setdefault(qid_a, seq_a)
        query_sequences.setdefault(qid_c, seq_c)
        m = master[ac["ac_id"]]
        pair_tokens = len(seq_a) + len(seq_c) + 3
        train_pair = int(seq_pair_hash(seq_a, seq_c) in train_pairs)
        manifest.append({
            "ac_id": ac["ac_id"],
            "protein_a": ac["protein_a"],
            "protein_c": ac["protein_c"],
            "qid_a": qid_a,
            "qid_c": qid_c,
            "family_a": m.get("mmseqs_family_A") or ac["protein_a"],
            "family_c": m.get("mmseqs_family_C") or ac["protein_c"],
            "pdb_id": ac["pdb_id"],
            "assembly_id": ac["assembly_id"],
            "tax_id": ac["tax_id"],
            "score": f"{score:.10g}",
            "len_a": len(seq_a),
            "len_c": len(seq_c),
            "pair_tokens": pair_tokens,
            "truncated": int(pair_tokens > 1024),
            "train_pair": train_pair,
            "n_train_proteins": int(hash_a in train_proteins) + int(hash_c in train_proteins),
            "core_flag": int(m.get("core_flag") == "1"),
            "assembly_size_stratum": ac["assembly_size_stratum"],
            "num_protein_chains": int(ac["num_protein_chains"]),
            "min_heavy": ac["ac_min_heavy_a"],
            "min_cb": ac["ac_min_cb_or_ca_a"] or ac["ac_min_heavy_a"],
            "n_pdb": int(ac["aggregate_distinct_pdb_count"] or 0),
            "kmer_jaccard": f"{kmer_jaccard(seq_a, seq_c):.8g}",
            "human": int(ac["tax_id"] == "9606"),
            "bridge_contact_strength": min(
                int(ac["ab_cbca_contact_pairs_8a"] or 0),
                int(ac["bc_cbca_contact_pairs_8a"] or 0),
            ),
        })

    manifest_path = args.output_dir / "figure5_pair_manifest_v2.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    write_fasta(args.output_dir / "figure5_query_proteins_v2.fasta", query_sequences)
    write_fasta(args.output_dir / "figure5_train_positive_proteins_v2.fasta", train_sequences)

    clean = [
        row for row in manifest
        if row["core_flag"] == 1 and row["truncated"] == 0 and row["train_pair"] == 0
    ]
    unique_pairs: dict[tuple[str, str], dict[str, str]] = {}
    for row in clean:
        low, high = sorted((str(row["protein_a"]), str(row["protein_c"])))
        unique_pairs[(low, high)] = {"acc_low": low, "acc_high": high}
    pair_path = args.output_dir / "figure5_clean_core_pairs_v2.tsv"
    with pair_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["acc_low", "acc_high"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(unique_pairs.values())

    summary = {
        "n_all": len(manifest),
        "n_core": sum(int(row["core_flag"]) for row in manifest),
        "n_truncated": sum(int(row["truncated"]) for row in manifest),
        "n_exact_string_positive": sum(int(row["train_pair"]) for row in manifest),
        "n_clean_core_before_intact_conflict_filter": len(clean),
        "n_clean_core_unique_pairs": len(unique_pairs),
        "n_query_sequences": len(query_sequences),
        "n_train_positive_rows": n_train_positive_rows,
        "n_train_unique_sequences": len(train_sequences),
    }
    (args.output_dir / "figure5_prepare_summary_v2.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def mi_ids(value: str) -> set[str]:
    return set(MI_RE.findall(value or ""))


def is_expanded(value: str) -> bool:
    value = (value or "").strip().lower()
    return value not in {"", "-", 'psi-mi:"mi:1060"(spoke expansion not applicable)'}


def classify_intact_hits(hits: list[dict[str, str]]) -> str:
    labels: set[str] = set()
    for hit in hits:
        if (hit.get("negative_interaction_flag") or "").lower() == "true":
            continue
        if not (hit.get("detection_method") or hit.get("interaction_type")):
            continue
        if is_expanded(hit.get("expansion_method") or ""):
            labels.add("expanded")
            continue
        methods = mi_ids(hit.get("detection_method") or "")
        types = mi_ids(hit.get("interaction_type") or "")
        if methods & BINARY_METHOD_MI or types & DIRECT_TYPE_MI:
            labels.add("direct_conflict")
        elif methods & ASSOCIATION_METHOD_MI:
            # Require a non-structural association assay; interaction type alone
            # is insufficient because it can inherit structure-derived records.
            labels.add("association")
        elif types & ASSOCIATION_TYPE_MI:
            labels.add("other_intact")
        else:
            labels.add("other_intact")
    if "direct_conflict" in labels:
        return "direct_conflict"
    if "association" in labels:
        return "association"
    if labels:
        return "other_intact"
    return "no_recorded_association"


def load_intact_cache(path: Path) -> tuple[dict[tuple[str, str], str], dict[str, int]]:
    if not path.exists():
        return {}, {"complete_pairs": 0, "error_pairs": 0}
    hits: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    complete: set[tuple[str, str]] = set()
    errors: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = tuple(sorted((row["acc_low"], row["acc_high"])))
            if row.get("query_status") == "complete":
                complete.add(key)
                hits[key].append(row)
            else:
                errors.add(key)
    return (
        {key: classify_intact_hits(hits.get(key, [])) for key in complete},
        {"complete_pairs": len(complete), "error_pairs": len(errors - complete)},
    )


def load_train_similarity(path: Path) -> dict[str, dict[str, float | str]]:
    best: dict[str, dict[str, float | str]] = {}
    if not path.exists():
        return best
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(
            handle,
            delimiter="\t",
            fieldnames=["query", "target", "pident", "alnlen", "qcov", "tcov", "evalue", "bits"],
        ):
            candidate = {
                "target": row["target"],
                "pident": float(row["pident"]),
                "qcov": float(row["qcov"]),
                "tcov": float(row["tcov"]),
                "bits": float(row["bits"]),
            }
            current = best.get(row["query"])
            if current is None or float(candidate["bits"]) > float(current["bits"]):
                best[row["query"]] = candidate
    return best


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def logistic_irls(x: np.ndarray, y: np.ndarray, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(x.shape[1])
    for _ in range(max_iter):
        p = sigmoid(x @ beta)
        w = np.clip(p * (1.0 - p), 1e-7, None)
        z = x @ beta + (y - p) / w
        beta_new = np.linalg.pinv((x.T * w) @ x) @ ((x.T * w) @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta, sigmoid(x @ beta)


def cluster_meat(x: np.ndarray, residual: np.ndarray, groups: list[str]) -> np.ndarray:
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        grouped[group].append(i)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for indices in grouped.values():
        score = x[indices].T @ residual[indices]
        meat += np.outer(score, score)
    return meat


def two_way_cluster_cov(
    x: np.ndarray, residual: np.ndarray, bread: np.ndarray,
    group_a: list[str], group_c: list[str],
) -> np.ndarray:
    intersection = [f"{a}\x1f{c}" for a, c in zip(group_a, group_c)]
    meat = (
        cluster_meat(x, residual, group_a)
        + cluster_meat(x, residual, group_c)
        - cluster_meat(x, residual, intersection)
    )
    return bread @ meat @ bread


def proportion_ci(rows: list[dict[str, object]]) -> list[float] | None:
    if not rows:
        return None
    y = np.array([float(row["score"] >= 0.5) for row in rows])
    p = float(y.mean())
    if p <= 0.0 or p >= 1.0:
        # Wilson fallback for separation.
        n = len(y)
        z = 1.96
        den = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / den
        err = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return [max(0.0, centre - err), min(1.0, centre + err)]
    x = np.ones((len(rows), 1))
    bread = np.array([[1.0 / (len(rows) * p * (1 - p))]])
    cov = two_way_cluster_cov(
        x, y - p, bread,
        [str(row["family_a"]) for row in rows],
        [str(row["family_c"]) for row in rows],
    )
    se = math.sqrt(max(0.0, float(cov[0, 0])))
    logit = math.log(p / (1 - p))
    return [
        1 / (1 + math.exp(-(logit - 1.96 * se))),
        1 / (1 + math.exp(-(logit + 1.96 * se))),
    ]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    values = sorted(float(row["score"]) for row in rows)
    if not values:
        return {"n": 0, "mean": None, "median": None, "frac_ge_0.5": None, "ci95": None}
    n = len(values)
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    return {
        "n": n,
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(median), 4),
        "frac_ge_0.5": round(sum(v >= 0.5 for v in values) / n, 4),
        "ci95": [round(x, 4) for x in proportion_ci(rows)],
    }


def grouped_summary(rows: list[dict[str, object]], key: str, order: list[str]) -> dict[str, object]:
    return {
        value: summarize([row for row in rows if str(row[key]) == value])
        for value in order
    }


def familiarity_bin(value: float) -> str:
    if value < 20:
        return "0-20"
    if value < 40:
        return "20-40"
    if value < 60:
        return "40-60"
    if value < 80:
        return "60-80"
    return "80-100"


def proximity_bin(value: float) -> str:
    if value < 15:
        return "8-15 A"
    if value < 40:
        return "15-40 A"
    return ">=40 A"


def fit_primary_logistic(rows: list[dict[str, object]]) -> dict[str, object]:
    continuous = [
        ("log_complex_size", lambda r: math.log(float(r["num_protein_chains"]))),
        ("log_min_heavy", lambda r: math.log(float(r["min_heavy"]))),
        ("train_familiarity", lambda r: float(r["train_familiarity"]) / 100.0),
        ("ac_3mer_similarity", lambda r: float(r["kmer_jaccard"])),
        ("log_pair_length", lambda r: math.log(float(r["pair_tokens"]))),
        ("log1p_replicate_pdb", lambda r: math.log1p(float(r["n_pdb"]))),
        ("log1p_bridge_contacts", lambda r: math.log1p(float(r["bridge_contact_strength"]))),
    ]
    binary = [("human", lambda r: float(r["human"]))]
    association_n = sum(row["association_class"] == "association" for row in rows)
    if association_n >= 10:
        binary.append(("intact_association", lambda r: float(row_bool(r, "association_plus"))))

    columns = [np.ones(len(rows))]
    names = ["intercept"]
    scaling: dict[str, dict[str, float]] = {}
    for name, fn in continuous:
        raw = np.array([fn(row) for row in rows], dtype=float)
        mean, sd = float(raw.mean()), float(raw.std())
        columns.append((raw - mean) / sd if sd > 1e-12 else np.zeros_like(raw))
        names.append(name)
        scaling[name] = {"mean": round(mean, 5), "sd": round(sd, 5)}
    for name, fn in binary:
        columns.append(np.array([fn(row) for row in rows], dtype=float))
        names.append(name)
    x = np.column_stack(columns)
    y = np.array([float(row["score"] >= 0.5) for row in rows])
    beta, p = logistic_irls(x, y)
    w = np.clip(p * (1 - p), 1e-7, None)
    bread = np.linalg.pinv((x.T * w) @ x)
    cov = two_way_cluster_cov(
        x, y - p, bread,
        [str(row["family_a"]) for row in rows],
        [str(row["family_c"]) for row in rows],
    )
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    table = []
    for name, coefficient, error in zip(names, beta, se):
        table.append({
            "term": name,
            "coef_log_odds": round(float(coefficient), 4),
            "se_two_way_family": round(float(error), 4),
            "ci95": [
                round(float(coefficient - 1.96 * error), 4),
                round(float(coefficient + 1.96 * error), 4),
            ],
            "odds_ratio": round(float(math.exp(np.clip(coefficient, -20, 20))), 4),
        })

    marginal = []
    for index, name in enumerate(names[1:], 1):
        x0, x1 = x.copy(), x.copy()
        x0[:, index] = 0.0
        x1[:, index] = 1.0
        marginal.append({
            "term": name,
            "P_reference": round(float(sigmoid(x0 @ beta).mean()), 4),
            "P_plus1sd_or_on": round(float(sigmoid(x1 @ beta).mean()), 4),
            "delta_probability": round(float((sigmoid(x1 @ beta) - sigmoid(x0 @ beta)).mean()), 4),
        })
    return {
        "n": len(rows),
        "outcome": "published PLM-Interact score >= 0.5",
        "cluster": "two-way MMseqs50 family of A and C",
        "association_in_model": association_n >= 10,
        "association_n": association_n,
        "scaling": scaling,
        "coefficients": table,
        "marginal_effects": marginal,
    }


def row_bool(row: dict[str, object], key: str) -> bool:
    return bool(int(row[key]))


def analyze(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = args.output_dir / "figure5_pair_manifest_v2.tsv"
    if not manifest_path.exists():
        raise SystemExit("run ./work.sh figure5-train-similarity-v2 first")
    similarities = load_train_similarity(args.similarity_hits)
    intact, intact_qc = load_intact_cache(args.intact_cache)
    rows: list[dict[str, object]] = []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            row: dict[str, object] = dict(raw)
            for key in (
                "score", "min_heavy", "min_cb", "kmer_jaccard",
            ):
                row[key] = float(raw[key])
            for key in (
                "len_a", "len_c", "pair_tokens", "truncated", "train_pair",
                "n_train_proteins", "core_flag", "num_protein_chains", "n_pdb",
                "human", "bridge_contact_strength",
            ):
                row[key] = int(raw[key])
            sim_a = similarities.get(raw["qid_a"], {})
            sim_c = similarities.get(raw["qid_c"], {})
            identity_a = float(sim_a.get("pident", 0.0))
            identity_c = float(sim_c.get("pident", 0.0))
            row["train_identity_a"] = identity_a
            row["train_identity_c"] = identity_c
            row["train_familiarity"] = min(identity_a, identity_c)
            row["familiarity_bin"] = familiarity_bin(float(row["train_familiarity"]))
            row["proximity_bin"] = proximity_bin(float(row["min_heavy"]))
            key = tuple(sorted((raw["protein_a"], raw["protein_c"])))
            row["association_class"] = intact.get(key, "intact_unqueried")
            row["association_plus"] = int(row["association_class"] == "association")
            rows.append(row)

    core_pre_intact = [
        row for row in rows
        if row_bool(row, "core_flag") and not row_bool(row, "truncated") and not row_bool(row, "train_pair")
    ]
    primary = [row for row in core_pre_intact if row["association_class"] != "direct_conflict"]
    all_no_overlap = [row for row in rows if not row_bool(row, "train_pair")]
    exact_overlap = [row for row in rows if row_bool(row, "train_pair")]
    clean_nontruncated = [
        row for row in rows
        if not row_bool(row, "truncated") and not row_bool(row, "train_pair")
        and row["association_class"] != "direct_conflict"
    ]
    intact_primary = [row for row in core_pre_intact if row["association_class"] != "intact_unqueried"]

    report: dict[str, object] = {
        "protocol": {
            "model": "published PLM-Interact humanV12; no new model inference",
            "primary_cohort": "core 3-24 chains; pair tokens <=1024; exact STRING-positive overlap removed; IntAct direct conflicts removed",
            "graph_distance": "not estimable: A-C pairs were selected as open wedges A-B-C and therefore have d_graph=2 by construction",
            "uncertainty": "two-way cluster-robust by MMseqs50 families of A and C",
        },
        "cohort_counts": {
            "all": len(rows),
            "exact_string_positive_overlap": len(exact_overlap),
            "truncated": sum(row_bool(row, "truncated") for row in rows),
            "core_pre_intact": len(core_pre_intact),
            "intact_direct_conflicts_removed": len(core_pre_intact) - len(primary),
            "primary_clean_core": len(primary),
            "primary_human": sum(row_bool(row, "human") for row in primary),
        },
        "5a_training_exposure": {
            "mmseqs_query_sequences_with_hit": len(similarities),
            "mmseqs_query_sequences_total": len(
                {str(row["qid_a"]) for row in rows} | {str(row["qid_c"]) for row in rows}
            ),
            "all": summarize(rows),
            "exact_string_positive": summarize(exact_overlap),
            "no_exact_string_positive": summarize(all_no_overlap),
            "primary_clean_core": summarize(primary),
            "by_endpoint_train_familiarity": grouped_summary(
                primary, "familiarity_bin", ["0-20", "20-40", "40-60", "60-80", "80-100"],
            ),
        },
        "5b_complex_size": {
            "primary_and_exploratory": grouped_summary(
                clean_nontruncated, "assembly_size_stratum", ["3_10", "11_24", "gt24"],
            ),
            "gt24_status": "exploratory Structure-C; excluded from primary regression",
        },
        "5c_structural_proximity": {
            "primary_by_min_heavy": grouped_summary(
                primary, "proximity_bin", ["8-15 A", "15-40 A", ">=40 A"],
            ),
            "spearman_min_heavy_score": spearman(
                [float(row["min_heavy"]) for row in primary],
                [float(row["score"]) for row in primary],
            ),
            "spearman_min_cb_score": spearman(
                [float(row["min_cb"]) for row in primary],
                [float(row["score"]) for row in primary],
            ),
        },
        "5d_intact_association": {
            "query_qc": intact_qc,
            "coverage_in_core_pre_intact": sum(row["association_class"] != "intact_unqueried" for row in core_pre_intact),
            "by_class": grouped_summary(
                intact_primary, "association_class",
                ["no_recorded_association", "association", "other_intact", "direct_conflict"],
            ),
            "interpretation_guard": "no_recorded_association means no positive record in the targeted IntAct query, not experimentally proven absence",
        },
        "5e_multivariable": fit_primary_logistic(primary),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figure5_score_drivers_v2.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    feature_path = args.output_dir / "figure5_ac_features_v2.tsv"
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "primary_n": len(primary),
        "intact_complete": intact_qc["complete_pairs"],
        "output": str(args.output_dir / "figure5_score_drivers_v2.json"),
    }, indent=2), flush=True)
    return report


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx = rankdata(xs)
    ry = rankdata(ys)
    value = np.corrcoef(rx, ry)[0, 1]
    return round(float(value), 4) if np.isfinite(value) else None


def rankdata(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        prepare_inputs(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
