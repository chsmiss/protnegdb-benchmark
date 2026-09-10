#!/usr/bin/env python3
"""Revision-v2 audits linking negative semantics, domain shift and controls.

This analysis performs no model training and no new model inference. It:
1. applies one harmonized assay ontology to Negatome and ProtNeg;
2. reports human-only structural score and partner-ranking results;
3. audits selection from 480 reviewed ProtNeg records to scoreable records;
4. checks covariate balance of the locked Structure/Random fine-tuning arms.

Structural noncontact always means direct noncontact in the observed assembly.
Random pairs remain unlabeled controls rather than asserted non-interactions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))

import analyze_negatome_figure4_subset_v1 as neg4


NEG_DIR = ROOT / "data/interim/negatome_manual_plminteract_v1"
FIG6_DIR = ROOT / "data/interim/figure6_protneg_assay_v1"
FIG5_DIR = ROOT / "data/interim/figure5_structure_score_drivers_v2"
FIG12_DIR = ROOT / "data/interim/figure12_master_v1"
FT_DIR = ROOT / "data/interim/structure_family_transfer_v4/id50"
DEFAULT_OUT = ROOT / "data/interim/revision_v2_semantic_robustness"
SWISSPROT = Path("/data/chs/12.codebuddy/bridge/data/uniprot_sprot.fasta.gz")
META = Path("/data/chs/01.data/15.pdb/pdb_candidates/uniprot_external_metadata.tsv")
STRING_TRAIN = ROOT / "data/external/model_training_sets/plm_interact/string_v12/protein.pairs_human_V12_train.tsv"

# The same three-level semantics are applied to both resources. The primary
# cross-resource comparison uses reconstituted_direct. Cell-based binary or
# proximity assays are retained as a boundary layer, not merged with co-complex.
NEG_RECONSTITUTED_MI = {
    "0009", "0012", "0047", "0059", "0065", "0096", "0107", "0114",
    "0405", "0411", "0415", "0848", "0892", "0921",
}
NEG_CELL_BINARY_MI = {"0016", "0018", "0030", "0055", "0099"}
NEG_COCOMPLEX_MI = set(neg4.ASSOCIATION_MI)

PROT_RECONSTITUTED = {
    "SPR", "ITC", "BLI", "MST", "fluorescence_anisotropy",
    "pull_down", "in_vitro_binding_unspecified", "far_western",
}
PROT_BIOPHYSICAL = {"SPR", "ITC", "BLI", "MST", "fluorescence_anisotropy"}
PROT_CELL_BINARY = {"Y2H", "PLA", "BiFC", "FRET"}
PROT_COCOMPLEX = {"coIP"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260817)
    return p.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h, delimiter="\t"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as h:
        return list(csv.DictReader(h))


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [centre - half, centre + half]


def score_stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"n": 0, "median": None, "mean": None, "frac_ge_0.5": None,
                "frac_ge_0.5_ci95": None}
    k = sum(v >= 0.5 for v in values)
    return {
        "n": len(values), "median": median(values), "mean": sum(values) / len(values),
        "frac_ge_0.5": k / len(values), "frac_ge_0.5_ci95": wilson(k, len(values)),
    }


def bootstrap_mean(values: list[float], n_boot: int, seed: int) -> list[float]:
    if not values:
        return []
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        out.append(sum(rng.choice(values) for _ in values) / len(values))
    out.sort()
    return out


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def ranking_stats(rows: list[dict[str, str]], model: str, n_boot: int, seed: int) -> dict[str, object]:
    ad_key, ac_key = f"score_AD_{model}", f"score_AC_{model}"
    by_ac: dict[str, list[float]] = defaultdict(list)
    n_rows = 0
    for row in rows:
        if not row.get(ad_key) or not row.get(ac_key):
            continue
        ad, ac = float(row[ad_key]), float(row[ac_key])
        by_ac[row["ac_id"]].append(1.0 if ad > ac else 0.5 if ad == ac else 0.0)
        n_rows += 1
    macro = [sum(v) / len(v) for v in by_ac.values()]
    boot = bootstrap_mean(macro, n_boot, seed)
    return {
        "triplets_scored": n_rows,
        "unique_ac": len(macro),
        "p_d_gt_c_macro": sum(macro) / len(macro) if macro else None,
        "ci95": [percentile(boot, 0.025), percentile(boot, 0.975)] if boot else None,
        "ties": sum(
            1 for row in rows if row.get(ad_key) and row.get(ac_key)
            and float(row[ad_key]) == float(row[ac_key])
        ),
    }


def neg_ontology(codes: set[str]) -> str:
    if codes & NEG_RECONSTITUTED_MI:
        return "reconstituted_direct"
    if codes & NEG_CELL_BINARY_MI:
        return "cell_binary_proximity"
    if codes & NEG_COCOMPLEX_MI:
        return "co_complex_association"
    return "other_or_unresolved"


def prot_ontology(assay: str) -> str:
    if assay in PROT_RECONSTITUTED:
        return "reconstituted_direct"
    if assay in PROT_CELL_BINARY:
        return "cell_binary_proximity"
    if assay in PROT_COCOMPLEX:
        return "co_complex_association"
    return "other_or_unresolved"


def prepare_negatome() -> list[dict[str, object]]:
    pairs = read_tsv(NEG_DIR / "negatome_manual_pairs_v1.tsv")
    seq_pairs = read_csv(NEG_DIR / "negatome_manual_pairs_v1.csv")
    scores = [float(r["score"]) for r in read_csv(NEG_DIR / "negatome_manual_scores_v1.csv")]
    accessions = {r["protein_a"] for r in pairs} | {r["protein_b"] for r in pairs}
    tax = neg4.load_swissprot_tax(SWISSPROT, accessions)
    rows = neg4.attach_flags(pairs, scores, tax, neg4.MAX_LENGTH)
    train_keys = neg4.load_string_positive_keys(STRING_TRAIN)
    if len(rows) != len(seq_pairs):
        raise SystemExit("Negatome pair/sequence alignment mismatch")
    out = []
    for row, seq in zip(rows, seq_pairs):
        codes = set(row["mi_codes"])
        overlap = neg4.seq_pair_key(seq["query"], seq["text"]) in train_keys
        eligible = bool(row["stringent"] and row["canonical"] and row["human_human"]
                        and row["pair_fits"] and not overlap)
        out.append({
            "source": "Negatome", "record_id": f"NEG:{len(out)+1:05d}",
            "protein_a": row["protein_a"], "protein_b": row["protein_b"],
            "taxon_a": row["tax_a"], "taxon_b": row["tax_b"],
            "assay_raw": row["methods"], "assay_ontology": neg_ontology(codes),
            "old_historical_class": row["method_class"],
            "eligible_common_qc": int(eligible), "score": float(row["score"]),
            "pair_tokens": int(row["len_a"]) + int(row["len_b"]) + 3,
            "exact_string_positive_overlap": int(overlap),
            "biophysical_only": int(bool(codes & {"0009", "0012", "0065", "0107", "0921"})),
        })
    return out


def prepare_protneg() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    manifest = read_tsv(FIG6_DIR / "protneg_assay_manifest_v1.tsv")
    scored = {r["pair2_key"]: r for r in read_tsv(FIG6_DIR / "figure6_scored_manifest_v1.tsv")}
    out = []
    for row in manifest:
        srow = scored.get(row["pair2_key"])
        eligible = bool(srow and row["exact_negatome_pair_overlap"] == "0")
        out.append({
            "source": "ProtNeg", "record_id": row["pair2_key"],
            "protein_a": row["accession_a"], "protein_b": row["accession_b"],
            "taxon_a": row["a_taxid"], "taxon_b": row["b_taxid"],
            "assay_raw": row["assay_normalized"],
            "assay_ontology": prot_ontology(row["assay_normalized"]),
            "old_historical_class": row["evidence_semantics"],
            "eligible_common_qc": int(eligible),
            "score": float(srow["score"]) if srow else "",
            "pair_tokens": int(row["pair_tokens"] or 0),
            "exact_string_positive_overlap": int(row["exact_string_positive_overlap"] or 0),
            "biophysical_only": int(row["assay_normalized"] in PROT_BIOPHYSICAL),
        })
    return out, manifest


def ontology_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for source in ("Negatome", "ProtNeg"):
        sub = [r for r in rows if r["source"] == source and r["eligible_common_qc"] == 1]
        groups = {}
        for cls in ("reconstituted_direct", "cell_binary_proximity", "co_complex_association",
                    "other_or_unresolved"):
            values = [float(r["score"]) for r in sub if r["assay_ontology"] == cls]
            groups[cls] = score_stats(values)
        bio = [float(r["score"]) for r in sub if r["biophysical_only"] == 1]
        result[source] = {"groups": groups, "biophysical_sensitivity": score_stats(bio)}
    return result


def human_structural(n_boot: int, seed: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    ac_master = read_tsv(FIG12_DIR / "figure1_ac_master_v1.tsv")
    features = {r["ac_id"]: r for r in read_tsv(FIG5_DIR / "figure5_ac_features_v2.tsv")}
    core = [r for r in ac_master if r["core_flag"] == "1"]
    human = [r for r in core if r["taxon"] == "9606"]
    nonhuman = [r for r in core if r["taxon"] != "9606"]
    rows_out = []
    for group, rows in (("human_human", human), ("nonhuman", nonhuman)):
        for row in rows:
            feat = features[row["ac_id"]]
            rows_out.append({"group": group, "ac_id": row["ac_id"], "protein_a": row["A_uniprot"],
                             "protein_c": row["C_uniprot"], "taxon": row["taxon"],
                             "score": float(feat["score"]), "train_pair": feat["train_pair"],
                             "truncated": feat["truncated"], "family_a": row["mmseqs_family_A"],
                             "family_c": row["mmseqs_family_C"]})
    triplets = [r for r in read_tsv(FIG12_DIR / "figure2_triplet_master_v1.tsv")
                if r["core_flag"] == "1"]
    metadata = {r["accession"]: r for r in read_tsv(META)}

    def is_human_triplet(r: dict[str, str]) -> bool:
        return all(metadata.get(r[x], {}).get("tax_id") == "9606" for x in ("A", "C", "D"))

    htrip = [r for r in triplets if is_human_triplet(r)]
    ntrip = [r for r in triplets if not is_human_triplet(r)]
    assignments = {r["triplet_id"]: r for r in read_tsv(FT_DIR / "triplet_assignments_v4.tsv")}
    h_unseen = [r for r in htrip if assignments.get(r["triplet_id"], {}).get("split") == "family_unseen"]
    return {
        "score_distribution": {
            "human_human": score_stats([float(features[r["ac_id"]]["score"]) for r in human]),
            "nonhuman": score_stats([float(features[r["ac_id"]]["score"]) for r in nonhuman]),
        },
        "partner_ranking": {
            model: {
                "human_human": ranking_stats(htrip, model, n_boot, seed + i),
                "nonhuman": ranking_stats(ntrip, model, n_boot, seed + 10 + i),
            }
            for i, model in enumerate(("PLMInteract", "DSCRIPT", "TopsyTurvy"))
        },
        "current_family_unseen_human_original": ranking_stats(
            h_unseen, "PLMInteract", n_boot, seed + 50),
    }, rows_out


def kmer_jaccard(a: str, b: str, k: int = 3) -> float:
    aa = {a[i:i+k] for i in range(max(0, len(a)-k+1))}
    bb = {b[i:i+k] for i in range(max(0, len(b)-k+1))}
    return len(aa & bb) / len(aa | bb) if aa or bb else 0.0


def smd(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt((va + vb) / 2)
    return (statistics.mean(a) - statistics.mean(b)) / pooled if pooled else 0.0


def random_arm_balance() -> tuple[dict[str, object], list[dict[str, object]]]:
    matches = read_tsv(FT_DIR / "train_matched_random_v4.tsv")
    assignments = {r["triplet_id"]: r for r in read_tsv(FT_DIR / "triplet_assignments_v4.tsv")}
    features = read_tsv(FIG5_DIR / "figure5_ac_features_v2.tsv")
    identity: dict[str, float] = {}
    for r in features:
        identity[r["protein_a"]] = float(r["train_identity_a"])
        identity[r["protein_c"]] = float(r["train_identity_c"])
    metadata = {r["accession"]: r for r in read_tsv(META)}
    clusters: dict[str, str] = {}
    for line in (ROOT / "data/interim/structure_family_transfer_v4/clusters_id50.tsv").read_text().splitlines():
        rep, member = line.split("\t")[:2]
        clusters[member] = rep
    rows = []
    for match in matches:
        a, cs, cr = match["anchor"], match["c_structure"], match["c_random"]
        if any(p not in metadata for p in (a, cs, cr)):
            raise SystemExit(f"missing metadata for balance triplet {match['triplet_id']}")
        sa, ss, sr = (metadata[p]["canonical_sequence"] for p in (a, cs, cr))
        for arm, c, seqc in (("Structure", cs, ss), ("Random", cr, sr)):
            rows.append({
                "triplet_id": match["triplet_id"], "arm": arm, "anchor": a, "negative": c,
                "pair_tokens": len(sa) + len(seqc) + 3,
                "kmer_jaccard": kmer_jaccard(sa, seqc),
                "train_identity_anchor": identity[a], "train_identity_negative": identity[c],
                "train_familiarity": min(identity[a], identity[c]),
                "taxon": metadata[c]["tax_id"], "family_negative": clusters.get(c, c),
                "known_contact_rate": 0,
                "ac_id": assignments.get(match["triplet_id"], {}).get("ac_id", ""),
            })
    balance_rows = []
    variables = ["pair_tokens", "kmer_jaccard", "train_identity_anchor",
                 "train_identity_negative", "train_familiarity"]
    for var in variables:
        s = [float(r[var]) for r in rows if r["arm"] == "Structure"]
        q = [float(r[var]) for r in rows if r["arm"] == "Random"]
        paired = [x-y for x, y in zip(s, q)]
        balance_rows.append({
            "variable": var, "structure_mean": statistics.mean(s),
            "random_mean": statistics.mean(q), "smd_structure_minus_random": smd(s, q),
            "paired_mean_difference": statistics.mean(paired),
        })
    sneg = [r["negative"] for r in rows if r["arm"] == "Structure"]
    rneg = [r["negative"] for r in rows if r["arm"] == "Random"]
    sfam = Counter(r["family_negative"] for r in rows if r["arm"] == "Structure")
    rfam = Counter(r["family_negative"] for r in rows if r["arm"] == "Random")
    stax = Counter(r["taxon"] for r in rows if r["arm"] == "Structure")
    rtax = Counter(r["taxon"] for r in rows if r["arm"] == "Random")
    return {
        "n_triplets": len(matches),
        "negative_protein_multiset_identical": Counter(sneg) == Counter(rneg),
        "negative_family_multiset_identical": sfam == rfam,
        "negative_taxon_multiset_identical": stax == rtax,
        "known_contact_rate_both": 0.0,
        "balance": balance_rows,
        "max_abs_smd": max(abs(float(r["smd_structure_minus_random"])) for r in balance_rows),
    }, rows


def categorical_selection(manifest: list[dict[str, str]], field: str) -> list[dict[str, object]]:
    selected = [r for r in manifest if r["reliable_reported_taxon_mapping"] == "1"
                and r["sequence_available"] == "1"]
    excluded = [r for r in manifest if r not in selected]
    levels = sorted({r[field] or "missing" for r in manifest})
    out = []
    for level in levels:
        ns = sum((r[field] or "missing") == level for r in selected)
        ne = sum((r[field] or "missing") == level for r in excluded)
        ps, pe = ns / len(selected), ne / len(excluded)
        pooled = math.sqrt(max(1e-12, (ps*(1-ps)+pe*(1-pe))/2))
        out.append({"field": field, "level": level, "scored_n": ns, "excluded_n": ne,
                    "scored_fraction": ps, "excluded_fraction": pe,
                    "standardized_difference": (ps-pe)/pooled})
    return out


def selection_audit(manifest: list[dict[str, str]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = [r for r in manifest if r["reliable_reported_taxon_mapping"] == "1"
                and r["sequence_available"] == "1"]
    reasons = Counter()
    for r in manifest:
        if r in selected:
            reasons["scoreable_reported_taxon_and_sequence"] += 1
        elif r["reliable_reported_taxon_mapping"] != "1":
            reasons[f"mapping:{r['pair_mapping_status']}"] += 1
        else:
            reasons["mapped_but_sequence_unavailable"] += 1
    comp = []
    for field in ("assay_normalized", "a_taxid", "pair_mapping_status"):
        comp.extend(categorical_selection(manifest, field))
    assay_rows = [r for r in comp if r["field"] == "assay_normalized"]
    return {
        "source_n": len(manifest), "scoreable_n": len(selected),
        "excluded_n": len(manifest)-len(selected), "failure_reasons": dict(reasons),
        "max_abs_assay_standardized_difference": max(abs(float(r["standardized_difference"]))
                                                     for r in assay_rows),
        "note": "Publication-year metadata is added in a separate provenance step.",
    }, comp


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    neg = prepare_negatome()
    prot, prot_manifest = prepare_protneg()
    combined = neg + prot
    ontology = ontology_summary(combined)
    human, human_rows = human_structural(args.bootstrap, args.seed)
    balance, balance_pairs = random_arm_balance()
    selection, selection_rows = selection_audit(prot_manifest)

    manifest_fields = ["source", "record_id", "protein_a", "protein_b", "taxon_a", "taxon_b",
                       "assay_raw", "assay_ontology", "old_historical_class", "eligible_common_qc",
                       "score", "pair_tokens", "exact_string_positive_overlap", "biophysical_only"]
    write_tsv(args.output_dir / "harmonized_negative_manifest_v2.tsv", combined, manifest_fields)
    write_tsv(args.output_dir / "human_structural_manifest_v2.tsv", human_rows,
              list(human_rows[0]) if human_rows else [])
    write_tsv(args.output_dir / "random_arm_balance_pairs_v2.tsv", balance_pairs,
              list(balance_pairs[0]) if balance_pairs else [])
    write_tsv(args.output_dir / "protneg_selection_composition_v2.tsv", selection_rows,
              list(selection_rows[0]) if selection_rows else [])

    result = {
        "status": "revision_v2_semantic_robustness_no_new_inference",
        "ontology_definition": {
            "reconstituted_direct": "biophysical or purified/reconstituted pair assay",
            "cell_binary_proximity": "Y2H/BRET/FRET/BiFC/PLA/crosslink/proximity boundary layer",
            "co_complex_association": "coIP/AP-MS/TAP/affinity co-complex evidence",
        },
        "harmonized_ontology": ontology,
        "human_only_structural": human,
        "matched_random_balance": balance,
        "protneg_selection_audit": selection,
    }
    (args.output_dir / "revision_v2_semantic_robustness.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
