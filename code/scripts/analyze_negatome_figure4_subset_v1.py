#!/usr/bin/env python3
"""Filter scored Negatome Manual pairs to a PLM-Interact-compatible subset.

No new training and no new model scores. Row-aligned published humanV12
scores are reused. Labels stay literature non-interaction; they are not
merged with assembly-context noncontact.

Main subset (Figure 4):
  Manual-stringent
  AND both proteins human (Swiss-Prot OX=9606)
  AND canonical accessions (no isoform suffix in the Negatome raw IDs)
  AND pair fits the 1024-token window without truncation
  AND at least one direct/binary detection method
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

MI_RE = re.compile(r"MI:(\d+)")
OX_RE = re.compile(r"OX=(\d+)")
SP_HEADER_RE = re.compile(r"^>sp\|([A-Z0-9]+)\|")
ISOFORM_RE = re.compile(r"-\d+$")

# Binary / biophysical methods that can support a pair-level physical
# non-interaction. Association-only methods (co-IP, TAP, affinity) are
# kept out of the direct subset.
DIRECT_MI = {
    "0009",  # SPR
    "0012",  # NMR (Negatome sometimes names this BRET; keep the code)
    "0016",  # BRET
    "0018",  # two hybrid
    "0030",  # cross-linking
    "0047",  # far western
    "0055",  # FRET
    "0059",  # GST pull down
    "0065",  # ITC
    "0096",  # pull down
    "0099",  # scintillation proximity
    "0107",  # SPR
    "0114",  # X-ray
    "0405",  # competition binding
    "0411",  # ELISA
    "0415",  # enzymatic study
    "0848",  # radiolabel
    "0892",  # solid phase
    "0921",  # SPR array
}
ASSOCIATION_MI = {
    "0004",  # affinity chromatography
    "0006",  # anti bait co-IP
    "0007",  # anti tag co-IP
    "0019",  # coimmunoprecipitation
    "0027",  # cosedimentation
    "0071",  # molecular sieving
    "0676",  # TAP
}
TEXT_MINING_MI = {"0110"}
SPECIAL_TOKENS = 3
MAX_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-tsv", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, default=None,
                        help="query/text CSV aligned with --pairs-tsv; used for STRING overlap")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--swissprot", type=Path, required=True)
    parser.add_argument("--structure-eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--string-train", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    return parser.parse_args()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    err = z * math.sqrt((p * (1.0 - p) / n) + z * z / (4.0 * n * n)) / den
    return round(centre - err, 4), round(centre + err, 4)


def stats(values: list[float]) -> dict[str, object]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "frac_ge_0.5": None,
                "frac_ge_0.5_ci95": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    k = sum(v >= 0.5 for v in values)
    n = len(values)
    ci = wilson(k, n)
    return {
        "n": n,
        "mean": round(sum(values) / n, 4),
        "median": round(median, 4),
        "frac_ge_0.5": round(k / n, 4),
        "frac_ge_0.5_ci95": list(ci) if ci else None,
        "n_ge_0.5": k,
    }


def extract_mi_codes(methods: str) -> set[str]:
    return set(MI_RE.findall(methods or ""))


def has_isoform_raw_id(raw_ids: str) -> bool:
    return any(ISOFORM_RE.search(part.strip()) for part in raw_ids.split(";") if part.strip())


def pair_fits_window(len_a: int, len_b: int, max_length: int = MAX_LENGTH) -> bool:
    return len_a + len_b + SPECIAL_TOKENS <= max_length


def method_class(codes: set[str]) -> str:
    if codes & DIRECT_MI:
        return "direct_binary"
    if codes & ASSOCIATION_MI:
        return "association"
    if codes and codes <= TEXT_MINING_MI:
        return "text_mining"
    if not codes:
        return "missing"
    return "other"


def load_swissprot_tax(path: Path, accessions: set[str]) -> dict[str, str]:
    tax: dict[str, str] = {}
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">sp|"):
                continue
            match = SP_HEADER_RE.match(line)
            if not match:
                continue
            acc = match.group(1)
            if acc not in accessions or acc in tax:
                continue
            ox = OX_RE.search(line)
            if ox:
                tax[acc] = ox.group(1)
            if len(tax) == len(accessions):
                break
    return tax


def seq_pair_key(seq_a: str, seq_b: str) -> bytes:
    left, right = (seq_a, seq_b) if seq_a <= seq_b else (seq_b, seq_a)
    return hashlib.sha1(f"{left}\t{right}".encode("utf-8")).digest()


def load_string_positive_keys(path: Path) -> set[bytes]:
    if str(path).endswith('.sha1.tsv'):
        return {bytes.fromhex(line.strip()) for line in Path(path).read_text().splitlines() if line.strip()}
    keys: set[bytes] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("label") != "1":
                continue
            keys.add(seq_pair_key(row["query"], row["text"]))
    return keys


def attach_flags(
    pairs: list[dict[str, str]],
    scores: list[float],
    tax: dict[str, str],
    max_length: int,
) -> list[dict[str, object]]:
    if len(pairs) != len(scores):
        raise SystemExit(f"pair/score mismatch {len(pairs)} vs {len(scores)}")
    rows = []
    for pair, score in zip(pairs, scores):
        codes = extract_mi_codes(pair["methods"])
        len_a, len_b = int(pair["len_a"]), int(pair["len_b"])
        tax_a, tax_b = tax.get(pair["protein_a"], ""), tax.get(pair["protein_b"], "")
        rows.append({
            **pair,
            "score": score,
            "mi_codes": codes,
            "method_class": method_class(codes),
            "stringent": pair["in_manual_stringent"] == "1",
            "canonical": not has_isoform_raw_id(pair["raw_ids"]),
            "human_human": tax_a == "9606" and tax_b == "9606",
            "same_taxon": bool(tax_a) and tax_a == tax_b,
            "pair_fits": pair_fits_window(len_a, len_b, max_length),
            "each_le_max": len_a <= max_length and len_b <= max_length,
            "tax_a": tax_a,
            "tax_b": tax_b,
        })
    return rows


def select(rows: list[dict[str, object]], **flags: bool) -> list[dict[str, object]]:
    out = []
    for row in rows:
        if all(bool(row[name]) is value for name, value in flags.items()):
            out.append(row)
    return out


def scores_of(rows: list[dict[str, object]]) -> list[float]:
    return [float(row["score"]) for row in rows]


def method_breakdown(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method_class"])].append(float(row["score"]))
    return {key: stats(vals) for key, vals in sorted(grouped.items())}


def funnel(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    steps = [
        ("manual_scored", rows),
        ("manual_stringent", select(rows, stringent=True)),
        ("stringent_canonical", select(rows, stringent=True, canonical=True)),
        ("stringent_canonical_human",
         select(rows, stringent=True, canonical=True, human_human=True)),
        ("stringent_canonical_human_length",
         select(rows, stringent=True, canonical=True, human_human=True, pair_fits=True)),
        ("figure4_direct_subset",
         [row for row in select(rows, stringent=True, canonical=True,
                                 human_human=True, pair_fits=True)
          if row["method_class"] == "direct_binary"]),
    ]
    out = []
    for name, subset in steps:
        item = {"filter": name, **stats(scores_of(subset))}
        out.append(item)
    return out


def main() -> None:
    args = parse_args()
    pairs = list(csv.DictReader(args.pairs_tsv.open(encoding="utf-8", newline=""),
                                delimiter="\t"))
    score_rows = list(csv.DictReader(args.scores.open(encoding="utf-8", newline="")))
    scores = [float(row["score"]) for row in score_rows]
    accessions = {row["protein_a"] for row in pairs} | {row["protein_b"] for row in pairs}
    tax = load_swissprot_tax(args.swissprot, accessions)
    if len(tax) != len(accessions):
        missing = sorted(accessions - set(tax))[:8]
        raise SystemExit(f"missing Swiss-Prot tax for {len(accessions) - len(tax)} e.g. {missing}")
    rows = attach_flags(pairs, scores, tax, args.max_length)
    figure4 = [row for row in select(rows, stringent=True, canonical=True,
                                     human_human=True, pair_fits=True)
               if row["method_class"] == "direct_binary"]
    structure = json.loads(args.structure_eval_json.read_text(encoding="utf-8"))
    plm = structure["models"]["plminteract_base"]["by_role"]

    string_overlap = None
    if args.string_train and args.string_train.is_file():
        pair_csv = args.pairs_csv or (args.pairs_tsv.parent / "negatome_manual_pairs_v1.csv")
        seqs = list(csv.DictReader(pair_csv.open(encoding="utf-8", newline="")))
        if len(seqs) != len(rows):
            raise SystemExit(f"sequence csv mismatch {len(seqs)} vs {len(rows)}")
        keys = load_string_positive_keys(args.string_train)
        n_all = 0
        n_fig = 0
        fig_clean = []
        for i, row in enumerate(rows):
            in_train = seq_pair_key(seqs[i]["query"], seqs[i]["text"]) in keys
            if in_train:
                n_all += 1
            is_fig = (row["stringent"] and row["canonical"] and row["human_human"]
                      and row["pair_fits"] and row["method_class"] == "direct_binary")
            if is_fig and in_train:
                n_fig += 1
            if is_fig and not in_train:
                fig_clean.append(row)
        string_overlap = {
            "train_positive_pairs": len(keys),
            "negatome_scored_in_train_positives": n_all,
            "figure4_subset_in_train_positives": n_fig,
        }
    else:
        fig_clean = figure4

    result = {
        "status": "negatome_figure4_experimental_negative_subset_v1",
        "model": "PLM-interact-650M-humanV12 published; no new training",
        "label_semantics": "literature_noninteraction_not_assembly_noncontact",
        "max_length": args.max_length,
        "direct_mi": sorted(DIRECT_MI),
        "funnel": funnel(rows),
        "figure4_subset": {
            **stats(scores_of(figure4)),
            "definition": (
                "manual_stringent AND human-human AND canonical AND "
                "len_a+len_b+3<=1024 AND >=1 direct/binary MI method"
            ),
        },
        "stringent_human_canonical": stats(scores_of(
            select(rows, stringent=True, canonical=True, human_human=True))),
        "stringent_method_breakdown": method_breakdown(select(rows, stringent=True)),
        "stringent_human_canonical_method_breakdown": method_breakdown(
            select(rows, stringent=True, canonical=True, human_human=True)),
        "comparators_published": {
            "structure_ac_negative": plm["ac_negative"],
            "structure_ad_positive": plm["ad_positive"],
            "structure_random_unlabeled": plm["random_unlabeled"],
        },
        "swissprot_tax_coverage": len(tax),
        "string_train_overlap": string_overlap,
        "figure4_excluding_string_train_positives": stats(scores_of(fig_clean)),
        "note": (
            "Figure 4 tests whether published PLM-Interact's high scores on "
            "assembly noncontacts also appear on experimentally supported "
            "non-interactions after matching the model's sequence/species/"
            "length/direct-assay regime. Do not merge the two negative labels."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "funnel": result["funnel"],
        "figure4_subset": {k: result["figure4_subset"][k]
                           for k in ("n", "mean", "median", "frac_ge_0.5",
                                     "frac_ge_0.5_ci95")},
        "string_train_overlap": string_overlap,
        "figure4_excluding_string_train_positives": result["figure4_excluding_string_train_positives"],
        "stringent_method_breakdown": result["stringent_method_breakdown"],
        "stringent_human_canonical_method_breakdown": result["stringent_human_canonical_method_breakdown"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
