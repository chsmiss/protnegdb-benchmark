"""Prepare Negatome 2.0 manual literature negatives for PLM-interact scoring.

Official Manual set is literature-curated non-interactions (not PDB
non-contacts). Pairs are undirected, isoform suffixes fall back to the
parent accession, and both sequences must resolve in Swiss-Prot or the
local UniProt metadata table.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

HEADER_RE = re.compile(r"^>(?:sp|tr)\|([A-Z0-9]+)\|")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--manual-stringent", type=Path, required=True)
    parser.add_argument("--swissprot", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parent_acc(raw: str) -> str:
    acc = raw.strip()
    return acc.split("-", 1)[0] if "-" in acc else acc


def pair_key(a: str, b: str) -> tuple[str, str]:
    left, right = sorted((parent_acc(a), parent_acc(b)))
    return left, right


def load_negatome(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            acc_a, acc_b = fields[0].strip(), fields[1].strip()
            if not acc_a or not acc_b:
                continue
            rows.append({
                "acc_a": acc_a,
                "acc_b": acc_b,
                "pmid": fields[2].strip() if len(fields) > 2 else "",
                "method": fields[3].strip() if len(fields) > 3 else "",
            })
    return rows


def collapse_undirected(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = pair_key(row["acc_a"], row["acc_b"])
        if key[0] == key[1]:
            continue
        item = grouped.setdefault(key, {
            "protein_a": key[0],
            "protein_b": key[1],
            "n_rows": "0",
            "pmids": "",
            "methods": "",
            "raw_ids": "",
        })
        item["n_rows"] = str(int(item["n_rows"]) + 1)
        pmids = set(p for p in item["pmids"].split(";") if p)
        if row["pmid"]:
            pmids.add(row["pmid"])
        item["pmids"] = ";".join(sorted(pmids))
        methods = set(m for m in item["methods"].split(" | ") if m)
        if row["method"]:
            methods.add(row["method"])
        item["methods"] = " | ".join(sorted(methods))
        raw = set(x for x in item["raw_ids"].split(";") if x)
        raw.update((row["acc_a"], row["acc_b"]))
        item["raw_ids"] = ";".join(sorted(raw))
    return [grouped[key] for key in sorted(grouped)]


def load_swissprot(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    acc = None
    chunks: list[str] = []
    handle_cm = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(
        encoding="utf-8")
    with handle_cm as handle:
        for line in handle:
            if line.startswith(">"):
                if acc and chunks:
                    seqs[acc] = "".join(chunks)
                match = HEADER_RE.match(line)
                acc = match.group(1) if match else None
                chunks = []
            elif acc:
                chunks.append(line.strip())
        if acc and chunks:
            seqs[acc] = "".join(chunks)
    return seqs


def load_metadata(path: Path) -> dict[str, str]:
    seqs = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            acc, seq = row.get("accession", ""), row.get("canonical_sequence", "")
            if acc and seq:
                seqs[acc] = seq.replace(" ", "").upper()
    return seqs


def resolve_seq(acc: str, swiss: dict[str, str], meta: dict[str, str]) -> str:
    return swiss.get(acc) or meta.get(acc) or swiss.get(parent_acc(acc)) or meta.get(parent_acc(acc)) or ""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    swiss = load_swissprot(args.swissprot)
    meta = load_metadata(args.metadata)
    manual_rows = load_negatome(args.manual)
    stringent_keys = {pair_key(r["acc_a"], r["acc_b"]) for r in load_negatome(args.manual_stringent)}
    collapsed = collapse_undirected(manual_rows)

    scored_rows = []
    skipped = defaultdict(int)
    for row in collapsed:
        seq_a = resolve_seq(row["protein_a"], swiss, meta)
        seq_b = resolve_seq(row["protein_b"], swiss, meta)
        if not seq_a or not seq_b:
            skipped["missing_sequence"] += 1
            continue
        if seq_a == seq_b:
            skipped["identical_sequence"] += 1
            continue
        key = (row["protein_a"], row["protein_b"])
        scored_rows.append({
            **row,
            "in_manual_stringent": "1" if key in stringent_keys else "0",
            "len_a": str(len(seq_a)),
            "len_b": str(len(seq_b)),
            "query": seq_a,
            "text": seq_b,
        })

    pairs_csv = args.output_dir / "negatome_manual_pairs_v1.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "text"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"query": r["query"], "text": r["text"]} for r in scored_rows)

    table = args.output_dir / "negatome_manual_pairs_v1.tsv"
    fields = ["protein_a", "protein_b", "in_manual_stringent", "n_rows", "pmids",
              "methods", "raw_ids", "len_a", "len_b"]
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in scored_rows)

    summary = {
        "status": "negatome2_manual_pairs_for_plminteract",
        "source": "https://mips.helmholtz-muenchen.de/proj/ppi/negatome/manual.txt",
        "raw_manual_rows": len(manual_rows),
        "unique_undirected_pairs": len(collapsed),
        "scored_pairs": len(scored_rows),
        "stringent_among_scored": sum(r["in_manual_stringent"] == "1" for r in scored_rows),
        "skipped": dict(skipped),
        "swissprot_entries": len(swiss),
    }
    (args.output_dir / "negatome_manual_prep_summary_v1.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
