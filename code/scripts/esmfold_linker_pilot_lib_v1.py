"""Metrics for ESMFold linker-dimer vs AF3 partner ranking.

ESMFold is a single-chain model. A dimer is folded as seqA + Gly*N + seqB.
Inter-chain PAE/contacts are derived after dropping the linker residues.
This is not native AF3 ipTM; keep the two scores in separate columns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LINKER_AA = "G"
DEFAULT_LINKER_LEN = 25
CONTACT_CA_CUTOFF = 8.0
PAE_INTERFACE_CUTOFF = 12.0
ATOM14_CA = 1


def read_tsv(path: Path) -> list[dict[str, str]]:
    import csv
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_monomer_sequence(monomer_inputs: Path, accession: str) -> str:
    payload = json.loads((monomer_inputs / f"{accession}.json").read_text(encoding="utf-8"))
    return payload["sequences"][0]["protein"]["sequence"]


def fused_sequence(seq_a: str, seq_b: str, linker_len: int = DEFAULT_LINKER_LEN) -> tuple[str, int, int, int]:
    if linker_len < 1:
        raise ValueError("linker_len must be >= 1")
    linker = LINKER_AA * linker_len
    return seq_a + linker + seq_b, len(seq_a), linker_len, len(seq_b)


def chain_slices(n_a: int, n_linker: int, n_b: int) -> tuple[slice, slice, slice]:
    return slice(0, n_a), slice(n_a, n_a + n_linker), slice(n_a + n_linker, n_a + n_linker + n_b)


def as_2d_pae(pae: np.ndarray) -> np.ndarray:
    array = np.asarray(pae, dtype=np.float64)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"PAE must be square, got {array.shape}")
    return array


def as_1d_plddt(plddt: np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(plddt, dtype=np.float64)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[-1] in (37, 50):
        # categorical lDDT bins; expected value roughly 0-1 or 0-100.
        bins = np.linspace(0.0, 1.0, array.shape[-1], endpoint=False) + 0.5 / array.shape[-1]
        array = array @ bins
    if array.ndim != 1:
        array = np.squeeze(array)
    if array.shape[0] != length:
        raise ValueError(f"pLDDT length {array.shape} != {length}")
    if float(np.nanmax(array)) <= 1.5:
        array = array * 100.0
    return array


def as_ca_xyz(positions: np.ndarray) -> np.ndarray:
    array = np.asarray(positions, dtype=np.float64)
    # Typical shapes: (L, 14, 3), (recycles, L, 14, 3), (B, recycles, L, 14, 3)
    if array.ndim == 5:
        array = array[0, -1]
    elif array.ndim == 4:
        array = array[-1]
    elif array.ndim == 3:
        pass
    else:
        raise ValueError(f"unexpected positions shape {array.shape}")
    if array.shape[-1] != 3 or array.ndim != 3:
        raise ValueError(f"expected (L, atoms, 3), got {array.shape}")
    return array[:, ATOM14_CA, :]


def interchain_block(pae: np.ndarray, n_a: int, n_linker: int, n_b: int) -> np.ndarray:
    square = as_2d_pae(pae)
    expected = n_a + n_linker + n_b
    if square.shape[0] != expected:
        raise ValueError(f"PAE L={square.shape[0]} != fused length {expected}")
    a, _linker, b = chain_slices(n_a, n_linker, n_b)
    return np.concatenate([square[a, b].ravel(), square[b, a].ravel()])


def interchain_scores(
    pae: np.ndarray,
    plddt: np.ndarray,
    positions: np.ndarray,
    n_a: int,
    n_linker: int,
    n_b: int,
    contact_cutoff: float = CONTACT_CA_CUTOFF,
    pae_cutoff: float = PAE_INTERFACE_CUTOFF,
) -> dict[str, float]:
    fused_len = n_a + n_linker + n_b
    block = interchain_block(pae, n_a, n_linker, n_b)
    plddt_1d = as_1d_plddt(plddt, fused_len)
    ca = as_ca_xyz(positions)
    if ca.shape[0] != fused_len:
        raise ValueError(f"CA L={ca.shape[0]} != fused length {fused_len}")
    a, _linker, b = chain_slices(n_a, n_linker, n_b)
    ca_a, ca_b = ca[a], ca[b]
    delta = ca_a[:, None, :] - ca_b[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    contacts = dist <= contact_cutoff
    n_contacts = int(contacts.sum())
    if_a = contacts.any(axis=1)
    if_b = contacts.any(axis=0)
    if_plddt = np.concatenate([plddt_1d[a][if_a], plddt_1d[b][if_b]]) if n_contacts else np.array([], dtype=float)
    mean_inter_pae = float(np.mean(block))
    return {
        "n_a": float(n_a),
        "n_b": float(n_b),
        "n_linker": float(n_linker),
        "fused_len": float(fused_len),
        "mean_inter_pae": mean_inter_pae,
        "min_inter_pae": float(np.min(block)),
        "frac_pae_lt_12": float(np.mean(block < pae_cutoff)),
        "n_ca_contacts_8A": float(n_contacts),
        "mean_interface_plddt": float(np.mean(if_plddt)) if if_plddt.size else float("nan"),
        "mean_chain_plddt": float(np.mean(np.concatenate([plddt_1d[a], plddt_1d[b]]))),
        "score_neg_mean_inter_pae": -mean_inter_pae,
    }


def ranking_hit(score_d: float, score_c: float) -> float:
    if math.isnan(score_d) or math.isnan(score_c):
        return float("nan")
    if score_d > score_c:
        return 1.0
    if score_d < score_c:
        return 0.0
    return 0.5


def wilson_interval(hits: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    p = hits / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return p, centre - half, centre + half


def mean(values: list[float]) -> float:
    finite = [x for x in values if not math.isnan(x)]
    return sum(finite) / len(finite) if finite else float("nan")
