# ProtNegDB Benchmark

Evaluation cohorts, model scores, figure source data and analysis code for ProtNegDB.

The complete negative-interaction resource is available at [http://protnegdb.bjmu.edu.cn/](http://protnegdb.bjmu.edu.cn/). Model checkpoints, source databases and GPU runtimes are not included.

## Contents

```text
code/notebooks/   Figure 2, Figure 4a/b and Supplementary S1–S9
code/scripts/     analysis and rendering programs
data/cohorts/     locked evaluation subsets
data/interim/     scores, splits and intermediate tables
data/source_data/ figure source tables
```

Scripts treat the repository root as two directories above `code/scripts/`. Notebooks search parent directories for both `code/` and `data/`. Run notebooks from the repository root or from `code/notebooks`.

The two 300-triplet files (`multimodel_300_256anchors.tsv` and `same_anchor_300_266anchors.tsv`) are independent cohorts and must not be merged.

Triplet notation: **A** is the shared anchor, **D** is a direct assembly contact of A, and **C** is a coordinate-QC noncontact in the same biological assembly.

## Evaluation cohorts

Row counts exclude the header.

| File | *n* | Use |
|---|---:|---|
| `structural_core_3365_triplets.tsv` | 3,365 | Fig. 2b,e; S2 |
| `structural_core_3051_noncontacts.tsv` | 3,051 | Fig. 2c; S1–S2 |
| `structural_repeated_ranking.tsv` | 2,329 | Fig. 2e; S2a |
| `structural_repeated_1894_noncontacts.tsv` | 1,894 | S1f; S2b,c |
| `clean_structural_2480.tsv` | 2,480 | Fig. 4–5; S8 |
| `multimodel_300_256anchors.tsv` | 300 | Fig. 3c,d; S6 |
| `same_anchor_300_266anchors.tsv` | 300 | Fig. 2d |
| `split50_train_eligible.tsv` | 1,865 | S4–S5 |
| `split50_family_unseen.tsv` | 540 | S4–S5 |
| `split50_protein_unseen_family_seen.tsv` | 101 | S4–S5 |
| `split50_train_ineligible_unpermuted.tsv` | 259 | S4–S5 |
| `split50_dropped_mixed_family.tsv` | 277 | S4–S5 |
| `split50_dropped_mixed_protein.tsv` | 273 | S4–S5 |
| `split50_dropped_pdb_overlap.tsv` | 50 | S4–S5 |
| `negatome_reconstituted_direct.tsv` | 298 | Fig. 4–5; S7 |
| `negatome_cell_binary_proximity.tsv` | 145 | Fig. 4–5; S7 |
| `negatome_co_complex_association.tsv` | 119 | Fig. 4–5; S7 |
| `negatome_other_or_unresolved.tsv` | 22 | Fig. 4–5; S7 |
| `protnegdb_direct_104.tsv` | 104 | Fig. 5; S9 |
| `supportive_ranking_192.tsv` | 192 | Fig. 5d; S9c–e |
| `verified_same_assembly_555.tsv` | 555 | S2f |

Negative labels are not interchangeable: experimental non-binding, cellular/proximity evidence, co-complex evidence, assembly noncontact and random pairing are distinct claims.

Internal path names such as Fig. 5/6 or “family transfer” are retained by the analysis scripts and do not indicate superseded figure versions.

## Code

| Path | Content |
|---|---|
| `code/notebooks/01_Figure2_sequence_partner_selection.ipynb` | Figure 2 |
| `code/notebooks/Figure4ab_colors_20260908.ipynb` | Figure 4a,b |
| `code/notebooks/S1.ipynb`–`S9.ipynb` | Supplementary S1–S9 |
| `code/scripts/render_nature_selected_figures_20260907.py` | Figures 3–5 |

Dependencies: [`requirements.txt`](requirements.txt). File checksums: [`SHA256SUMS`](SHA256SUMS). Additional notes: [`code/README.md`](code/README.md).

## Citation

Cheng, H., Sun, C., Ye, C. & Zhao, D. ProtNegDB Benchmark. https://github.com/chsmiss/protnegdb-benchmark

See [`CITATION.cff`](CITATION.cff). Correspondence: Dongyu Zhao, Peking University.

## License

Code is released under the MIT License. Redistributed fields may inherit upstream terms from PDB, UniProt, Negatome, IntAct, STRING and the original model publications.
