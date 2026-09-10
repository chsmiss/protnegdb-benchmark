# ProtNegDB Benchmark

Frozen evaluation cohorts, model scores, figure source data and analysis code for the ProtNegDB manuscript:

**Assembly context confounds direct-partner prediction by sequence-based interaction models**

This repository is the analysis package associated with the 10 September 2026 manuscript revision. It is intended for inspection of the reported test sets, score tables and figure-generating code. It is not a redistribution of the complete ProtNegDB resource.

The complete evidence-resolved negative interaction resource is served separately at [http://protnegdb.bjmu.edu.cn/](http://protnegdb.bjmu.edu.cn/). Model checkpoints, full PDB/UniProt databases and GPU inference environments are external prerequisites and are not included here.

## Citation

Please cite the ProtNegDB manuscript and this versioned code/data package. Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Until a journal DOI is assigned, cite the GitHub repository and release tag.

Cheng, H., Sun, C., Ye, C. & Zhao, D. ProtNegDB Benchmark. GitHub. https://github.com/chsmiss/protnegdb-benchmark

Correspondence: Dongyu Zhao, Peking University.

## Scope and limitations

This package contains:

- explicit evaluation subsets used in the manuscript figures;
- per-model scores, split assignments, training inputs and summary statistics for those subsets;
- notebooks and Python scripts used to produce the reported panels;
- frozen layer representations required to refit the Figure 3 probes;
- SHA-1 keys of positive training sequence pairs for exact overlap checks.

This package does **not** contain:

- the full literature-negative resource (11,071 records);
- the full structural noncontact resource (558,112 records);
- original model weights, embeddings computed from those weights, or GPU runtimes;
- a claim of one-command reconstruction of every upstream database query.

Legacy internal names such as Fig. 5/6 or “family transfer” are retained because the analysis scripts still call those paths. They do not indicate that superseded figure versions were submitted. Figure 2b/c and Supplementary Figures S2a–c follow the 10 September 2026 PDF numbering.

Python scripts and notebook cells were syntax-checked, and explicit cohort counts were verified. Full model inference and all notebooks were **not** re-executed in this package. Some upstream construction scripts still accept paths to local databases; those steps require explicit command-line arguments.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── requirements.txt
├── SHA256SUMS
├── code/
│   ├── README.md
│   ├── notebooks/     # figure and supplementary panel notebooks
│   └── scripts/       # analysis, scoring and rendering scripts
└── data/
    ├── cohorts/       # locked evaluation subsets
    ├── interim/       # scores, splits, manifests and probe inputs
    ├── reported_stats/
    ├── source_data/   # figure source tables
    ├── structures/    # three example PDB biological assemblies
    └── training_reference/  # SHA-1 keys of positive training pairs
```

Analysis scripts resolve the repository root as two directories above `code/scripts/` (`Path(__file__).resolve().parents[2]`). Notebooks search parent directories for a folder that contains both `code/` and `data/`. Run notebooks from the repository root or from `code/notebooks`.

## Evaluation cohorts

Locked tables live in `data/cohorts/`. Row counts below exclude the header.

| File | *n* | Role in the manuscript |
|---|---:|---|
| `structural_core_3365_triplets.tsv` | 3,365 | Full structural ranking set (Fig. 2b,e; S2) |
| `structural_core_3051_noncontacts.tsv` | 3,051 | Core A–C noncontacts (Fig. 2c; S1–S2) |
| `structural_repeated_ranking.tsv` | 2,329 | Repeated-evidence ranking (Fig. 2e; S2a) |
| `structural_repeated_1894_noncontacts.tsv` | 1,894 | Repeated-PDB noncontacts (S1f; S2b,c) |
| `clean_structural_2480.tsv` | 2,480 | Clean structural noncontacts (Fig. 4–5; S8) |
| `multimodel_300_256anchors.tsv` | 300 | Shared seven-model test (Fig. 3c,d; S6) |
| `same_anchor_300_266anchors.tsv` | 300 | Same-anchor random control (Fig. 2d) |
| `split50_train_eligible.tsv` | 1,865 | 50% identity train-eligible split (S4–S5) |
| `split50_family_unseen.tsv` | 540 | Cluster-unseen evaluation (S4–S5) |
| `split50_protein_unseen_family_seen.tsv` | 101 | Protein-unseen, family-seen split (S4–S5) |
| `split50_train_ineligible_unpermuted.tsv` | 259 | Train-ineligible unpermuted split (S4–S5) |
| `split50_dropped_mixed_family.tsv` | 277 | Dropped mixed-family records (S4–S5) |
| `split50_dropped_mixed_protein.tsv` | 273 | Dropped mixed-protein records (S4–S5) |
| `split50_dropped_pdb_overlap.tsv` | 50 | Dropped PDB-overlap records (S4–S5) |
| `negatome_reconstituted_direct.tsv` | 298 | Reconstituted direct negatives (Fig. 4–5; S7) |
| `negatome_cell_binary_proximity.tsv` | 145 | Cellular binary/proximity negatives (Fig. 4–5; S7) |
| `negatome_co_complex_association.tsv` | 119 | Co-complex association negatives (Fig. 4–5; S7) |
| `negatome_other_or_unresolved.tsv` | 22 | Other or unresolved Negatome labels (Fig. 4–5; S7) |
| `protnegdb_direct_104.tsv` | 104 | Independent direct experimental negatives (Fig. 5; S9) |
| `supportive_ranking_192.tsv` | 192 | Direct-positive partner pairing (Fig. 5d; S9c–e) |
| `verified_same_assembly_555.tsv` | 555 | Same-assembly contact verification (S2f) |

The two 300-triplet tables are independent cohorts and must not be merged. Repeated-PDB subsets, 30% versus 50% identity cluster definitions, and the three fine-tuning seeds are distinct analyses, not superseded releases.

Triplet notation follows the manuscript: **A** is the shared anchor, **D** is a direct assembly contact of A, and **C** is a coordinate-QC noncontact sampled in the same biological assembly.

## Evidence semantics

ProtNegDB does not treat every negative label as the same biological claim.

- **Experimental direct non-binding** is conditional on constructs, assay and conditions.
- **Cellular binary/proximity negative evidence** is conditional on the cellular measurement configuration.
- **Co-complex association negative evidence** concerns detection in a complex-level assay.
- **Assembly noncontact** means that no resolved direct interface was observed in a specified PDB biological assembly.
- **Random unlabeled pairing** is a reference control, not demonstrated biological non-interaction.

## Code

### Notebooks

| Notebook | Content |
|---|---|
| `code/notebooks/01_Figure2_sequence_partner_selection.ipynb` | Figure 2 panels |
| `code/notebooks/Figure4ab_colors_20260908.ipynb` | Figure 4a,b colour and panel export |
| `code/notebooks/S1.ipynb` … `S9.ipynb` | Supplementary Figures S1–S9 |

Final main Figures 3–5 are assembled by `code/scripts/render_nature_selected_figures_20260907.py`. Notebooks contain separate panel cells. Some notebooks still write figures to an original working-directory path; change the output directory before re-running.

### Scripts

`code/scripts/` contains the analysis, scoring, split-construction and rendering programs used for the reported results. Upstream reconstruction scripts that query local PDB, UniProt or model checkpoints remain in the package for provenance, but they are not a one-command rebuild. Prefer the frozen tables in `data/` unless those external resources are available.

Operational notes are in [`code/README.md`](code/README.md).

### Software environment

The figure notebooks were executed in a Python environment with NumPy, pandas, Matplotlib, seaborn and SciPy. A minimal requirement file is provided as [`requirements.txt`](requirements.txt). Notebooks specify Arial; if that font is unavailable, Matplotlib will substitute a default sans-serif font and panel metrics may differ slightly.

## Frozen representation arrays

Four NumPy archives used to refit the Figure 3 / S3 layer-wise probes exceed GitHub’s 100 MB blob limit and are distributed as GitHub Release assets rather than Git objects:

| Release asset | Local path | Approximate size |
|---|---|---:|
| `mint_esm2_layer_cls_v1.npz` | `data/interim/mint_layerwise_probes_v1/esm2/layer_cls_v1.npz` | 1.1 GB |
| `mint_mint_layer_cls_v1.npz` | `data/interim/mint_layerwise_probes_v1/mint/layer_cls_v1.npz` | 1.1 GB |
| `plminteract_esm2_layer_cls_v1.npz` | `data/interim/plminteract_layerwise_probes_v1/esm2/layer_cls_v1.npz` | 0.56 GB |
| `plminteract_plminteract_layer_cls_v1.npz` | `data/interim/plminteract_layerwise_probes_v1/plminteract/layer_cls_v1.npz` | 0.56 GB |

After cloning the repository:

```bash
bash code/scripts/download_representation_arrays.sh
```

Checksums for all packaged files, including these arrays, are listed in [`SHA256SUMS`](SHA256SUMS). Classifier and MLP head parameters (`*.npy`) remain in Git because they are smaller.

## Training-overlap reference

`data/training_reference/string_positive_sequence_pairs.sha1.tsv` contains SHA-1 keys of positive training sequence pairs. The table supports exact overlap checks against the reported negatives without distributing the original large training matrix.

## Third-party resources

Redistributed fields may inherit terms from the source resources used to construct ProtNegDB, including PDB biological assemblies, UniProt accessions, Negatome, IntAct, STRING, and the original model publications (PLM-Interact, MINT, D-SCRIPT, Topsy-Turvy, SPRINT, ESMFold and AlphaFold 3). Obtain model weights and licensed databases from the original providers.

## License

Repository code is released under the MIT License (see [`LICENSE`](LICENSE)). Database records and source-derived fields may be governed by upstream licenses. Attribution to those resources is required when the tables are reused.

## Contact

Questions about the manuscript or this package: Dongyu Zhao, Peking University.

Website: [http://protnegdb.bjmu.edu.cn/](http://protnegdb.bjmu.edu.cn/)
