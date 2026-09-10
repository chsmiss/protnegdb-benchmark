# Analysis code

This directory contains the notebooks and scripts for the ProtNegDB Benchmark
package. The repository root is the directory that contains both `code/` and
`data/`. Scripts resolve that root as two directories above `code/scripts/`.
Notebooks search parent directories for the same pair of folders.

Reference manuscript files: ProtNegDB_20260910.pdf and supplementary-0910.pdf.
Figure 2b/c and Supplementary S2a–c labels follow those PDFs. Final main
Figures 3–5 are assembled by `code/scripts/render_nature_selected_figures_20260907.py`.

`data/cohorts` contains the locked evaluation subsets. `data/interim` retains
sample metadata, score inputs, split assignments and intermediate statistics.
Repeated-PDB subsets, 30%/50% cluster definitions and three fine-tuning seeds
are distinct analyses, not superseded releases. The two 300-triplet cohorts
must not be merged. Frozen representation arrays are supplied to refit probes.

The four `layer_cls_v1.npz` archives are GitHub Release assets. After cloning:

```bash
bash code/scripts/download_representation_arrays.sh
```

The complete ProtNegDB negative resource is intentionally excluded. Model
checkpoints, full PDB/UniProt databases and GPU inference environments remain
external prerequisites. Recorded sequence-pair SHA-1 keys support exact
positive training-overlap checks without distributing the original large
training table.

Python scripts and notebook cells have been syntax checked and cohort counts
verified. Full inference and all notebooks have not been re-executed in this
package. Some upstream construction scripts retain external database paths;
use explicit CLI arguments for those steps. This is not a claim of a
one-command rebuild.

See the repository README for cohort tables, license terms and citation
metadata.
