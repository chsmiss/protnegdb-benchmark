# Analysis code

The repository root contains both `code/` and `data/`. Scripts resolve that
root as two directories above `code/scripts/`. Notebooks search parent
directories for the same pair of folders.

`data/cohorts` holds the locked evaluation subsets. The two 300-triplet
cohorts must not be merged. Main Figures 3–5 are assembled by
`render_nature_selected_figures_20260907.py`.
