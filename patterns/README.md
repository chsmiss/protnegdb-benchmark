# ProtNegDB benchmark for the Patterns manuscript

This package corresponds to the 24 November 2026 manuscript revision. It replays the evaluation from frozen cohort membership, labels, model scores and splits. Python 3.8–3.11 and a CPU are sufficient.

```bash
python -m pip install -r patterns/requirements.txt
python patterns/benchmark.py --output /tmp/protnegdb-patterns-results
```

| Output | Evaluation |
|---|---|
| `structural.tsv` | Anchor-weighted D-over-C ranking and held-out contact AUROC and average precision (Figure 2 and Figure S2) |
| `matched_reference.tsv` | D-over-C, D-over-R and C-over-R win rates (Figure 2B and Figure S2D) |
| `assay_discrimination.tsv` | 14 assay categories, three coverage scopes and the same-assay conflict sensitivity (Figure 3 and Figure S3) |
| `assay_ranking.tsv` | Same-anchor win rates for comparisons whose two scores share an inference protocol or archive source (Figure 3D) |
| `assay_operating_points.tsv` | Assay-specific five-fold FPR-cap and maximum-F1 rules, evaluated within protocol (Figure 4 and Figure S4) |
| `dip_dpr.tsv` | Top-two nomination yields, average precision and the expected random yield for the three Figure 6 predictors |

`common11` requires a score from every predictor except PPLM-Contact. `common12` requires all 12. `available` uses each predictor's available scores. Ranking ties count as half a win, with equal weight per anchor. The random reference R has no structural contact label. The literature folds keep observations linked by protein pair or publication together.

The `data/` directory also includes the 12-predictor specifications and the resource and assay counts from Tables S1–S2. Local integer IDs preserve the repeated pairs, anchors, components and protocol strata required for the analysis. Structural evidence comes from PDB. Experimental outcomes come from ProtNegDB and Negatome 2.0 Manual, with assay-matched IntAct positives. The DIP–Dpr labels and flagged measurements come from [Cosmanescu et al. (2018)](https://pubmed.ncbi.nlm.nih.gov/30467080/). The complete negative-interaction resource is available from [ProtNegDB](http://protnegdb.bjmu.edu.cn/).

This entry point reproduces point estimates from the saved scores. The manuscript reports the bootstrap intervals, pooled-threshold sensitivity, interface-score comparisons and hardware profiles. Model checkpoints and raw input sequences are distributed by their original providers under their respective terms.
