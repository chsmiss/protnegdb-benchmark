# Changelog

## 1.0.1 — 2026-09-10

- Removed `data/reported_stats`, `data/training_reference` and
  `data/structures` from the public package. These files are not required
  to inspect the reported evaluation cohorts or figure source tables.
- Excluded the four frozen `layer_cls_v1.npz` representation archives.
  Probe refitting is outside the scope of the public reader package.

## 1.0.0 — 2026-09-10

- Released the frozen ProtNegDB benchmark package corresponding to the
  10 September 2026 manuscript revision.
- Included locked evaluation cohorts, per-model scores, figure source data,
  notebooks and analysis scripts.
- Distributed frozen layer-representation arrays as GitHub Release assets
  because they exceed GitHub's 100 MB Git blob limit.
