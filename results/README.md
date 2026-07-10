# Results

Canonical, single-source-of-truth storage for experiment data, per root `README.md` §7.

```text
results/
├── raw/<experiment-id>/         untouched output from experiment scripts
├── processed/<experiment-id>/   cleaned/aggregated data produced by analysis scripts
└── figures/<experiment-id>/     plots produced by analysis scripts
```

Raw data is never hand-edited. If a run looks wrong, re-run the experiment.
