# Case 1 Solution Report: Model Design, Theory, and Reproducibility

Date: 2026-03-21  
Project: CompData, Case 1

## 1. Executive Summary

This report explains how Case 1 was solved end-to-end, with emphasis on modeling choices, theoretical motivation, and reproducibility.

The pipeline compares three intentionally different regression families under the same nested cross-validation protocol:
- Elastic Net family (ridge, lasso, elastic net switching)
- Latent-factor family (PCR and PLS switching)
- Tree-ensemble family (random forest and extra trees switching)

Final ranking (lower RMSE is better):

| Rank | Approach | Mean Outer RMSE | Std Outer RMSE |
|---|---:|---:|---:|
| 1 | elastic_net | 35.810060338909004 | 7.032797817317284 |
| 2 | pls_pcr | 37.26275520286941 | 9.600035052667437 |
| 3 | tree_ensembles | 48.4498737035524 | 12.534620138815436 |

Winner: elastic_net  
Submission estimate policy: $\text{RMSE}_{est}=\mu_{outer}+0.2\cdot\sigma_{outer}$  
Final estimate: 37.21661990237246

The strongest practical finding is that regularized linear structure generalized better than the nonlinear tree family on this specific case, while PCR/PLS was competitive but slightly less stable than the winning elastic-net family.

## 2. What Case 1 Requires and How We Framed It

Case 1 is a supervised regression task with one labeled training file and one unlabeled prediction file.

Observed data schema in this implementation:
- Training file: 100 rows, 100 features, 1 target (`y`)
- New/prediction file: 1000 rows, same 100 features, no target
- Feature types:
  - 95 numeric predictors with prefix `x_`
  - 5 categorical/context predictors with prefix `C_`
- Missing values detected in current files: none

From the assignment workflow embodied by the codebase, the practical deliverables are:
- A model-comparison leaderboard across candidate approaches
- A winner model selection based on outer-CV RMSE
- A final prediction file for unlabeled data
- A self-estimated RMSE file for submission

This framing matters because the objective is not only to minimize training error; it is to produce the most defensible estimate of unseen-data performance while preserving a reproducible procedure.

## 3. Core Methodology: Why Nested CV and Why the 1-SE Rule

### 3.1 Why nested cross-validation is central here

If we tune hyperparameters and report that same validation score as final performance, the estimate becomes optimistic. Nested CV separates these concerns:
- Inner CV chooses hyperparameters.
- Outer CV evaluates generalization using held-out folds never used in tuning.

In this project:
- Outer folds: 10
- Inner folds: 5
- Seed: 42
- Same protocol across all model families for fairness.

This yields a distribution of outer-fold RMSE values, not just one number. That distribution gives two critical quantities:
- Mean outer RMSE (average expected error)
- Std outer RMSE (stability and uncertainty proxy)

### 3.2 Why the 1-SE rule was used

Hyperparameter tuning often picks the single best mean CV score, but that candidate may be overly complex and less robust. The 1-SE rule selects the simplest candidate whose CV score is within one standard error of the best.

Operationally, if best inner score is $s^*$ with standard deviation $\sigma^*$ across inner folds, then:
$$
\text{threshold}=s^* - \frac{\sigma^*}{\sqrt{K}}
$$
where $K$ is the number of inner folds.

All candidates above threshold are considered statistically similar; complexity-based ordering then chooses the simpler candidate. This injects stability bias into model selection, which is desirable in small-to-moderate data regimes.

### 3.3 Why the final RMSE estimate is mean + margin*std

The submission estimate is computed as:
$$
\widehat{\text{RMSE}} = \mu_{outer} + m\cdot\sigma_{outer},\quad m=0.2
$$
This is a conservative point estimate: not the most optimistic mean, and not a worst-case bound either. It encodes uncertainty directly into a single submitted number.

## 4. Why These Three Model Families Were Chosen

The key design principle was to test complementary inductive biases under identical evaluation conditions.

### 4.1 Elastic Net family: regularized linear bias

This family addresses two common properties in tabular predictive tasks:
- Collinearity among numeric predictors
- Need for shrinkage to reduce variance

The switcher evaluates ridge, lasso, and elastic net within one consistent pipeline. Regularization strength ($\alpha$) is searched on a log grid, and penalty type controls sparsity-vs-stability behavior.

Why this family is essential in Case 1:
- Numeric features are dominant (95 of 100).
- With moderate train size, regularization often stabilizes coefficients and improves generalization.
- It provides a transparent baseline that is often hard to beat when signal is mostly additive/near-linear.

### 4.2 PLS/PCR family: latent-space bias

PCR and PLS explicitly compress the feature space into lower-dimensional latent representations.

- PCR: projects $X$ to principal directions that maximize predictor variance, then regresses target on those components.
- PLS: chooses components that maximize covariance with target, often stronger when response-aligned structure is present.

Why this family is essential in Case 1:
- It tests whether predictive structure lives in a lower-dimensional manifold.
- It offers a different solution to collinearity than coefficient shrinkage.
- It can improve robustness when raw predictors are noisy but latent factors are stable.

### 4.3 Tree-ensemble family: nonlinear interaction bias

Random forest and extra trees model nonlinearities and interactions without explicit manual feature engineering.

Why this family is essential in Case 1:
- It tests whether strong non-additive interactions exist.
- It does not rely on global linear assumptions.
- It is less sensitive to scaling and can discover local structure.

Including this family prevents over-committing to linear assumptions and broadens the hypothesis space.

## 5. End-to-End Pipeline Design

Each family follows the exact same lifecycle:
1. Load data
2. Build preprocessing + estimator pipeline
3. Run nested CV and write artifacts
4. Refit chosen configuration on full training data
5. Predict the unlabeled set and write predictions
6. Aggregate all family summaries into final leaderboard and winner

Uniform artifacts per approach:
- `metrics_outer.csv`
- `tuning_trace.csv`
- `summary.json`
- `run_metadata.json`
- `refit/chosen_config.json`
- `refit/predictions.csv`

Global artifacts:
- `leaderboard.csv`
- `winner.json`
- `sample_predictions_<student>.csv`
- `sample_estimatedRMSE_<student>.csv`

This artifact contract is a major reproducibility advantage: every approach is forced to expose comparable diagnostics.

## 6. How Each Model Tackled Case 1 in Practice

### 6.1 Elastic Net family behavior (winner)

Selected final config:
- model: ridge
- alpha: 5.1794746792312125
- l1_ratio: 0.1

Interpretation:
- The winner landed close to a ridge regime, implying the task benefited more from coefficient shrinkage than feature sparsity.
- That is consistent with correlated numeric predictors where many weak-to-moderate effects combine additively.
- Low std outer RMSE (7.03) indicates stronger fold-to-fold stability than the alternatives.

### 6.2 PLS/PCR behavior (second place)

Selected final config:
- model: pcr
- n_components: 60
- alpha: 4.328761281083057

Interpretation:
- The method kept a fairly large latent dimension, suggesting useful information remained distributed across many components.
- Performance was close to elastic net, which supports the idea that latent linear structure exists.
- Higher std than elastic net indicates slightly less stable generalization under fold perturbations.

### 6.3 Tree-ensemble behavior (third place)

Selected final config:
- model: rf
- n_estimators: 200
- max_depth: 8
- max_features: 0.8
- min_samples_leaf: 5

Interpretation:
- The selected depth and leaf settings indicate the 1-SE rule favored a controlled-complexity forest rather than very deep trees.
- Despite this, outer RMSE stayed substantially higher with larger variance.
- A likely explanation is that the available sample size and signal geometry favored smoother linear/latent models over highly partitioned nonlinear structure.

## 7. Comparative Result Interpretation

Winner selection rule:
1. Lowest mean outer RMSE
2. If tied, lowest std outer RMSE

Observed outcome:
- Elastic net was best on both criteria.
- PLS/PCR was competitive but not better on mean or stability.
- Tree ensembles underperformed on both mean and variance.

Estimate check:
- $35.810060338909004 + 0.2\cdot7.032797817317284 = 37.21661990237246$
- Exactly matches output estimate file.

The key scientific takeaway is not only that elastic net won, but why: it best matched the data regime implied by this case under strict out-of-sample evaluation.

## 8. Reproducibility Protocol (Step-by-Step)

### 8.1 Local reproducible run

From repository root:

```bash
python -m pip install -e .
export PYTHONPATH=Case1/src
python -m case1_comp.orchestrator.run_all --student-no 123456
```

### 8.2 HPC reproducible run

```bash
export COMPDATA_ROOT=/path/to/CompData
export STUDENT_NO=123456
export RMSE_MARGIN=0.2
bash Case1/hpc/submit_all.sh
```

### 8.3 What to verify after execution

1. Per-family nested summaries exist and contain finite RMSE values.
2. `Case1/results/leaderboard.csv` has three rows and correct ordering by mean/std RMSE.
3. `Case1/results/winner.json` agrees with leaderboard top row.
4. `Case1/results/sample_predictions_<student>.csv` exists and has 1000 predictions.
5. `Case1/results/sample_estimatedRMSE_<student>.csv` matches the formula with configured margin.

### 8.4 Reproducibility details that protect consistency

- Fixed seed and fixed fold counts across all approaches.
- Shared preprocessing policy:
  - Numeric: median imputation (+ scaling for linear/latent models)
  - Categorical: most-frequent imputation + one-hot encoding
- Same orchestration and artifact schema for every family.

## 9. What Was Learned During Implementation

The project also surfaced important engineering lessons that directly affect scientific validity:
- Model wrappers must expose proper fitted-state behavior in sklearn pipelines; otherwise CV scoring can silently collapse into non-finite model-selection inputs.
- Aggregation must fail loudly when expected artifacts are missing.
- HPC scripts should propagate explicit root paths and validate environment assumptions before fan-out submissions.

These changes matter because reproducibility is part of the solution quality, not a separate concern.

## 10. Final Decision and Practical Next Steps

Final decision:
- Submit predictions from the elastic_net winner.
- Submit RMSE estimate 37.21661990237246 from the standardized margin policy.

If a next modeling cycle is allowed, the most useful extensions would be:
1. Add gradient-boosted trees as a fourth family to test nonlinear smooth functions beyond bagged trees.
2. Add fold-wise residual diagnostics to test whether remaining error is heteroscedastic or subgroup-specific.
3. Evaluate simple stacking (elastic-net + latent model blend) while keeping nested evaluation to avoid leakage.

## Appendix: Final Metrics Snapshot

- elastic_net: mean=35.810060338909004, std=7.032797817317284
- pls_pcr: mean=37.26275520286941, std=9.600035052667437
- tree_ensembles: mean=48.4498737035524, std=12.534620138815436
- winner estimate: 37.21661990237246
