The most critical upgrade in Week 12 is the attempt to model the global spread of influenza natively top-down via a **Spatio-Temporal Graph Neural Network (A3T-GCN)**.

### Representation & Topology
* **Graph Definition:** The network topology leverages airline mobility data. Nodes are countries, and edges are route connections weighted by `log1p(route_count)`.
* **State Tensor Construction:** Tabular data was projected into a dense 3D tensor sequence $(T, N, F)$ combining:
  * *Time-varying features*: climate lags, flu cases, mobility tracking.
  * *Static features*: PageRank, Centrality metrics.
  * *Sinusoidal Temporal logic*: Incorporating `.sin()` and `.cos()` for week-of-year representation.
* **Missing Data Channel:** To resolve disconnected nodes (countries missing active flu records for specific weeks), Week 12 deployed an **observed-mask channel**. Missing cells are filled with a literal `0` accompanied by an indicator mask, stabilizing gradient paths rather than interpolating or dropping spatial graph edges.

### Network Mechanics
* **Window Size:** $L=8$ (8 weeks of look-back observations evaluating dynamically to predict future step $t+2$).
* **Huber Loss:** Models are fitted using PyTorch Geometric Temporal mapping `HuberLoss(delta=1.0)` coupled with an Adam Optimizer (`1e-2` learning rate). 
* **Outcome Compatibility:** GNN arrays are successfully mapped back into Pandas DataFrames to compute standard $R^2$, MAE, RMSE, and F1 threshold metrics so it can natively rank against GBM and Ridge Regression.

## 4. Evaluation & Results
* **[model_dashboard.png](Module4_output/model_dashboard.png):**
  The 6-panel summary integrates the `gnn_a3tgcn` against `ridge_enhanced` and `gbm_enhanced`. All models share one temporal split (train < 2018 | val 2018–2019 | test ≥ 2020) and are scored in raw case-count units so the leaderboard is directly comparable.

### 4.1 Leaderboard

| Model | Val RMSE | MAE | Test RMSE | $R^2$ | F1 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **ridge_enhanced** | **301.6** | **57.8** | **552.4** | **0.802** | 0.751 |
| gbm_enhanced (XGBoost) | 324.1 | 65.7 | 734.9 | 0.650 | **0.817** |
| gnn_a3tgcn (A3T-GCN) | 561.4 | 128.5 | 1034.8 | 0.263 | 0.598 |
| gbm_baseline | 804.7 | 157.9 | 1198.9 | 0.014 | 0.432 |
| ridge_baseline | 808.5 | 164.9 | 1208.2 | −0.002 | 0.426 |

The headline result is counter-intuitive: the most sophisticated architecture (A3T-GCN) places **last among the enhanced models**, and a plain L2-regularized linear regression tops both the boosted trees and the graph network on every regression metric ($R^2$, MAE, RMSE). The two panels that explain the story are *$R^2$ (higher is better)* and *Predicted vs Actual*, read together with the *baseline → enhanced* jump that every algorithm shares.

### 4.2 Why the A3T-GCN Falls Short

The GNN is not broken — it converges (best val RMSE 561.4 at epoch 69) — but its inductive biases are mismatched to this dataset:

* **The sparsity tax.** ~42% of the dense $(T, N, F)$ grid is *absent* country-weeks. The observed-mask channel stabilizes gradients, but message passing still aggregates over neighbourhoods where many cells are literal-`0` placeholders. Spatial convolution therefore averages real signal with non-informative zeros. The tabular models pay none of this cost: they simply **drop** unobserved rows and train only on real records.
* **The static graph fights the test era.** Edges are fixed airline routes, and per-snapshot edge pruning was found to *hurt*, so the topology never changes. But the test period (2020+) is precisely when air mobility collapsed under COVID restrictions. The GNN keeps diffusing influenza along routes that were effectively shut — its core assumption (*flu spreads along airline edges*) is most wrong exactly where it is being scored.
* **The dominant signal is diluted, not amplified.** The single most predictive feature is the autoregressive term — recent observed flu (`flu_cases_total` at $t$) predicting $t+2$. Ridge and XGBoost consume it raw as a column. The GNN folds it into a tensor channel that is scaled, masked ($\times 0$ where unobserved), message-passed across the graph, then pushed through a GRU + temporal attention — every stage can attenuate a signal the tabular models exploit directly.
* **Capacity spent on structure the data doesn't reward.** With a near-linear dominant relationship and only ~58k training rows, the GNN's representational machinery models spatial diffusion that contributes little, while the noisy, jumpy validation curve (793 → 561 with large bounces) shows an optimization landscape far harder than a closed-form ridge solve. The result is the highest MAE (128.5) and RMSE (1034.8) of the enhanced models — i.e. it systematically misses the large outbreaks.

### 4.3 Why a Linear Model Tops Both XGBoost and the GNN

Ridge winning is not an accident of tuning; it is the correct tool for *this* problem:

* **The features were engineered to be linear.** The lift from `ridge_baseline` ($R^2 = -0.002$, climate only) to `ridge_enhanced` ($R^2 = 0.802$) is almost entirely the autoregressive flu lag plus mobility/risk/graph features. The lagged climate anomalies were themselves selected by **Pearson correlation** with the target. When the feature set is hand-built to relate near-linearly to the outcome, a linear model is the matched estimator and extracts that signal with zero excess variance.
* **Only the linear model can extrapolate through the regime shift.** The test set (2020+) drives flu counts far *below* the pre-2018 training range. Tree ensembles and the GNN can only emit values in or near the range they were trained on — they cannot extrapolate into the collapsed COVID regime — whereas Ridge extends its linear response and degrades gracefully. This is why Ridge's advantage is widest on **RMSE** (552 vs 735 vs 1035): RMSE punishes the large regime-shift misses, and only the linear model bends with them.
* **Bias–variance favours simplicity here.** Modest data + near-linear signal means the low-variance linear model generalizes best; the added capacity of boosting and especially the GNN overfits pre-COVID structure and transfers worse across the 2020 break. Ridge's L2 shrinkage also absorbs the collinear weather block (temperature/dew point/humidity) without coefficient blow-up, giving coefficients stable enough to survive the regime change.

### 4.4 Caveats — the Leaderboard Is Not One-Dimensional

Ridge does **not** dominate outright. XGBoost wins **outbreak detection F1 (0.817 vs 0.751)** — when the task is reframed from "predict the count" to "flag the threshold crossing," the trees' decision-boundary flexibility pays off (visible in the *F1* and *Precision & Recall* panels). And the GNN's contribution is methodological rather than metric: it demonstrates a defensible spatio-temporal formulation that would be expected to overtake the tabular models given denser per-country reporting and a *dynamic* mobility graph that tracks the 2020 collapse. The honest reading of the dashboard is that **regression accuracy under regime shift rewards the simplest model, while outbreak classification rewards the trees** — and the choice of deployed model (`ridge_enhanced`, by validation RMSE) reflects the project's primary metric, not a universal verdict.
