# Project Proposal
## Mining Behavioral Risk Patterns for Diabetes Prediction: An Explainable Ensemble Approach

**CLC01 — Group 9**

---

## 1. Project Title & Clarification of Key Terms

**Title:** Mining Behavioral Risk Patterns for Diabetes Prediction: An Explainable Ensemble Approach

### What does "Mining" mean here?

"Mining" refers specifically to **Association Rule Mining (ARM)** using the FP-Growth algorithm — a formal data mining technique that discovers co-occurrence patterns among behavioral features. This is not a metaphor for feature analysis. The term is technically grounded:

- ARM produces explicit if-then rules of the form: `{HighBP=1, PhysActivity=0, BMI_Obese=1} → At_Risk` with measurable support, confidence, and lift.
- These rules represent **association patterns between behavioral features** — the first interpretation of "patterns" in this project.

### What does "Patterns" mean — two interpretations, one pipeline

This project deliberately uses both interpretations and cross-validates them:

| Interpretation | Method | Output |
|---|---|---|
| **Association patterns** between behavioral features | FP-Growth ARM | If-then rules with support/confidence/lift |
| **Feature importance clusters** from predictive model | SHAP on XGBoost | Global ranking + local contribution plots |

The novel contribution is the **ARM–SHAP consistency check**: after obtaining both outputs independently, we compute Jaccard similarity between the top-15 ARM features and top-10 SHAP features. Agreement validates both methods; disagreement is itself an informative finding about the limits of each approach (ARM captures joint co-occurrence; SHAP captures marginal conditional contribution).

These two interpretations require different design and presentation:
- ARM output → formatted rule table, clinically interpreted, filtered by lift ≥ 1.5
- SHAP output → summary plot, beeswarm plot, dependence plots for non-linear features (BMI, Age)

---

## 2. Problem Statement

### 2.1 Background

Type 2 diabetes affects approximately **11.1% of adults aged 20–79 globally** (IDF Diabetes Atlas 2025), with over 40% undiagnosed. More than 90% of Type 2 cases are linked to modifiable lifestyle behaviors, making early behavioral intervention the most viable prevention strategy.

Most predictive models operate as black boxes — producing probability scores without explaining *which behavioral combinations* drive risk. A clinician cannot act on a score alone; they need interpretable, actionable evidence about specific behavioral thresholds.

### 2.2 Research Questions

- **RQ1:** Which combinations of lifestyle and biological indicators (via FP-Growth ARM) form the most critical behavioral "tipping points" for diabetes risk?
- **RQ2:** To what extent can a tuned Ensemble model (XGBoost + LightGBM + Random Forest) accurately predict diabetes risk, and how well do SHAP explanations align with the high-lift rules from RQ1?
- **RQ3:** In the younger demographic (ages 18–44), which behavioral features drive early-onset risk, and how do these differ from the general population in both ARM rules and SHAP importance?

---

## 3. Dataset

### 3.1 Dataset Selection — Explicit Justification

**Selected dataset:** [Diabetes Health Indicators Dataset — BRFSS 2015](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (`data.csv`)

We explicitly chose BRFSS over the Pima Indians Diabetes dataset for the following reasons:

| Criterion | Pima Indians Dataset | BRFSS 2015 (selected) |
|---|---|---|
| Sample size | 768 rows | **269,131 rows** |
| Features | 8 clinical features | **21 behavioral + biological features** |
| ARM feasibility | Too small for meaningful support thresholds | Sufficient for min_support = 0.05 (~13,450 records) |
| Behavioral features | Absent (clinical only) | Rich: PhysActivity, Fruits, Veggies, Smoker, HvyAlcoholConsump, etc. |
| Population scope | Single ethnic group, women only | U.S. national survey, all adults |
| Research alignment | Cannot answer RQ1 (no behavioral co-occurrence patterns) | Directly supports all three RQs |

The BRFSS dataset is the only publicly available dataset that simultaneously supports Association Rule Mining on behavioral features AND ensemble classification at scale.

### 3.2 Dataset Overview

| Attribute | Details |
|---|---|
| Source | CDC Behavioral Risk Factor Surveillance System (BRFSS), 2015 |
| Records | 269,131 rows |
| Features | 22 columns: 1 target + 21 behavioral/biological indicators |
| Feature types | 14 binary (0/1) + 7 ordinal/continuous |
| Missing values | 0 |
| Duplicate rows | 0 |

### 3.3 Critical Issue: Three-Class Target — Binarization Decision

The raw dataset contains **three target classes**, not two. This must be addressed explicitly:

| Original Class | Clinical Meaning | Count | Proportion | After Binarization |
|---|---|---|---|---|
| 0 | No diabetes / No pre-diabetes | 194,377 | 72.22% | Class 0 — No Risk |
| 1 | Pre-diabetes | 39,657 | 14.74% | Class 1 — At Risk |
| 2 | Diabetes | 35,097 | 13.04% | Class 1 — At Risk |

**Justification:** In the context of behavioral risk prevention, both pre-diabetes and diabetes require the same immediate behavioral intervention (dietary change, physical activity, weight management). Merging them into a single "At Risk" class reflects clinical reality, not analytical convenience.

### 3.4 Class Imbalance — Mandatory Concern

After binarization: **72.22% No Risk vs. 27.78% At Risk** — a pronounced imbalance that directly affects model evaluation and training.

**Why this matters:**
- A naive classifier predicting "No Risk" for all records achieves 72.22% accuracy — making accuracy a misleading metric.
- Standard training without correction will bias the model toward the majority class, causing high False Negatives (missed at-risk patients) — the highest-cost error in a clinical context.

**Handling strategy — three experimental conditions evaluated inside CV folds:**

| Strategy | Method | Risk |
|---|---|---|
| SMOTE oversampling | Synthetic minority samples generated in training folds only | Overfitting to synthetic samples if applied before split |
| NearMiss undersampling | Majority class reduced to match minority | Information loss from discarding real records |
| class_weight='balanced' | Loss function reweighted — no data modification | Safest for large datasets; no leakage risk |

**Critical constraint:** SMOTE/NearMiss applied **only within each training fold** — never on validation or test folds. Applying before splitting causes data leakage and inflates reported metrics.

**Primary evaluation metric:** Macro F1-score and Recall (sensitivity) — not accuracy. Minimizing False Negatives is the clinical priority.

---

## 4. Justification of Each Pipeline Component

### 4.1 Why Association Rule Mining (FP-Growth)?

ARM is not included for completeness — it is the primary method for answering RQ1.

- **FP-Growth over Apriori:** On 269,131 records, Apriori's candidate generation is computationally prohibitive. FP-Growth uses a compressed tree structure, reducing memory and runtime by an order of magnitude.
- **ARM over correlation analysis:** Correlation captures pairwise linear relationships between individual features and the target. ARM captures **multi-feature co-occurrence patterns** — e.g., `{HighBP=1, PhysActivity=0, BMI_Obese=1} → At_Risk` — which correlation cannot detect because it ignores interaction structure.
- **Lift as the key metric:** Lift measures how much more likely a rule fires than random chance. Lift ≥ 1.5 ensures rules represent genuine behavioral associations, not spurious co-occurrences driven by base rates.

### 4.2 Why an Ensemble (XGBoost + LightGBM + Random Forest)?

Each component is justified independently:

| Component | Justification | What it adds |
|---|---|---|
| **Random Forest** | Bagging ensemble; robust to overfitting; provides feature importance as baseline | Establishes ensemble floor; interpretable importance scores |
| **XGBoost** | Boosting; handles class imbalance via scale_pos_weight; primary SHAP target (TreeExplainer is exact for tree models) | Best single-model performance expected; SHAP source |
| **LightGBM** | Gradient boosting with leaf-wise growth; faster than XGBoost on large datasets | Benchmarks speed vs. accuracy trade-off on 269K records |
| **Soft Voting Ensemble** | Averages predicted probabilities across all three | Reduces variance from any single model's weaknesses |

**Why not just XGBoost alone?** The soft voting ensemble is the final model because it reduces variance. However, XGBoost is the SHAP target because TreeExplainer provides exact (not approximate) SHAP values for tree-based models — making it the most reliable source for the ARM–SHAP consistency check.

**Why Optuna over GridSearchCV?** With 5+ hyperparameters per model, grid search is computationally infeasible. Optuna's Bayesian optimization (50 trials) converges to near-optimal configurations in a fraction of the time, with a convergence plot as evidence that tuning contributed meaningfully.

### 4.3 Why SHAP?

SHAP is not included as a buzzword — it serves a specific analytical function:

- **Global SHAP (summary plot):** Ranks features by mean |SHAP value| across all predictions → directly comparable to ARM feature frequency in top rules.
- **Local SHAP:** Explains individual predictions → enables the "tipping point" narrative for specific patient profiles.
- **Dependence plots for BMI and Age:** Reveals non-linear threshold effects (e.g., risk accelerates sharply above BMI=30) that correlation coefficients cannot capture.
- **ARM–SHAP Jaccard check:** The formal cross-validation between methods is the novel analytical contribution of this project. If both methods agree on the top features, the finding is robust. If they disagree, the discrepancy reveals something meaningful about the difference between joint co-occurrence (ARM) and marginal conditional contribution (SHAP).

**Why not LIME instead of SHAP?** LIME produces locally faithful but globally inconsistent explanations. SHAP has theoretical guarantees (Shapley values from cooperative game theory) and is consistent across local and global views — essential for the cross-method comparison.

---

## 5. Proposed Approach (CRISP-DM)

### Phase 1 — Data Preparation
- Binarize target (documented in Section 3.3)
- BMI outlier experiment: IQR-capping (Strategy A) vs. record removal (Strategy B) — both carried through modeling for comparison
- Discretize continuous/ordinal features into clinically meaningful bins for ARM
- Stratified 80/20 global split; 5-fold stratified CV on training set
- Class balancing experiment inside CV folds (Section 3.4)

### Phase 2 — Association Rule Mining
- Algorithm: FP-Growth (mlxtend)
- Input: discretized binary transaction matrix
- Thresholds: min_support=0.05, min_confidence=0.60, min_lift=1.5
- Filter: consequent = {Diabetes_binary=1} only
- Output: top 30 rules by lift, with clinical interpretation

### Phase 3 — Ensemble Classification
- Baselines: Logistic Regression, Decision Tree (untuned)
- Tuned models: Random Forest, XGBoost, LightGBM (Optuna, 50 trials each)
- Final model: Soft Voting Ensemble

### Phase 4 — Explainability & Cross-Method Validation
- SHAP TreeExplainer on XGBoost: summary plot, beeswarm, dependence plots
- Youth sub-analysis (Age codes 1–4, n ≈ 36,273): re-run SHAP to compare feature importance
- ARM–SHAP Jaccard consistency check: formal set comparison, score reported and interpreted

---

## 6. Evaluation Plan

### Classification Metrics

| Metric | Priority | Target |
|---|---|---|
| Recall (Sensitivity) | PRIMARY | ≥ 0.75 — minimizing False Negatives is paramount |
| F1-Score (macro) | PRIMARY | ≥ 0.70 — accounts for class imbalance |
| AUC-ROC | Secondary | ≥ 0.82 |
| MCC | Secondary | ≥ 0.45 — robust under imbalance |
| Accuracy | Reference only | Not used for model selection |

### ARM Metrics

| Metric | Threshold |
|---|---|
| Support | ≥ 0.05 (~13,450 records) |
| Confidence | ≥ 0.60 |
| Lift | ≥ 1.5 |

---

## 7. Limitations & Scope Boundaries

- **Correlation ≠ Causation:** SHAP values and ARM rules describe statistical associations within BRFSS 2015. All findings are framed as risk associations, not causal mechanisms.
- **Self-reported data bias:** BRFSS is a telephone survey. Underreporting of behaviors (alcohol, diet) introduces measurement error not correctable in the pipeline.
- **Population scope:** Findings apply to the U.S. adult population in 2015. No cross-population claims are made.
- **Decision support only:** All outputs are Decision Support System (DSS) tools — intended to assist, not replace, clinical judgment.
- **SHAP additive approximation:** SHAP assumes feature independence. Correlated features (e.g., HighBP and HeartDiseaseorAttack, r≈0.38) may have importance distributed arbitrarily between them — flagged where relevant.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| FP-Growth generates >500 rules | Raise min_support to 0.10 and min_lift to 2.0; add confidence ≥ 0.70 filter |
| SMOTE causes train/test Recall gap | Switch to class_weight='balanced' — no synthetic data, no leakage risk |
| Optuna exceeds compute budget | Reduce to n_trials=20 with MedianPruner; fallback to RandomizedSearchCV n_iter=30 |
| ARM and SHAP findings contradict | Treated as informative finding — analyzed in Discussion as methodological difference |
| Youth sub-analysis too sparse for ARM | Lower min_support to 0.03; if still insufficient, report SHAP-only for youth segment |
