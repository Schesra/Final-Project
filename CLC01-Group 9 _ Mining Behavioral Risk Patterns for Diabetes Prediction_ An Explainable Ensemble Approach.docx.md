**CLC01 \- GROUP 9**

# **1\.  Project Title**

**Mining Behavioral Risk Patterns for Diabetes Prediction: An Explainable Ensemble Approach**

# **2\.  Track Alignment**

| Track | Role in this Project | Concrete Methods |
| ----- | ----- | ----- |
| Data Preparation & Cleaning | Primary — foundational to all subsequent work | Target binarization (3→2 classes), BMI outlier capping, feature discretization for ARM, stratified train/test split, SMOTE balancing only on training folds |
| Association Rule Mining | Secondary — interpretable pattern discovery | FP-Growth on discretized behavioral features; rule filtering by lift ≥ 1.5 and clinical relevance; cross-validation against SHAP rankings |
| Outlier / Anomaly Detection | Supporting — data quality assurance | Detection of 993 records with BMI \> 60 (physiologically implausible); comparison of IQR-capping vs. record removal strategies |

# **3\.  Problem Statement / Research Question**

## **3.1  Background & Motivation**

Type 2 diabetes has escalated into one of the most pressing global health challenges of the 21st century. According to the **IDF Diabetes Atlas 2025**, approximately **11.1% of adults aged 20–79 (roughly 1 in 9\)** live with diabetes worldwide, and over **40% remain undiagnosed**. In Vietnam, an estimated 2.5 million people are affected. Critically, more than 90% of Type 2 cases are directly linked to modifiable lifestyle behaviors — making early, data-driven behavioral intervention the most viable prevention strategy.

Despite this, most predictive models operate as **"black boxes"** — producing probability scores without explaining *which behavioral combinations* drive risk and *how strongly*. This opacity limits their clinical utility: a healthcare practitioner cannot act on a score alone; they need interpretable, actionable evidence about specific behavioral thresholds.

## **3.2  Research Question**

**RQ1 —** Which combinations of lifestyle and biological indicators (identified via Association Rule Mining with FP-Growth) form the most critical behavioral “tipping points” for diabetes risk?

**RQ2 —** To what extent can a hyperparameter-tuned Ensemble model (XGBoost \+ LightGBM \+ Random Forest) accurately predict diabetes risk, and how well do the SHAP explanations (Global & Local) align with the high-lift rules discovered in RQ1?

**RQ3 —** In the younger demographic (ages 18–44), which behavioral features drive early-onset diabetes risk, and how do these drivers differ from the general population in terms of both ARM rules and SHAP importance?

## **3.3  What Makes This Project Different from Typical Kaggle Notebooks**

1. Correct handling of the 3-class target: We binarize the target based on actionable clinical responses rather than convenience (detailed justification in Section 5.2).

2. Dual-method pipeline with cross-validation: We combine Association Rule Mining and Ensemble Classification within a single pipeline — then use SHAP to validate whether both methods agree on which features drive risk. Most Kaggle submissions apply only one approach. The cross-validation between methods is the novel analytical contribution.

3. Youth-segmented analysis: Dedicated sub-analysis for the 18–44 age group to identify early-stage behavioral triggers.

4. Explicit consistency test between ARM and XAI: After obtaining SHAP global feature rankings and FP-Growth top rules, we formally compare the two feature sets. If GenHlth appears as a top SHAP predictor but rarely in high-lift rules, that is itself an informative finding about the limits of each method — and will be discussed critically in the final report.

# **4\.  Why This Topic?  (INSIGHT Required)**

## **4.1  Personal Relevance & Urgency**

We chose this topic because it directly intersects with behavioral patterns common among university students — the very demographic we belong to. Sedentary study routines, chronic sleep deprivation, irregular meals, and the prevalence of fast-food consumption are lifestyle traps that accumulate silently. The standard clinical response — "you should exercise more and eat better" — lacks the specificity to motivate change. We are driven by a genuine scientific question:

***Can data quantitatively identify the precise behavioral "tipping point" at which accumulated unhealthy habits cross the threshold into pathology — and how close to that threshold is a typical young adult today?***

This is not merely an academic exercise. Vietnam's diabetes burden — 2.5 million affected, with prevalence rising fastest among adults under 45 — makes the youth-focused angle of this project directly relevant to public health in our own context.

## **4.2  Why Data Mining**

* Traditional clinical diagnostics are reactive: they require symptoms to be already present before testing begins. Data mining enables proactive identification of high-risk individuals before symptoms appear.

* Association Rule Mining can surface co-occurrence patterns (e.g., "sedentary \+ high BP \+ poor self-rated health → diabetes risk") that individual clinical tests cannot detect, because it captures the interaction structure rather than individual feature values.

* Explainable AI (SHAP, PDP) bridges the gap between model accuracy and clinical trust: a clinician can act on "BMI above 32 combined with no physical activity raises predicted risk by 18 percentage points" in a way they cannot act on a black-box probability score.

* The scale of the BRFSS dataset (269,131 records) requires computational approaches that go beyond what manual analysis or simple statistics can achieve — making this genuinely a Big Data Mining problem.

## **4.3  Expected Findings**

* Quantified risk tipping points: We expect to identify specific thresholds (e.g., BMI ≥ 30 \+ HighBP \= 1 \+ PhysActivity \= 0\) that predict diabetes risk with high lift. These thresholds will be expressed as actionable clinical rules.

* SHAP will confirm GenHlth (r=0.31) and HighBP (r=0.29) as the dominant predictors — but we expect BMI and Age to show non-linear SHAP dependence plots, revealing threshold effects not visible in correlation analysis.

* The youth sub-analysis will show that PhysActivity and HvyAlcoholConsump have stronger relative importance for younger cohorts compared to the full population, where biological indicators (HighBP, HighChol) dominate.     

* The ensemble (XGBoost \+ LightGBM \+ Random Forest) will outperform all baselines by at least 5 percentage points on macro F1, with the tuning contribution from Optuna being quantifiable as a standalone experiment.

# **5\.  Dataset Plan**

## **5.1  Dataset Overview**

| Attribute | Details |
| ----- | ----- |
| Name | [Diabetes Health Indicators Dataset (BRFSS 2015\)](https://drive.google.com/drive/folders/1sozbahwF3JMJt3rMh-ca3TQzFxA_rfeU?usp=drive_link) |
| Source | Kaggle — alexteboul/diabetes-health-indicators-dataset  (data.csv) |
| Records | 269,131 rows  (verified from loaded file) |
| Features | 22 columns:  1 target  \+  21 behavioral / biological indicators |
| Feature types | 14 binary features  \+  7 ordinal / continuous features (see breakdown below) |
| Time range | 2015 CDC Behavioral Risk Factor Surveillance System (BRFSS) national survey |
| Missing values | 0 missing values  (dataset is structurally clean) |
| Duplicate rows | 0 duplicate rows  (verified) |

## **5.2  Critical Finding: Three-Class Target — Binarization Decision**

The dataset contains three distinct target labels, not two. This is a key data characteristic that was not addressed in the original draft and directly caused the instructor's concern about insufficient problem framing. 

| Original Class | Clinical Meaning | Count | Proportion | After Binarization |
| ----- | ----- | ----- | ----- | ----- |
| 0 | No diabetes / No pre-diabetes | 194,377 | 72.22% | Class 0  (No Risk) |
| 1 | Pre-diabetes | 39,657 | 14.74% | Class 1  (At Risk) |
| 2 | Diabetes | 35,097 | 13.04% | Class 1  (At Risk) |

\=\> **JUSTIFICATION**

Clinical justification for binarization: In the context of behavioral risk prevention — which is the explicit goal of this project — both pre-diabetes and diabetes require the same immediate behavioral intervention (dietary change, increased physical activity, weight management). Treating them as a single "at-risk" class is not a simplification for convenience; it reflects the clinical reality that the actionable response is identical for both groups. This decision is therefore medically grounded, not arbitrary.

After binarization: Class balance is 72.22% (No Risk) vs. 27.78% (At Risk). The imbalance is pronounced but manageable, and class balancing remains a required step in the pipeline.

## **5.3  Feature Inventory**

| Feature | Type | Description | Correlation with Target |
| ----- | ----- | ----- | ----- |
| GenHlth | Ordinal 1–5 | Self-rated general health (1=excellent, 5=poor) | 0.310  (strongest predictor) |
| HighBP | Binary 0/1 | Ever told had high blood pressure | 0.289 |
| BMI | Continuous | Body Mass Index (range 12–98; outliers detected) | 0.229 |
| HighChol | Binary 0/1 | High cholesterol | 0.223 |
| DiffWalk | Binary 0/1 | Difficulty walking or climbing stairs | 0.223 |
| Age | Ordinal 1–13 | Age group (1=18–24, ..., 13=80+) | 0.208 |
| HeartDiseaseorAttack | Binary 0/1 | Coronary heart disease or myocardial infarction | 0.179 |
| PhysHlth | Ordinal 0–30 | Days of poor physical health (past 30 days) | 0.170 |
| Income | Ordinal 1–8 | Household income level (inverse relationship) | \-0.159 |
| PhysActivity | Binary 0/1 | Physical activity in past 30 days | \-0.113 |
| Education | Ordinal 1–6 | Education level (inverse relationship) | \-0.117 |
| HvyAlcoholConsump | Binary 0/1 | Heavy alcohol consumption | \-0.078 |

Remaining binary features: Smoker, Stroke, CholCheck, Fruits, Veggies, AnyHealthcare, NoDocbcCost, Sex — included in model training and Association Rule Mining.

## **5.4  Data Quality Issues & Handling Strategy**

| Issue | Scope | Planned Treatment | Experiment Design |
| ----- | ----- | ----- | ----- |
| BMI outliers (BMI \> 60\) | 993 records (0.37%); max \= 98 — physiologically implausible | Compare two strategies: (A) IQR capping at Q3 \+ 1.5×IQR; (B) record removal | Train both versions of the dataset and compare downstream model performance — treating this as an experiment, not just a preprocessing step |
| Class imbalance | 72.22% No Risk vs 27.78% At Risk after binarization | Evaluate three strategies: SMOTE, NearMiss undersampling, class\_weight='balanced' in classifier | Each strategy is a separate experimental condition; best strategy selected by macro F1 on validation fold |
| Feature scale heterogeneity | Binary features (0/1) mixed with continuous (BMI) and ordinal (GenHlth 1–5) | MinMaxScaler applied to continuous/ordinal features for Logistic Regression baseline only; tree-based models are scale-invariant | No scaling applied to XGBoost/LightGBM/RF — documented explicitly |

## **5.5  Train/Test Split & Cross-Validation**

| Step | Decision | Rationale |
| ----- | ----- | ----- |
| 1\. Global split | 80% train (215,305) / 20% test (53,826), stratified by target class | Stratification ensures class proportions are preserved in both sets; test set is held out entirely until final evaluation |
| 2\. Validation | 5-fold Stratified K-Fold Cross-Validation on training set | 5-fold provides stable performance estimates; stratification handles class imbalance |
| 3\. Balancing scope | SMOTE / NearMiss applied ONLY within each training fold — never on validation or test folds | Applying SMOTE before splitting causes data leakage; synthetic samples must not influence validation metrics |
| 4\. Hyperparameter tuning | Optuna Bayesian optimization (n\_trials=50) on training set, evaluated by macro F1 on validation fold | Bayesian search is more efficient than GridSearchCV for 5+ hyperparameters; 50 trials balances thoroughness and runtime |
| 5\. Final evaluation | Tuned model evaluated once on held-out test set; no further tuning after this | Single test-set evaluation prevents inadvertent hyperparameter leakage through repeated testing |
| 6\. Reproducibility | Random seed \= 42 throughout all experiments | Ensures all results are exactly reproducible |

## **5.6  Privacy & Ethics**

* The BRFSS dataset is collected and anonymized by the U.S. Centers for Disease Control and Prevention (CDC) and released as a public research resource. No personally identifiable information (PII) is present.

* This project is strictly a research exercise. All claims are scoped to the 2015 BRFSS population and do not purport to generalize beyond this cohort.

* Any presentation of risk rules will include explicit disclaimers that correlation does not imply causation, consistent with responsible AI practice in healthcare contexts.

# **6\.  Proposed Approach**

The project follows an extended CRISP-DM methodology organized into four sequential phases. Each phase produces measurable outputs that feed directly into the next.

**Phase 1  —  Data Preparation & Binarization**

* Binarize target: merge Class 1 (pre-diabetes) and Class 2 (diabetes) → Class 1 (At Risk). Document label mapping explicitly in the notebook.

* BMI outlier experiment: apply IQR-capping strategy on one copy of the dataset and record-removal on another. Both versions will be carried through modeling for comparison.

* Discretize continuous/ordinal features into clinically meaningful bins for Association Rule Mining: BMI → {Underweight \<18.5, Normal 18.5–25, Overweight 25–30, Obese 30–35, Severely Obese \>35}; Age → {Young Adult 1–4, Middle-Aged 5–9, Senior 10–13}; similarly for GenHlth, MentHlth, PhysHlth, Income, Education.

* Apply stratified 80/20 global split; then 5-fold stratified CV on training set.

* Class balancing experiment: evaluate SMOTE, NearMiss, and class\_weight inside CV folds; select best by macro F1.

**Phase 2  —  Association Rule Mining**

* Algorithm: FP-Growth (mlxtend library) — superior to Apriori on datasets with 200K+ records due to compressed tree structure.

* Input: discretized binary transaction matrix; each record maps to a set of binary/categorical indicators.

* Thresholds: min\_support \= 0.05 (flags patterns in ≥5% of records, \~13,450 records), min\_confidence \= 0.60, min\_lift \= 1.5.

* Filter to consequent \= {Diabetes\_binary=1} only — we are exclusively interested in rules that predict 'At Risk'.

* Post-filter: retain top 30 rules by lift; apply domain-driven filter to exclude trivially obvious rules (e.g., {HighBP=1} alone).

* Output: formatted rule table showing Antecedent, Support, Confidence, Lift, and clinical interpretation.

* Cross-check: extract the union of features appearing in top 15 rules; compare against SHAP top-10 feature ranking (Phase 4 output).

**Phase 3  —  Ensemble Classification with Hyperparameter Tuning**

Establish baselines (Logistic Regression, Decision Tree) and train single tuned ensembles (RF, XGBoost, LightGBM) to form a Soft Voting Ensemble (see comparison matrix in Section 7.2).

**Phase 4  —  Explainability (XAI) & Cross-Method Validation**

* SHAP TreeExplainer applied to the best single model (expected: XGBoost). Generate: (a) Summary Plot — global feature importance ranked by mean |SHAP|; (b) Beeswarm Plot — distribution of SHAP values per feature; (c) Dependence Plots for BMI and Age — reveals non-linear threshold effects.

* Partial Dependence Plots (PDP) for top 3 features: shows marginal effect of each feature while averaging out others.

* Youth sub-analysis (Age codes 1–4, n ≈ 35,000): re-run SHAP on this subset to compare feature importance rankings against full population.

* ARM-SHAP consistency check: compare top-10 SHAP features vs. features in top-15 rules. Compute Jaccard similarity. Discuss agreement and discrepancies explicitly in Section 5 (Discussion) of the final report.

# **7\.  Evaluation Plan**

## **7.1  Classification Metrics**

| Metric | Formula / Notes | Priority | Target (At Risk class) |
| ----- | ----- | ----- | ----- |
| Recall (Sensitivity) | TP / (TP \+ FN) | PRIMARY — highest priority | ≥ 0.75 — minimizing False Negatives is paramount; missing an at-risk patient has greater cost than a false alarm |
| F1-Score (macro) | Harmonic mean of Precision and Recall, averaged across both classes | PRIMARY | ≥ 0.70 — accounts for class imbalance; macro average weights both classes equally |
| AUC-ROC | Area under the Receiver Operating Characteristic curve | SECONDARY | ≥ 0.82 — measures overall discriminative ability independent of threshold |
| MCC (Matthews Corr. Coeff.) | Balanced metric specifically designed for imbalanced binary classification | SECONDARY | ≥ 0.45 — more informative than accuracy when class sizes differ significantly |
| Precision | TP / (TP \+ FP) | MONITORING | Tracked but not primary objective; too-low precision may cause alarm fatigue in clinical use |
| Accuracy | Overall correct predictions | REFERENCE ONLY | Reported for completeness but not used for model selection — misleading under 72:28 imbalance |

## **7.2  Full Model Comparison Matrix**

All six models will be evaluated on the same held-out test set and presented in a unified comparison table:

| Model | Type | Tuning | Balancing | Primary Purpose in Study |
| ----- | ----- | ----- | ----- | ----- |
| Logistic Regression | Linear | None (default) | None | Baseline 1 — establishes performance floor |
| Decision Tree | Tree | None (default) | None | Baseline 2 — single non-linear model, untuned |
| Random Forest (tuned) | Ensemble | Optuna 50 trials | Best from Phase 1 | Intermediate — bagging ensemble, interpretable feature importance |
| XGBoost (tuned) | Ensemble | Optuna 50 trials | Best from Phase 1 | Intermediate — boosting; primary SHAP target |
| LightGBM (tuned) | Ensemble | Optuna 50 trials | Best from Phase 1 | Intermediate — benchmarks speed vs. accuracy trade-off |
| Soft Voting Ensemble | Ensemble | Derived | Best from Phase 1 | Final model — expected best overall performance |

## **7.3  Association Rule Evaluation**

| Metric | Threshold | Interpretation |
| ----- | ----- | ----- |
| Support | ≥ 0.05 | Rule covers at least 5% of records (\~13,450 patients) |
| Confidence | ≥ 0.60 | When antecedent is true, diabetes risk exceeds 60% |
| Lift | ≥ 1.5 | Pattern is 1.5× more likely than random co-occurrence — indicates genuine association beyond base rate |
| Rule count | Target: 15–30 high-quality rules | Too few rules: raise minimum support threshold; too many (\>100): raise lift threshold to 2.0 |

## **7.4  Additional Evaluation Dimensions**

* Imbalance strategy comparison: side-by-side table of macro F1 / Recall for SMOTE, NearMiss, and class\_weight across 5-fold CV; best strategy selected transparently.

* BMI outlier strategy comparison: model trained on IQR-capped dataset vs. record-removed dataset; comparison of test set Recall and F1 to quantify the impact of the preprocessing choice.

* Runtime analysis: wall-clock training time for each model on 215K records; reported alongside accuracy metrics to illustrate efficiency trade-offs.

* Optuna convergence plot: visualization of objective function improvement across 50 trials for the best single model, demonstrating that tuning contributed meaningfully.

* ARM–SHAP Jaccard consistency: formal set comparison between top-10 SHAP features and features in top-15 rules; Jaccard similarity score reported and interpreted.

# **8\.  Expected Outputs**

| Deliverable | Format | Content |
| ----- | ----- | ----- |
| Presentation Slides | PowerPoint / PDF | 10–15 slides summarizing motivation, key findings, top association rules, SHAP plots, and model comparison table. Designed for a 10-minute oral presentation. |
| Behavioral Risk Rules | Formatted table in report | Top 15 association rules with Support, Confidence, Lift, and a one-sentence clinical interpretation for each. Example: {HighBP=1, PhysActivity=0, BMI\_Obese=1} → At\_Risk  \[Confidence: 0.73, Lift: 2.14\]. |

# **9\.  Risks & Backup Plan**

| Risk | Likelihood | Impact | Mitigation / Backup |
| ----- | ----- | ----- | ----- |
| FP-Growth generates \>500 rules with default thresholds | High | Medium | Immediately raise min\_support to 0.10 and min\_lift to 2.0. Focus exclusively on rules with consequent \= {At\_Risk=1}. If still too many, add a third filter: confidence ≥ 0.70. |
| Model overfitting due to high inter-feature correlation (e.g., HighBP and HeartDiseaseorAttack, r ≈ 0.38) | Medium | Medium | Apply Recursive Feature Elimination (RFE) to identify and remove redundant features. Alternatively, use L1-regularized Logistic Regression to zero out weakly correlated features before ensemble modeling. |
| SMOTE causes overfitting to synthetic samples (Recall inflates on train, drops on test) | Low | High | Switch entirely to class\_weight='balanced' inside classifier, which adjusts the loss function without generating synthetic data. This is the safest fallback for large datasets. |
| Optuna tuning exceeds available compute time (50 trials × 5-fold × 3 models) | Low | Low | Reduce to n\_trials=20 with Optuna's pruning feature enabled (MedianPruner). Alternatively, replace with RandomizedSearchCV with n\_iter=30 for models that are slow to train. |
| ARM and SHAP findings are directly contradictory, making the cross-validation section difficult to write | Medium | Low | This is treated as an informative research finding, not a failure. If the methods disagree, Section 5 (Discussion) will analyze why — e.g., ARM captures joint behavioral co-occurrences while SHAP captures marginal conditional contributions — and what this tells us about the limits of each approach. |
| Youth sub-analysis (Age 1–4) has too few diabetic cases for reliable pattern mining | Medium | Medium | Lower min\_support to 0.03 for the sub-analysis, and report fewer rules with higher confidence thresholds. If the subset is genuinely too small for meaningful ARM, report this limitation explicitly and focus the youth analysis on SHAP only. |

# **10\.  Limitations & Scope Boundaries**

The instructor's review specifically flagged the absence of risk-interpretation controls for a medical XAI project. This section directly addresses that requirement. All limitations listed here will be re-stated and discussed in depth in the final report's Discussion section.

* **Correlation ≠ Causation:** SHAP values and association rules describe statistical relationships within the BRFSS 2015 survey data. A high SHAP value for HighBP does not imply that high blood pressure causes diabetes — both may be caused by shared underlying factors (obesity, age, sedentary lifestyle). All findings will be framed as risk associations, not causal mechanisms.

* **Self-Reported Data Bias:** BRFSS is a telephone-based self-reported survey. Respondents may underreport behaviors (e.g., alcohol consumption, poor diet) and subjectively rate GenHlth. This introduces measurement error that is not correctable within our pipeline.

* **Population Generalizability:** The dataset represents the U.S. adult population in 2015\. Findings may not generalize to Vietnamese populations, other cultural contexts, or more recent years. This project does not make cross-population claims.

* **Decision Support Scope:** All models, rules, and visualizations produced in this project are Decision Support System (DSS) outputs. They are intended to assist — not replace — clinical judgment. No automated diagnosis is implied or should be inferred from this work.

* **SHAP Additive Approximation:** SHAP values assume feature independence when computing individual contributions. In the presence of highly correlated features (e.g., HighBP and HeartDiseaseorAttack), SHAP may distribute importance arbitrarily between correlated predictors. This will be flagged in the report where relevant.

* **Static Snapshot:** The analysis is based on a single cross-sectional survey year. Temporal trends in diabetes prevalence or behavioral patterns cannot be inferred from this dataset alone.

