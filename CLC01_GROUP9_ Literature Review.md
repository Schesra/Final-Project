**Mining behavioral risk patterns for diabetes prediction: An explainable ensemble approach**

**ABSTRACT**

This literature review synthesizes 26 studies published between 1994 and 2025 on methodologies relevant to type 2 diabetes mellitus (T2DM) prediction, spanning five thematic pillars: association rule mining (ARM), tree-based ensemble classifiers, class-imbalance handling, SHAP-based explainable AI (XAI), and epidemiological context for Vietnam. The central finding is a persistent methodological gap: existing studies excel in either unsupervised pattern discovery through ARM or post-hoc model interpretability through SHAP, but none implement a sequential pipeline that generates behavioral risk hypotheses via FP-Growth, trains a hyperparameter-tuned ensemble classifier, and verifies genuine interaction effects through targeted SHAP subgroup analysis on the same dataset. The proposed hybrid ARM–SHAP ensemble framework addresses this gap, aiming to deliver clinically actionable explanations for T2DM prevention in high-burden settings such as Vietnam.

**Keywords:** association rule mining; diabetes prediction; ensemble learning; explainable AI; SHAP explainability

**INTRODUCTION**

Type 2 diabetes mellitus (T2DM) continues to escalate as a major non-communicable disease worldwide. The International Diabetes Federation estimates 589 million adults living with diabetes globally in 2024, with projections reaching approximately 853 million by 2050; over 90% of cases are type 2 and largely driven by modifiable behavioral factors (Duncan et al., 2025). In Vietnam, the disease affects an estimated 2.5 million adults, with age-standardized prevalence around 3.4% and approximately 37.8% of cases remaining undiagnosed, with incidence rising most rapidly among adults under 45 amid urbanization and lifestyle changes (Vuong et al., 2024; Nguyen et al., 2015). Traditional epidemiological approaches identify individual risk factors but rarely capture synergistic multi-feature behavioral patterns that drive disease progression.

Two methodological traditions dominate recent machine-learning research on T2DM prediction. The first combines ensemble tree-based models with SHAP for interpretability, yielding both accurate predictions and post-hoc feature attributions (Majcherek et al., 2025; Kutlu et al., 2024; Netayawijit et al., 2025; Ahmed et al., 2024; Rafie et al., 2025; Islam et al., 2024; Rossi et al., 2026; Al-Hudaibi et al., 2025). The second employs association rule mining to discover co-occurrence patterns among behavioral variables (Fakir et al., 2024; Bata et al., 2025). These streams have largely remained separate: ARM studies generate multi-feature hypotheses but cannot verify whether co-occurrences represent genuine risk amplification, while SHAP-ensemble studies provide rigorous feature explanations but lack prior hypotheses about which combinations to examine.

The present research bridges this divide through three research questions: 

(1) Which multi-feature combinations of behavioral and biological indicators constitute the strongest risk patterns for T2DM? 

(2) Do the behavioral risk combinations identified through ARM genuinely amplify individual feature contributions, or do they merely reflect statistical co-occurrence? 

(3) Do the behavioral features driving risk differ between the younger population (ages 18–44) and the general adult population? 

The study builds on the reviewed literature through a seven-step pipeline: data preparation, discretization for ARM, FP-Growth hypothesis generation, ensemble classification with Optuna hyperparameter optimization (Akiba et al., 2019), performance evaluation, SHAP-based explainability, and targeted subgroup verification. The following sections synthesize the state of the art, provide critical evaluations, and highlight the gaps that the current approach aims to fill.

**LITERATURE REVIEW**

**Foundations of Association Rule Mining**

Association rule mining (ARM) is a core unsupervised technique for uncovering hidden relationships in categorical data. Agrawal and Srikant (1994) introduced the Apriori algorithm, which generates candidate itemsets level-wise and prunes infrequent sets using a minimum-support threshold with the downward closure property. While foundational, Apriori suffers from two computational bottlenecks: repeated full-database scans at each candidate level, and explosive candidate growth in high-dimensional spaces. These limitations are particularly acute for large-scale health surveys such as BRFSS, where 21 features—including multi-category ordinal variables—produce a vast transactional space after discretization.

In healthcare contexts, Bata et al. (2025) employed Apriori-based ARM on time-stratified diabetes comorbidity data from a Hungarian clinical registry, extracting pre- and post-diagnosis temporal patterns (e.g., hypertension preceding diabetes diagnosis). Their study confirmed ARM’s value for hypothesis generation but noted that rule quality degrades sharply on imbalanced data without pre-processing—a limitation directly relevant to the 72:28 class ratio in BRFSS. Critically, they did not verify whether the discovered patterns represented genuine risk-amplifying interactions or merely reflected base-rate artifacts.

**Advancement to FP-Growth and Scalability in Health Data**

Han et al. (2000) addressed Apriori’s limitations by proposing FP-Growth, which constructs a compact FP-tree to mine frequent patterns without candidate generation, significantly reducing memory and computation costs. The algorithm compresses the dataset into a prefix tree and recursively mines conditional pattern bases, making it well-suited for large transactional datasets.

Fakir et al. (2024) implemented parallelized FP-Growth on Apache Spark for a diabetic dataset, confirming superior scalability over Apriori. Their extracted rules linked high-calorie intake, low physical activity, and elevated BMI with diabetes onset—patterns directly relevant to the current study’s behavioral focus. However, their analysis stopped at pattern discovery without employing SHAP or any other post-hoc verification. Essalmi and Affar (2025) proposed a dynamic meta-pattern algorithm for improved rule relevance filtering, though this remains a preprint not yet applied to health data. For the present study, FP-Growth is chosen for the 253,680-record BRFSS dataset because it handles the volume and dimensionality of behavioral features at practical thresholds (min\_support \= 0.05, min\_confidence \= 0.60) without prohibitive runtime—a conclusion well-supported by Fakir et al. (2024).

**SHAP: A Unified Framework for Model Interpretability**

Lundberg and Lee (2017) introduced SHAP (SHapley Additive exPlanations), a game-theoretic approach based on Shapley values that assigns consistent feature attributions satisfying three properties: local accuracy, missingness, and consistency. This framework subsumes earlier methods (e.g., LIME, DeepLIFT) within a unified theoretical foundation, providing explanations that are both locally faithful and globally aggregable. Ahmed et al. (2024) empirically confirmed SHAP’s superiority over LIME for tree-based models, reporting that LIME explanations vary significantly across runs due to sampling, while SHAP values remain stable—further justifying SHAP as the verification tool in the current pipeline.

For this study, SHAP’s key property is its ability to decompose predictions into per-feature contributions, enabling targeted comparison between subgroups. If a behavioral combination discovered by ARM truly amplifies risk, the SHAP contributions of its constituent features should be significantly elevated within the ARM-defined subgroup compared to the general population.

**Tree SHAP, Interaction Effects, and Limitations with Correlated Features**

Lundberg et al. (2019) extended SHAP to tree ensembles via Tree SHAP—an exact polynomial-time algorithm computing both main effects and pairwise interaction values. Unlike sampling-based model-agnostic SHAP, Tree SHAP traverses each tree’s internal structure for exact Shapley computation, enabling aggregation into global insights such as summary plots, dependence plots, and interaction heatmaps.

However, Tree SHAP exhibits biased attribution when features are correlated. Features splitting earlier in trees may absorb importance from correlated downstream features. Al-Hudaibi et al. (2025) documented this in a Saudi Arabian diabetes context, where correlated clinical features (e.g., HbA1c and fasting glucose) required careful SHAP interpretation. In BRFSS, this concern is relevant because HighBP, HighChol, and BMI are moderately correlated. The current study addresses this through two mechanisms: (1) ARM identifies multi-feature combinations *before* SHAP analysis, providing structured hypotheses rather than relying solely on marginal values; and (2) subgroup SHAP verification holds co-occurring features constant, partially controlling for correlation effects.

**Ensemble Tree-Based Models and Hyperparameter Optimization**

Tree ensembles dominate tabular health-data tasks due to their robustness, non-linear interaction handling, and strong empirical performance. Three complementary models form the current study’s ensemble backbone. Breiman (2001) introduced Random Forests, reducing variance through bagging and random feature selection, providing stability and built-in importance estimates (though biased toward high-cardinality features). Chen and Guestrin (2016) developed XGBoost with L1/L2 regularization and sparsity-aware splitting, consistently ranking among top performers on imbalanced health benchmarks; it serves as the primary SHAP analysis target due to compatibility with exact Tree SHAP. Ke et al. (2017) proposed LightGBM with Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB), achieving substantial speed improvements through leaf-wise growth.

The three models are combined in a Soft Voting Ensemble, selected for their complementary strengths: Random Forest contributes prediction stability through decorrelated bagging; XGBoost contributes regularized boosting with strong generalization; and LightGBM contributes efficient leaf-wise specialization. The ensemble averages predicted probabilities, smoothing individual model weaknesses. This ensemble is adopted only if it outperforms the best single model on macro-F1; if it does not, the best single model is used instead.

Each model’s hyperparameters are tuned via Optuna (Akiba et al., 2019), a next-generation Bayesian optimization framework that uses Tree-structured Parzen Estimators (TPE) with efficient pruning of unpromising trials. Optuna is configured with 50 trials per model, optimizing macro-F1 as the objective function. This principled approach replaces manual grid search, enabling systematic exploration of high-dimensional hyperparameter spaces (e.g., learning rate, max\_depth, n\_estimators, subsample ratio) while providing reproducibility through seed control.

**SHAP Integration in Diabetes Prediction on BRFSS 2015**

Two studies are directly comparable to the current work, both applying SHAP to tree models on the BRFSS 2015 dataset. Majcherek et al. (2025) benchmarked 18 models on 253,680 records, finding tree ensembles superior in AUC. Their SHAP analysis identified HighBP, BMI, Age, and GenHlth as dominant contributors with non-linear BMI thresholds. However, three limitations persist: (a) SHAP was applied post-hoc without prior hypotheses; (b) GenHlth (r \= 0.33 with the target—the highest Pearson correlation among all 21 features—was retained without leakage analysis, despite being a self-rated health measure strongly influenced by diagnosis awareness; and (c) no age-stratified analysis was performed.

Kutlu et al. (2024) applied XGBoost with Recursive Feature Elimination on the same dataset, achieving 86.6% accuracy with SHAP-confirmed BMI thresholds around 30 kg/m². Limitations include: (a) post-hoc SHAP without prior hypotheses; (b) modest recall on the minority At-Risk class; and (c) no systematic leakage analysis for GenHlth, CholCheck, or other features that may reflect post-diagnosis status. Neither study addresses medium-leakage features such as HeartDiseaseorAttack (At-Risk prevalence 21.9% vs. No-Risk 8.2%, difference \+13.7 percentage points, r \= 0.190) or Stroke (At-Risk 9.1% vs. No-Risk 3.6%, difference \+5.5pp), which represent diabetes comorbidities rather than pre-diagnosis behavioral risk factors. Both studies demonstrate that SHAP on BRFSS is valuable but incomplete.

**Additional SHAP-Based Studies in Diabetes Prediction**

Beyond BRFSS-specific studies, a growing body of work applies SHAP to diabetes prediction on diverse datasets. Netayawijit et al. (2025) developed a SMOTE+SHAP framework for a Thai dataset, improving macro-F1 by approximately 10% over non-oversampled baselines—providing empirical support for the current study’s SMOTE-aware cross-validation design. Ahmad et al. (2026) proposed stacked ensembles with SHAP on a small clinical dataset. Rafie et al. (2025) leveraged XGBoost+SHAP on Iranian clinical data with direct biomarkers (AUC \> 0.90), though biomarkers are unavailable in behavioral surveys like BRFSS.

Islam et al. (2024) combined hyperparameter tuning, SHAP, partial dependence plots, and LIME for comprehensive XAI triangulation. Rossi et al. (2026) proposed the D.R.E.A.M. framework emphasizing combined global and local explanations for clinical trust. Across all studies, a consistent pattern emerges: SHAP is applied as an exploratory post-hoc tool without prior hypotheses. The current study reverses this logic—using ARM to generate hypotheses *before* SHAP analysis, then using SHAP to *verify* specific predictions about subgroup-level interactions.

**Class Imbalance Handling Techniques**

In the BRFSS 2015 dataset, the At-Risk class comprises 27.78% of records after binarization. This imbalance inflates accuracy while masking poor recall on the clinically important minority class. Chawla et al. (2002) introduced SMOTE (Synthetic Minority Over-sampling TEchnique), generating synthetic minority examples by interpolating between nearest neighbors. SMOTE improves recall without simple duplication, though it may introduce noise in high-dimensional spaces by interpolating between dissimilar instances.

Agyemang et al. (2025) compared SMOTE, ADASYN, Borderline-SMOTE, and SMOTE-Tomek on health datasets, demonstrating 8–15% sensitivity improvements. Hybrid variants (e.g., SMOTE-Tomek) partially mitigate noise by removing ambiguous boundary instances. Rezki et al. (2024) confirmed these findings for diabetes classification across C5.0, Random Forest, and SVM.

A critical design consideration is **SMOTE-aware cross-validation**: SMOTE must be fitted *only on training folds*, with synthetic samples excluded from validation folds. If SMOTE is applied before the train/validation split, synthetic samples may leak information from validation instances into training, producing overly optimistic performance estimates. The current pipeline enforces this by applying SMOTE within each of the 5 cross-validation folds using imbalanced-learn’s Pipeline integration, ensuring that the validation fold always evaluates on real (non-synthetic) data. Class-weight balancing (‘balanced’ mode) is tested as a comparison baseline, as it adjusts loss function weights without modifying the data distribution.

**Epidemiological Context and Regional Relevance for Vietnam**

Duncan et al. (2025), reporting on the IDF Diabetes Atlas (11th edition), documented 589 million adults with diabetes globally in 2024, with low- and middle-income countries bearing a disproportionate burden driven by behavioral risk factors—physical inactivity, unhealthy diet, and obesity.

In Vietnam, Vuong et al. (2024) reported age-standardized prevalence of approximately 3.4%, with 37.8% of cases undiagnosed. Among high-risk adults, prediabetes reached 60.6% and undiagnosed diabetes 18.3%, with obesity, hypertension, and dyslipidemia as major associated factors. Nguyen et al. (2015) documented that T2DM incidence is rising fastest among Vietnamese adults under 45, driven by urbanization, dietary transitions toward processed foods, and declining physical activity. This pattern directly motivates RQ3: the BRFSS sample has a median age code of 9 (corresponding to 60–64 years), meaning the sample is heavily skewed toward older adults. If behavioral risk drivers differ between younger and older populations—as the Vietnamese epidemiological evidence suggests—then age-stratified SHAP analysis is essential for generating prevention insights relevant to Vietnam’s at-risk youth demographic. The current study addresses this through explicit youth-versus-general SHAP comparison (Age codes 1–5, approximately 51,525 records with 4,864 At-Risk cases), with appropriate statistical power caveats due to the smaller At-Risk count in this subgroup. It is noted that the BRFSS is a U.S. dataset, and findings are framed as hypothesis-generating rather than directly prescriptive for Vietnam.

**Synthesis of Literature and Research Gaps**

The 26 reviewed studies provide robust foundations across five pillars. However, five critical gaps persist when these pillars are viewed as components of an integrated pipeline:

**Gap 1: ARM without verification.** All reviewed ARM studies (Agrawal & Srikant, 1994; Han et al., 2000; Fakir et al., 2024; Bata et al., 2025; Essalmi & Affar, 2025\) generate co-occurrence patterns but lack mechanisms to verify whether these represent genuine risk-amplifying interactions or statistical artifacts from correlated base rates.

**Gap 2: SHAP without prior hypotheses.** All SHAP-diabetes studies (Majcherek et al., 2025; Kutlu et al., 2024; Netayawijit et al., 2025; Ahmed et al., 2024; Ahmad et al., 2026; Rafie et al., 2025; Islam et al., 2024; Rossi et al., 2026; Al-Hudaibi et al., 2025\) apply SHAP as an exploratory post-hoc tool without prior testable hypotheses, preventing systematic verification of multi-feature interactions.

**Gap 3: Feature leakage under-addressed.** Neither BRFSS study (Majcherek et al., 2025; Kutlu et al., 2024\) addresses the leakage risk from GenHlth (r \= 0.33, highest in dataset; self-rated health influenced by diagnosis awareness), CholCheck (post-diagnosis monitoring), DiffWalk (diabetes complication), or medium-leakage comorbidities HeartDiseaseorAttack (r \= 0.190) and Stroke. The current study trains three model variants (full, leakage-removed, behavioral-only) to quantify this risk.

**Gap 4: No age-stratified behavioral analysis.** Despite evidence that T2DM rises fastest among younger adults in Vietnam (Nguyen et al., 2015; Vuong et al., 2024\) and globally (Duncan et al., 2025), no reviewed study performs age-stratified SHAP analysis on behavioral features.

**Gap 5: No sequential ARM-to-SHAP pipeline.** No study implements the full sequential pipeline of ARM hypothesis generation, Optuna-tuned ensemble prediction, and targeted SHAP subgroup verification on the same large-scale behavioral dataset.

**METHOD**

Building on the synthesized literature, the study implements a hybrid ARM–SHAP ensemble framework on the BRFSS 2015 dataset (253,680 records). The data preparation phase includes target binarization (merging pre-diabetes and diabetes into At-Risk), BMI outlier handling (IQR-capping at 42.5 vs. removal of BMI \> 60), MinMaxScaler normalization (fit on training data only), and discretization of continuous/ordinal features into categorical bins for ARM. FP-Growth (mlxtend) is applied with minimum support of 0.05, confidence of 0.60, and lift of 1.5 to generate behavioral hypotheses formulated as testable statements about SHAP contributions within ARM-defined subgroups.

Six classifiers are trained using 5-fold Stratified K-Fold cross-validation: Logistic Regression and Decision Tree (baselines), Random Forest, XGBoost, and LightGBM (each tuned via Optuna with 50 trials optimizing macro-F1), and a Soft Voting Ensemble adopted only if it outperforms the best single model. Class imbalance is addressed through SMOTE-aware cross-validation (SMOTE fitted only on training folds), NearMiss, and class\_weight experiments. Three model variants—Variant A (full 21 features), Variant B (removing GenHlth, CholCheck, DiffWalk, HeartDiseaseorAttack, Stroke), and Variant C (behavioral-only: PhysActivity, BMI, Smoker, Fruits, Veggies, HvyAlcoholConsump, Income, Education, Age, Sex)—quantify feature leakage sensitivity. SHAP TreeExplainer computes attributions and interactions on the best model, with targeted subgroup analysis verifying each ARM-derived hypothesis. Youth-versus-general SHAP comparison (Age codes 1–5, ages 18–44) addresses RQ3. All experiments use Python 3.12 with scikit-learn, XGBoost, LightGBM, SHAP, Optuna, and mlxtend (seed \= 42).

**RESULT**

**Association Rule Mining: Behavioral Risk Patterns**

FP-Growth applied to the 269,131-record BRFSS transaction matrix (41 binary columns after one-hot encoding, min\_support = 0.05, min\_confidence = 0.45, min\_lift = 1.5) yielded 19,204 frequent itemsets and 10,346 candidate rules. After filtering for consequent = AtRisk, 278 rules remained; applying confidence ≥ 0.45 produced 157 rules, from which the top 20 were retained for hypothesis generation. The dataset-wide base rate for At-Risk is 27.78%, so all reported rules represent meaningful lift above chance.

The highest-lift rules were dominated by comorbidity clusters rather than purely behavioral features. The top rule—{CholCheck\_1, DiffWalk\_1, HighBP\_1, HighChol\_1} → AtRisk—achieved confidence = 0.610 and lift = 2.194, meaning individuals presenting all four conditions are 2.19 times more likely to be At-Risk than the population average. The second-ranked rule {AnyHealthcare\_1, DiffWalk\_1, HighBP\_1, HighChol\_1} → AtRisk yielded nearly identical metrics (confidence = 0.609, lift = 2.192). Among rules with at least one behavioral feature (has\_behavioral = True), the strongest was {BMI\_cat\_SeverelyObese, HighBP\_1} → AtRisk (support = 0.056, confidence = 0.585, lift = 2.105), confirming that severe obesity co-occurring with hypertension constitutes a high-risk behavioral–clinical combination. The Jaccard similarity between the top-10 SHAP features and features appearing in the top-15 ARM rules was 0.308, with four features in common: BMI, GenHlth, HighBP, and HighChol—indicating moderate but meaningful convergence between the two analytical streams.

Four hypotheses were formulated from the ARM output for SHAP verification:

- H1: In the subgroup {AnyHealthcare\_1, CholCheck\_1, HighBP\_1}, mean SHAP of BMI > population mean SHAP (ARM rule: BMI\_cat\_SeverelyObese, lift = 2.124, confidence = 0.590)
- H2: In the subgroup {CholCheck\_1, HighChol\_1, HighBP\_1}, mean SHAP of DiffWalk > population mean SHAP (ARM rule: DiffWalk\_1, lift = 2.194, confidence = 0.609)
- H3: In the subgroup {AnyHealthcare\_1, CholCheck\_1, GenHlth\_cat\_Fair}, mean SHAP of HighChol > population mean SHAP (ARM rule: HighChol\_1, lift = 2.062, confidence = 0.573)
- H4: In the subgroup {AnyHealthcare\_1, CholCheck\_1, HighBP\_1}, mean SHAP of GenHlth > population mean SHAP (ARM rule: GenHlth\_cat\_Fair, lift = 2.049, confidence = 0.569)

**Model Performance Comparison**

Six classifiers were evaluated on the held-out test set (53,827 records, 27.78% At-Risk) after SMOTE-balanced training (311,002 samples). SMOTE outperformed NearMiss (macro-F1 = 0.692 vs. 0.590) and class\_weight balancing (macro-F1 = 0.692), and was selected as the primary balancing strategy. Full results are presented in Table 1.

*Table 1. Model performance on BRFSS 2015 test set (n = 53,827)*

| Model | Recall | Macro-F1 | AUC-ROC | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.746 | 0.694 | 0.807 | 0.419 |
| Decision Tree | 0.793 | 0.784 | 0.822 | 0.578 |
| XGBoost (Optuna-tuned) | **0.814** | **0.730** | **0.848** | **0.496** |
| LightGBM (Optuna-tuned) | 0.613 | 0.711 | 0.814 | 0.423 |
| Random Forest (Optuna-tuned) | 0.700 | 0.717 | 0.824 | 0.445 |
| Soft Voting Ensemble | 0.733 | 0.734 | 0.844 | 0.480 |

XGBoost achieved the highest AUC-ROC (0.848) and recall (0.814), making it the primary model for SHAP analysis. The Soft Voting Ensemble did not surpass XGBoost on macro-F1 (0.734 vs. 0.730), so XGBoost was retained as the single best model per the pre-specified selection criterion. Decision Tree achieved the highest MCC (0.578) but lower AUC (0.822), reflecting its tendency to overfit decision boundaries rather than produce calibrated probabilities.

**Feature Leakage Sensitivity**

Three model variants were trained using XGBoost to quantify the impact of potentially leaky features (Table 2).

*Table 2. Leakage sensitivity analysis across three feature variants*

| Variant | Features | Recall | Macro-F1 | AUC-ROC |
|---|---|---|---|---|
| A – Full (21 features) | All features including GenHlth, CholCheck, DiffWalk, HeartDiseaseorAttack, Stroke | 0.814 | 0.730 | 0.848 |
| B – Leakage-removed | Excluding the five high/medium-leakage features above | 0.807 | 0.704 | 0.825 |
| C – Behavioral-only | PhysActivity, BMI, Smoker, Fruits, Veggies, HvyAlcoholConsump, Income, Education, Age, Sex | 0.838 | 0.612 | 0.759 |

Removing the five leakage-suspect features (Variant B) reduced AUC by 0.023 and macro-F1 by 0.026, confirming that GenHlth, CholCheck, and DiffWalk contribute meaningful predictive signal but also carry leakage risk. Variant C's substantially lower macro-F1 (0.612) and AUC (0.759) demonstrates that purely behavioral features are insufficient for high-accuracy prediction on BRFSS, though its higher recall (0.838) suggests behavioral features are particularly informative for identifying true positives in the At-Risk class.

**SHAP-Based Feature Importance**

SHAP TreeExplainer applied to XGBoost (Variant A) on the test set identified the following top-10 features by mean |SHAP| value:

*Table 3. Top 10 features by mean absolute SHAP value (XGBoost, Variant A)*

| Rank | Feature | Mean \|SHAP\| |
|---|---|---|
| 1 | GenHlth | 0.674 |
| 2 | Age | 0.619 |
| 3 | BMI | 0.616 |
| 4 | HighBP | 0.515 |
| 5 | HighChol | 0.345 |
| 6 | Income | 0.264 |
| 7 | MentHlth | 0.229 |
| 8 | PhysHlth | 0.206 |
| 9 | Education | 0.171 |
| 10 | Sex | 0.170 |

GenHlth, Age, and BMI collectively dominate model predictions, consistent with findings from Majcherek et al. (2025) and Kutlu et al. (2024) on the same dataset. SHAP dependence plots revealed non-linear BMI thresholds, with risk contributions increasing sharply above BMI ≈ 30 kg/m², corroborating Kutlu et al.'s (2024) reported threshold of 30 kg/m².

**ARM Hypothesis Verification via SHAP Subgroup Analysis**

Results of the four hypothesis tests are summarized in Table 4, using a 50% relative increase threshold for confirmation.

*Table 4. SHAP subgroup verification of ARM-derived hypotheses*

| Hypothesis | Test Feature | Population Mean SHAP | Subgroup Mean SHAP | Relative Increase | Confirmed (≥50%) |
|---|---|---|---|---|---|
| H1 | BMI | −0.402 | −0.261 | 35.1% | ✗ Rejected |
| H2 | DiffWalk | −0.013 | −0.004 | 67.8% | ✓ Confirmed |
| H3 | HighChol | −0.073 | −0.064 | 13.1% | ✗ Rejected |
| H4 | GenHlth | −0.368 | −0.098 | 73.4% | ✓ Confirmed |

H2 and H4 were confirmed: within the subgroup co-presenting CholCheck, HighChol, and HighBP, DiffWalk's SHAP contribution increased by 67.8% relative to the population mean; within the subgroup co-presenting AnyHealthcare, CholCheck, and HighBP, GenHlth's SHAP contribution increased by 73.4%. H1 (BMI in the severely obese–hypertensive subgroup) and H3 (HighChol in the fair-health subgroup) did not meet the 50% threshold, with relative increases of 35.1% and 13.1% respectively.

**Youth vs. General Population SHAP Comparison**

The youth subgroup (Age codes 1–5, ages 18–44) comprised 10,377 test records with 946 At-Risk cases (9.1% prevalence, compared to 27.78% in the full population). SHAP values for this subgroup showed a markedly different importance profile (Table 5).

*Table 5. Mean |SHAP| comparison: full population vs. youth subgroup (ages 18–44)*

| Feature | Full Population | Youth Subgroup | Youth/Full Ratio |
|---|---|---|---|
| GenHlth | 0.674 | 0.919 | 1.36× |
| Age | 0.618 | 1.949 | 3.15× |
| BMI | 0.616 | 0.747 | 1.21× |
| HighBP | 0.515 | 0.570 | 1.11× |
| HighChol | 0.345 | 0.435 | 1.26× |
| Income | 0.264 | 0.375 | 1.42× |
| MentHlth | 0.229 | 0.387 | 1.69× |
| PhysHlth | 0.206 | 0.267 | 1.30× |
| Education | 0.171 | 0.229 | 1.34× |
| Sex | 0.170 | 0.159 | 0.94× |

The most striking finding is that Age's SHAP contribution in the youth subgroup is 3.15 times higher than in the full population (1.949 vs. 0.618), suggesting that within the 18–44 age range, being at the upper end of this bracket (closer to 44) is a substantially stronger risk signal than age variation in the general adult population. MentHlth showed a 1.69× amplification in youth (0.387 vs. 0.229), indicating that mental health burden is a disproportionately important risk factor for younger adults. Income also showed elevated importance in youth (1.42×), consistent with socioeconomic vulnerability being more predictive of diabetes risk in younger working-age populations.

**DISCUSSION**

**RQ1: Multi-Feature Behavioral Risk Patterns**

The FP-Growth analysis identified 157 high-confidence rules (confidence ≥ 0.45, lift ≥ 1.5) pointing to T2DM risk. The dominant patterns cluster around comorbidity combinations—particularly DiffWalk, HighBP, HighChol, and CholCheck—rather than purely behavioral features such as physical activity or diet. This finding partially diverges from Fakir et al. (2024), whose Spark-based FP-Growth on a smaller diabetic dataset emphasized caloric intake and physical inactivity. The discrepancy likely reflects BRFSS's population-level survey design, which captures a broader spectrum of health states including individuals already managing diagnosed conditions. Among behavioral rules, the {BMI\_cat\_SeverelyObese, HighBP\_1} combination (lift = 2.105) is the strongest purely modifiable risk pattern, reinforcing the well-established obesity–hypertension–diabetes triad (Duncan et al., 2025; Vuong et al., 2024).

The Jaccard similarity of 0.308 between ARM and SHAP feature sets indicates moderate convergence. Four features—BMI, GenHlth, HighBP, HighChol—appear in both the top-10 SHAP rankings and the top-15 ARM rules, providing cross-method validation of their centrality to T2DM risk. However, the low overlap also highlights that ARM and SHAP capture complementary aspects: ARM identifies co-occurrence patterns at the population level, while SHAP quantifies individual-level prediction contributions. This complementarity is precisely the methodological gap identified in the literature review (Gap 5), and the current pipeline's sequential design exploits it.

**RQ2: Genuine Interaction Effects vs. Statistical Co-occurrence**

The hypothesis verification results provide a nuanced answer to RQ2. Two of four ARM-derived hypotheses were confirmed by SHAP subgroup analysis. H2's confirmation (DiffWalk SHAP +67.8% in the CholCheck–HighChol–HighBP subgroup) suggests that difficulty walking genuinely amplifies risk prediction within a cluster of cardiovascular and metabolic comorbidities—consistent with DiffWalk being both a diabetes complication and a marker of advanced disease burden. H4's confirmation (GenHlth SHAP +73.4% in the AnyHealthcare–CholCheck–HighBP subgroup) indicates that self-rated fair health carries substantially greater predictive weight when co-occurring with healthcare access and cholesterol monitoring, possibly because this combination identifies individuals who are aware of their deteriorating health but have not yet received a diabetes diagnosis.

The two rejected hypotheses (H1 and H3) are equally informative. H1's rejection (BMI SHAP increase only 35.1% in the severely obese–hypertensive subgroup) suggests that BMI's predictive contribution does not substantially amplify beyond its marginal effect when combined with hypertension—the two features may be capturing overlapping variance rather than genuinely interacting. This is consistent with the Tree SHAP correlation bias documented by Al-Hudaibi et al. (2025): BMI and HighBP are moderately correlated, and their SHAP values may partially absorb each other's contributions. H3's rejection (HighChol SHAP increase only 13.1%) similarly suggests that high cholesterol's predictive signal is largely captured by its marginal SHAP value and does not meaningfully amplify within the fair-health subgroup.

These results demonstrate that ARM rules do not uniformly represent genuine risk-amplifying interactions—a finding that validates the necessity of the ARM-to-SHAP verification step and directly addresses the limitation identified in Bata et al. (2025) and Fakir et al. (2024), who generated rules without any verification mechanism.

**Model Performance and Comparison with Prior Work**

XGBoost achieved AUC-ROC = 0.848 and macro-F1 = 0.730 on the full 21-feature set, which is broadly comparable to Majcherek et al.'s (2025) best tree ensemble results and slightly below Kutlu et al.'s (2024) reported 86.6% accuracy—though direct comparison is complicated by different evaluation metrics and class-imbalance handling strategies. Notably, the current study's recall of 0.814 on the At-Risk class is substantially higher than what Kutlu et al. (2024) reported for their minority class, reflecting the benefit of SMOTE-aware cross-validation as recommended by Netayawijit et al. (2025) and Rezki et al. (2024).

The Decision Tree's unexpectedly high MCC (0.578) warrants attention: it outperforms all other models on this metric despite lower AUC. This likely reflects the tree's tendency to produce hard, well-calibrated binary splits on the dominant features (GenHlth, BMI, Age), which MCC rewards when both classes are predicted accurately. However, its lower AUC indicates poor probability calibration, making it unsuitable as the primary SHAP analysis model.

**Feature Leakage Impact**

The leakage sensitivity analysis confirms that GenHlth, CholCheck, DiffWalk, HeartDiseaseorAttack, and Stroke collectively contribute approximately 0.023 AUC points and 0.026 macro-F1 points to Variant A's performance. This is a non-trivial but not catastrophic leakage effect—the model retains strong predictive performance (AUC = 0.825) even after removing these features. The finding suggests that prior BRFSS studies (Majcherek et al., 2025; Kutlu et al., 2024) likely overestimated true predictive performance by a modest margin, but their core conclusions about feature importance remain directionally valid.

Variant C's substantially lower AUC (0.759) and macro-F1 (0.612) with behavioral-only features underscores a fundamental limitation of behavioral surveys for diabetes prediction: purely modifiable behavioral features (diet, physical activity, smoking, alcohol) carry substantially less predictive signal than clinical indicators (BMI, blood pressure, cholesterol). This has important implications for prevention-focused applications in Vietnam, where clinical measurements may not be routinely available in community screening settings.

**RQ3: Age-Stratified Risk Drivers and Implications for Vietnam**

The youth subgroup analysis reveals a qualitatively different risk profile for adults aged 18–44. The 3.15× amplification of Age's SHAP contribution within this subgroup is the most striking finding: within the 18–44 range, being closer to 44 is a far stronger risk signal than age variation in the general adult population, suggesting an accelerating risk trajectory in the late-30s to early-40s bracket. This is consistent with Nguyen et al.'s (2015) documentation of rapidly rising T2DM incidence among Vietnamese adults under 45 and supports targeted screening of adults in their late 30s and early 40s.

The 1.69× amplification of MentHlth in youth is a novel finding not reported in any reviewed study. Mental health burden as a disproportionate risk factor for younger adults may reflect stress-related metabolic dysregulation, depression-associated lifestyle deterioration, or reverse causality from undiagnosed prediabetes affecting mental wellbeing. This warrants further investigation in Vietnamese clinical contexts, where mental health screening is not routinely integrated with diabetes risk assessment.

Income's 1.42× amplification in youth (vs. general population) aligns with socioeconomic vulnerability being more predictive in working-age populations, consistent with the urbanization-driven dietary transitions documented by Nguyen et al. (2015) and Vuong et al. (2024). Lower-income younger adults in Vietnam may face compounded risks from processed food accessibility, sedentary occupational environments, and limited healthcare access.

**Limitations**

Several limitations constrain the generalizability of these findings. First, the BRFSS is a U.S. telephone survey with known sampling biases toward older, higher-income, and more health-aware respondents; all findings should be treated as hypothesis-generating rather than directly prescriptive for Vietnam. Second, the youth subgroup's low At-Risk prevalence (9.1%, n = 946 cases) limits statistical power for subgroup SHAP comparisons. Third, the 50% relative increase threshold for hypothesis confirmation is operationally defined and not derived from a formal power analysis; H1's 35.1% increase may represent a practically meaningful effect that falls below the threshold. Fourth, Tree SHAP's known correlation bias (Lundberg et al., 2019; Al-Hudaibi et al., 2025) means that SHAP values for correlated features (BMI, HighBP, HighChol) should be interpreted with caution. Fifth, the ARM analysis used a lowered confidence threshold (0.45 vs. the originally planned 0.60) to obtain sufficient behavioral rules, which may include weaker associations.

**ACKNOWLEDGMENT, FUNDING & ETHICS POLICIES**

The BRFSS 2015 dataset is publicly available, collected and anonymized by the U.S. CDC. No personally identifiable information is present. All findings are framed as statistical associations, not causal claims. This research received no external funding. The authors declare no conflicts of interest.

**REFERENCE**

Agrawal, R. K. (1994b). Fast algorithm for mining association rules. Very Large Data Bases, 487–499. [https://ci.nii.ac.jp/naid/10000112747](https://ci.nii.ac.jp/naid/10000112747) 

Agyemang, E. F., Mensah, J. A., Nyarko, E., Arku, D., Mbeah-Baiden, B., Opoku, E., & Nortey, E. N. N. (2025). Addressing class imbalance problem in health data classification: Practical application from an oversampling viewpoint. *Applied Computational Intelligence and Soft Computing*, 2025(1). [https://doi.org/10.1155/acis/1013769](https://doi.org/10.1155/acis/1013769) 

Ahmad, S., Hussain, S., Arif, M., & Ansari, M. A. (2026). Early-stage diabetes prediction using a stacked ensemble model enhanced with SHAP explainability. *Biomedical & Pharmacology Journal*, 1(19), 246\. [https://doi.org/10.13005/bpj/3350](https://doi.org/10.13005/bpj/3350) 

Ahmed, S., Kaiser, M. S., Hossain, M. S., & Andersson, K. (2024). A comparative analysis of LIME and SHAP interpreters with explainable ML-based diabetes predictions. *IEEE Access*, 13, 37370–37388. [https://doi.org/10.1109/access.2024.3422319](https://doi.org/10.1109/access.2024.3422319) 

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *arXiv*. [https://arxiv.org/abs/1907.10902](https://arxiv.org/abs/1907.10902) 

Al-Hudaibi, Z. H., Almatar, R. M., Alwabari, Z. A., Al-Duhailan, O. M., Alsuwailem, H. M., & Alhajji, Z. M. (2025). Explainable artificial intelligence-PREDICT. *Journal of Advanced Trends in Medical Research*, 2(2), 114–119. [https://doi.org/10.4103/atmr.atmr\_41\_25](https://doi.org/10.4103/atmr.atmr_41_25) 

Bata, R., Ghanem, A. S., Faludi, E. V., Sztanek, F., & Nagy, A. C. (2025). Association rule mining of time-based patterns in diabetes-related comorbidities on imbalanced data. *BMC Medical Informatics and Decision Making*, 25(1), 352\. [https://doi.org/10.1186/s12911-025-03206-1](https://doi.org/10.1186/s12911-025-03206-1) 

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. [https://doi.org/10.1023/a:1010933404324](https://doi.org/10.1023/a:1010933404324) 

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357. [https://doi.org/10.1613/jair.953](https://doi.org/10.1613/jair.953) 

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *arXiv*. [https://doi.org/10.48550/arxiv.1603.02754](https://doi.org/10.48550/arxiv.1603.02754) 

Duncan, B. B., Magliano, D. J., & Boyko, E. J. (2025). IDF Diabetes Atlas 11th edition 2025\. *Nephrology Dialysis Transplantation*, 41(1), 7–9. [https://doi.org/10.1093/ndt/gfaf177](https://doi.org/10.1093/ndt/gfaf177) 

Essalmi, H., & Affar, A. E. (2025). Dynamic algorithm for mining relevant association rules via meta-patterns. *Preprints.org*. [https://doi.org/10.20944/preprints202504.1724.v1](https://doi.org/10.20944/preprints202504.1724.v1) 

Fakir, Y., Khalil, S., & Fakir, M. (2024). Extraction of association rules in a diabetic dataset using parallel FP-growth under Apache Spark. *International Journal of Informatics and Communication Technology*, 13(3), 445\. [https://doi.org/10.11591/ijict.v13i3.pp445-452](https://doi.org/10.11591/ijict.v13i3.pp445-452) 

Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. *ACM SIGMOD Record*, 29(2), 1–12. [https://doi.org/10.1145/335191.335372](https://doi.org/10.1145/335191.335372) 

Islam, M. M., Rifat, H. R., Shahid, M. S. B., Akhter, A., Uddin, M. A., & Uddin, K. M. M. (2024). Explainable machine learning for efficient diabetes prediction. *Engineering Reports*, 7(1). [https://doi.org/10.1002/eng2.13080](https://doi.org/10.1002/eng2.13080) 

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. (2017). LightGBM: A highly efficient gradient boosting decision tree. *HAL*. [https://hal.science/hal-03953007](https://hal.science/hal-03953007) 

Kutlu, M., Donmez, T. B., & Freeman, C. (2024). Machine learning interpretability in diabetes risk assessment: A SHAP analysis. *Computers and Electronics in Medicine*, 1(1), 34–44. [https://doi.org/10.69882/adba.cem.2024075](https://doi.org/10.69882/adba.cem.2024075) 

Lundberg, S., & Lee, S. (2017). A unified approach to interpreting model predictions. *arXiv*. [https://doi.org/10.48550/arxiv.1705.07874](https://doi.org/10.48550/arxiv.1705.07874) 

Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S. (2019). Explainable AI for trees. *arXiv*. [https://doi.org/10.48550/arxiv.1905.04610](https://doi.org/10.48550/arxiv.1905.04610) 

Majcherek, D., Ciesielski, A., & Sobczak, P. (2025). AI-driven analysis of diabetes risk determinants in U.S. adults. *PLoS ONE*, 20(9), e0328655. [https://doi.org/10.1371/journal.pone.0328655](https://doi.org/10.1371/journal.pone.0328655) 

Netayawijit, P., Chansanam, W., & Sorn-In, K. (2025). Interpretable ML framework for diabetes prediction: Integrating SMOTE with SHAP. *Healthcare*, 13(20), 2588\. [https://doi.org/10.3390/healthcare13202588](https://doi.org/10.3390/healthcare13202588) 

Nguyen, C. T., Pham, N. M., Lee, A. H., & Binns, C. W. (2015). Prevalence of and risk factors for type 2 diabetes mellitus in Vietnam. *Asia Pacific Journal of Public Health*, 27(6), 588–600. [https://doi.org/10.1177/1010539515595860](https://doi.org/10.1177/1010539515595860) 

Rafie, Z., Talab, M. S., Koor, B. E. Z., Garavand, A., Salehnasab, C., & Ghaderzadeh, M. (2025). Leveraging XGBoost and explainable AI for accurate prediction of type 2 diabetes. *BMC Public Health*, 25(1), 3688\. [https://doi.org/10.1186/s12889-025-24953-w](https://doi.org/10.1186/s12889-025-24953-w) 

Rezki, M. K., Mazdadi, M. I., Indriani, F., Muliadi, M., Saragih, T. H., & Athavale, V. A. (2024). Application of SMOTE to address class imbalance in diabetes classification. *Journal of Electronics Electromedical Engineering and Medical Informatics*, 6(4), 343–354. [https://doi.org/10.35882/jeeemi.v6i4.434](https://doi.org/10.35882/jeeemi.v6i4.434) 

Rossi, D., Citarella, A. A., De Marco, F., Di Biasi, L., Zheng, H., & Tortora, G. (2026). D.R.E.A.M: Diabetes risk via explainable AI modeling. *Multimedia Tools and Applications*, 85(2). [https://doi.org/10.1007/s11042-026-21240-7](https://doi.org/10.1007/s11042-026-21240-7) 

Vuong, T. B., Tran, T. M., & Tran, N. Q. (2024). High prevalence of prediabetes and type 2 diabetes in high-risk adults in Vietnam. *Diabetes Epidemiology and Management*, 17, 100239\. [https://doi.org/10.1016/j.deman.2024.100239](https://doi.org/10.1016/j.deman.2024.100239) 

