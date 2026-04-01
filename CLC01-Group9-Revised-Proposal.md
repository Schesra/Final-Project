# **Mining Behavioral Risk Patterns for Diabetes Prediction: An Explainable Ensemble Approach**
**CLC01 - GROUP 9**

## **1. Problem Statement**

### **1.1 Research Objective**
This project addresses the critical gap between diabetes prediction accuracy and clinical interpretability. While machine learning models can achieve high predictive performance, healthcare practitioners need **actionable behavioral insights** to guide patient interventions.

### **1.2 Core Research Questions**
**RQ1 (Pattern Discovery)**: Which combinations of behavioral factors form high-risk patterns for diabetes? 
- Method: Association Rule Mining (FP-Growth)
- Output: Interpretable rules like {HighBP=1, BMI>30, PhysActivity=0} → Diabetes (confidence=0.75)

**RQ2 (Predictive Modeling)**: Can ensemble methods achieve clinical-grade accuracy while maintaining explainability?
- Method: Ensemble Classification + SHAP Analysis  
- Output: Risk probabilities + feature importance explanations

**RQ3 (Cross-Validation)**: Do ARM-discovered patterns align with SHAP feature importance rankings?
- Method: Comparative analysis between rule antecedents and SHAP top features
- Output: Consistency validation between interpretability methods

### **1.3 Clinical Relevance**
Vietnam has 2.5 million diabetes patients with rising prevalence in adults under 45. Current clinical guidelines lack specific behavioral thresholds. This project aims to provide **quantified risk combinations** that practitioners can use for early intervention.

## **2. Dataset & Preprocessing**

### **2.1 Dataset Selection: BRFSS 2015**
- **Source**: CDC Behavioral Risk Factor Surveillance System (data.csv)
- **Scale**: 269,131 records, 22 features (21 behavioral/biological + 1 target)
- **Rationale**: Large scale required for ARM; rich behavioral features for pattern discovery

### **2.2 Target Variable Handling**
**Critical Decision**: Original target has 3 classes (0=No diabetes, 1=Pre-diabetes, 2=Diabetes)

**Binarization Strategy**:
- Class 0 → "No Risk" (194,377 records, 72.2%)  
- Classes 1+2 → "At Risk" (74,754 records, 27.8%)

**Clinical Justification**: Pre-diabetes and diabetes require identical behavioral interventions (diet modification, exercise increase, weight management), making binary classification clinically appropriate.

### **2.3 Class Imbalance Handling**
- **Strategy**: SMOTE applied only within CV training folds
- **Rationale**: Prevents data leakage while addressing 72:28 imbalance
- **Validation**: Compare SMOTE vs class_weight vs NearMiss on macro F1

### **2.4 Data Leakage Risk Analysis**
**High-Risk Features** (potentially symptomatic):
- `GenHlth` (self-rated health): May reflect undiagnosed symptoms
- `DiffWalk` (walking difficulty): Potential diabetic complication  
- `PhysHlth` (poor physical health days): May correlate with symptoms

**Mitigation Strategies**:
1. **Temporal Logic**: Only use features measurable BEFORE diagnosis
2. **Sensitivity Analysis**: Test model performance excluding high-risk features
3. **Clinical Review**: Each feature justified by medical literature

## **3. Methodology Justification**

### **3.1 Why Association Rule Mining?**
**Purpose**: Discover behavioral co-occurrence patterns invisible to correlation analysis
- **Algorithm**: FP-Growth (efficient for 269K records)
- **Output**: Rules like {HighBP=1, BMI_Obese=1, PhysActivity=0} → At_Risk
- **Clinical Value**: Provides specific behavioral thresholds for intervention

### **3.2 Why Ensemble Classification?**
**Purpose**: Maximize predictive accuracy for clinical decision support
- **Models**: XGBoost + LightGBM + Random Forest (Soft Voting)
- **Tuning**: Optuna Bayesian optimization (50 trials)
- **Clinical Value**: High recall (≥0.75) minimizes missed at-risk patients

### **3.3 Why SHAP Explainability?**
**Purpose**: Cross-validate ARM findings and provide model transparency
- **Method**: TreeExplainer on best single model
- **Output**: Global feature importance + individual prediction explanations
- **Clinical Value**: Builds practitioner trust through interpretable AI

### **3.4 Integration Logic**
```
ARM Rules → Behavioral Pattern Discovery
    ↓
Ensemble Model → Accurate Risk Prediction  
    ↓
SHAP Analysis → Feature Importance Validation
    ↓
Cross-Method Comparison → Consistency Check
```

## **4. Evaluation Plan**

### **4.1 Pattern Discovery Metrics (ARM)**
| Metric | Threshold | Clinical Interpretation |
|--------|-----------|------------------------|
| Support | ≥ 0.05 | Pattern affects ≥5% of population (~13,450 patients) |
| Confidence | ≥ 0.60 | When pattern present, diabetes risk >60% |
| Lift | ≥ 1.5 | Pattern is 1.5× more likely than random chance |

**Target Output**: 15-20 high-quality rules with clinical interpretations

### **4.2 Prediction Metrics (Ensemble)**
| Metric | Target | Priority | Rationale |
|--------|--------|----------|-----------|
| Recall | ≥ 0.75 | PRIMARY | Missing at-risk patient has higher cost than false alarm |
| F1-macro | ≥ 0.70 | PRIMARY | Accounts for class imbalance |
| AUC-ROC | ≥ 0.82 | SECONDARY | Overall discriminative ability |
| MCC | ≥ 0.45 | SECONDARY | Balanced metric for imbalanced data |

### **4.3 Cross-Method Validation**
**ARM-SHAP Consistency Analysis**:
- Extract features from top 15 ARM rules
- Compare with top 10 SHAP global importance
- Compute Jaccard similarity coefficient
- **Expected**: 60-80% overlap indicates method consistency

## **5. Risk Analysis & Limitations**

### **5.1 Technical Risks**
| Risk | Mitigation |
|------|------------|
| ARM generates >500 rules | Raise min_support to 0.10, min_lift to 2.0 |
| SMOTE overfitting | Switch to class_weight='balanced' |
| Hyperparameter tuning timeout | Reduce Optuna trials to 20 with pruning |

### **5.2 Methodological Limitations**
- **Correlation ≠ Causation**: All findings represent statistical associations, not causal relationships
- **Self-Reported Bias**: BRFSS data subject to underreporting of negative behaviors
- **Population Scope**: Results apply to US adults (2015); generalization to other populations requires validation
- **Temporal Snapshot**: Cross-sectional data cannot capture behavioral change over time

### **5.3 Clinical Scope Boundaries**
- **Decision Support Only**: All outputs assist, not replace, clinical judgment
- **No Automated Diagnosis**: Models provide risk assessment, not diagnostic conclusions
- **Practitioner Validation Required**: All behavioral thresholds need clinical review before implementation

---

**Expected Timeline**: 4 weeks
**Deliverables**: 
1. Jupyter notebook with complete pipeline
2. Top 15 behavioral risk rules with clinical interpretations  
3. Ensemble model with SHAP explanations
4. Cross-method validation analysis
5. Final presentation (10-15 slides)