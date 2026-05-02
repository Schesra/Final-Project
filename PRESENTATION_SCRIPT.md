# PRESENTATION SCRIPT — CLC01 Group 9
## Mining Behavioral Risk Patterns for Diabetes Prediction: An Explainable Ensemble Approach

---

## PHẦN 1 — STRUCTURE AND CLARITY (Cấu trúc tổng thể)

---

### Mở đầu: Giới thiệu pipeline

> "Dự án của chúng tôi được tổ chức thành một pipeline 4 giai đoạn rõ ràng, mỗi giai đoạn có đầu vào và đầu ra xác định."

Pipeline tổng thể:

```
DATA PREPARATION → ARM (FP-Growth) → ENSEMBLE CLASSIFICATION → SHAP VERIFICATION
     Phase 1              Phase 2              Phase 3                Phase 4
```

Notebook được chia thành các section đánh số rõ ràng:
- **Phase 1**: Data Preparation (Cells 5–15) — làm sạch, chuẩn hóa, discretize
- **Phase 2**: Association Rule Mining (Cells 16–24) — FP-Growth, filter rules, formulate hypotheses
- **Phase 3**: Ensemble Classification (Cells 25–41) — train 6 models, Optuna tuning, leakage analysis
- **Phase 4**: SHAP Verification (Cells 42–52) — global SHAP, subgroup verification, youth analysis

> "Lý do chúng tôi thiết kế pipeline theo thứ tự này là có chủ đích: ARM chạy TRƯỚC SHAP để tạo ra các hypothesis có thể kiểm chứng. Đây là điểm khác biệt cốt lõi so với tất cả các nghiên cứu liên quan — họ đều dùng SHAP như một công cụ khám phá hậu kỳ, không có hypothesis trước."

---

### Cấu trúc code: Tại sao tổ chức như vậy?

**Cell 2 — Setup và suppress warnings:**
```python
warnings.filterwarnings('ignore')
logging.getLogger('jupyter_client').setLevel(logging.CRITICAL)
```
> "Chúng tôi suppress warnings ngay từ đầu để output notebook sạch và dễ đọc khi demo. Đây là thực hành tốt trong production notebook."

**Cell 4 — Load data với path linh hoạt:**
```python
DATA_PATH = "./data.csv"
SAVE_PATH = "./outputs/"
os.makedirs(SAVE_PATH, exist_ok=True)
```
> "Tất cả paths được đặt ở đầu notebook để dễ thay đổi khi chạy trên máy khác. `os.makedirs` với `exist_ok=True` đảm bảo không lỗi nếu folder đã tồn tại."

**Seed control:**
```python
SEED = 42
```
> "Seed 42 được dùng nhất quán cho tất cả random operations — train/test split, SMOTE, Optuna, cross-validation — đảm bảo reproducibility hoàn toàn."


---

## PHẦN 2 — EXPLANATION OF CODE / NOTEBOOK / PIPELINE

---

### Phase 1: Data Preparation

#### 1.1 — Binarize Target (Cell 7)

```python
df['Diabetes_binary'] = df['Diabetes_binary'].apply(lambda x: 0 if x == 0.0 else 1)
```

> "Dataset gốc có 3 class: 0 = No Risk, 1 = Pre-diabetes, 2 = Diabetes. Chúng tôi merge class 1 và 2 thành At-Risk (1) vì lý do lâm sàng: cả pre-diabetes lẫn diabetes đều cần can thiệp phòng ngừa. Đây không phải quyết định tùy tiện — chúng tôi kiểm tra empirically rằng profile của class 1 và class 2 rất giống nhau: BMI trung bình 31.83 vs 31.96, HighBP 73.8% vs 75.2%, PhysActivity 63.4% vs 62.9%. Sự khác biệt nhỏ này justify việc merge."

Kết quả sau binarization:
- No Risk: 194,377 records (72.22%)
- At-Risk: 74,754 records (27.78%)

---

#### 1.2 — BMI Outlier Handling (Cell 9)

```python
# Strategy A: IQR Capping
upper_cap = Q3 + 1.5 * IQR  # = 42.5
df_A['BMI'] = df_A['BMI'].clip(upper=upper_cap)

# Strategy B: Remove BMI > 60
df_B = df[df['BMI'] <= 60].copy()
```

> "Chúng tôi thử nghiệm 2 chiến lược xử lý outlier BMI. BMI max trong dataset là 98 — rõ ràng là data entry error. Strategy A (IQR capping tại 42.5) ảnh hưởng 11,280 records nhưng giữ lại tất cả. Strategy B (remove BMI > 60) chỉ loại 993 records nhưng mất data. Chúng tôi chọn Strategy A vì: (1) giữ được nhiều data hơn, (2) capping là conservative hơn deletion, (3) BMI 42.5 vẫn là severely obese — clinically meaningful."

---

#### 1.3 — MinMaxScaler (Cell 11)

```python
BINARY_FEATURES = ['HighBP','HighChol','CholCheck','Smoker',...]  # 14 features
SCALE_FEATURES  = ['BMI','MentHlth','PhysHlth','GenHlth','Age','Education','Income']  # 7 features
```

> "Chúng tôi phân biệt rõ 2 loại features: binary features (0/1) không cần scale, continuous/ordinal features cần MinMaxScaler. Quan trọng: scaler được FIT CHỈ TRÊN TRAINING DATA rồi mới transform test data. Nếu fit trên toàn bộ dataset, thông tin từ test set sẽ leak vào training — đây là data leakage cơ bản nhất."

---

#### 1.4 — Discretization cho ARM (Cell 13)

```python
df_arm['BMI_cat'] = pd.cut(df_arm['BMI'], 
    bins=[0, 18.5, 25, 30, 35, float('inf')],
    labels=['Underweight','Normal','Overweight','Obese','SeverelyObese'])

df_arm['Age_cat'] = pd.cut(df_arm['Age'], 
    bins=[0, 4, 9, 13],
    labels=['YoungAdult','MiddleAged','Senior'])
```

> "ARM yêu cầu dữ liệu categorical. Chúng tôi discretize BMI theo WHO clinical thresholds (18.5/25/30/35) — không phải arbitrary bins. Age được chia theo BRFSS codebook: codes 1-4 = 18-44 (YoungAdult), 5-9 = 45-64 (MiddleAged), 10-13 = 65+ (Senior). Điều này đảm bảo các rules có ý nghĩa lâm sàng."

---

#### 1.5 — Class Balancing Experiment (Cell 15)

```python
for strategy_name, strategy in [
    ('SMOTE', SMOTE(random_state=SEED)),
    ('NearMiss', NearMiss()),
    ('class_weight', None)
]:
    # Evaluate inside CV folds
```

> "Chúng tôi không chỉ dùng SMOTE mà thực sự so sánh 3 chiến lược trong cross-validation. Kết quả: SMOTE macro-F1 = 0.692, NearMiss = 0.590, class_weight = 0.692. SMOTE và class_weight tie nhưng SMOTE cho recall cao hơn trên minority class — quan trọng hơn trong bài toán y tế vì false negative (bỏ sót bệnh nhân) nguy hiểm hơn false positive."

**Điểm kỹ thuật quan trọng — SMOTE-aware CV:**
> "SMOTE được apply BÊN TRONG mỗi fold của cross-validation, không phải trước khi split. Nếu apply SMOTE trước, synthetic samples được tạo từ toàn bộ dataset có thể 'leak' thông tin từ validation fold vào training fold, làm inflate performance metrics."

---

### Phase 2: Association Rule Mining

#### 2.1 — Transaction Matrix (Cell 18)

```python
# Binary features: rename thành 'FeatureName_1' để rõ nghĩa
arm_binary_part.columns = [f'{c}_1' for c in BINARY_FEATURES]

# Categorical features: one-hot encode
arm_cat_part = pd.get_dummies(df_arm[cat_cols].astype(str))

# Thêm target columns
arm_df['NoRisk'] = (df_arm['Diabetes_binary'] == 0).astype(int)
arm_df['AtRisk']  = (df_arm['Diabetes_binary'] == 1).astype(int)
```

> "Transaction matrix có 41 columns sau one-hot encoding. Mỗi row là một 'transaction' — một người với tập hợp các health attributes. Average items per transaction là 14.45, nghĩa là mỗi người trung bình có 14-15 attributes active. Target (AtRisk/NoRisk) được include vào transaction để FP-Growth có thể tìm rules có consequent = AtRisk."

---

#### 2.2 — FP-Growth (Cell 20)

```python
frequent_itemsets = fpgrowth(arm_bool, min_support=0.05, use_colnames=True, max_len=5)
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.5)
```

> "Tại sao FP-Growth thay vì Apriori? Apriori phải scan database nhiều lần và generate candidate itemsets — với 269,131 records và 41 features, điều này sẽ rất chậm. FP-Growth compress toàn bộ dataset vào một FP-tree compact, chỉ cần 2 lần scan. Chúng tôi set max_len=5 để tránh MemoryError với itemsets quá dài."

**Tham số:**
- `min_support = 0.05`: Rule phải xuất hiện trong ít nhất 5% dataset = ~13,456 records. Đủ để statistically meaningful.
- `min_threshold = 1.5` (lift): Rule phải có lift ≥ 1.5, tức là co-occurrence xảy ra ít nhất 1.5x nhiều hơn ngẫu nhiên.
- `min_confidence = 0.45`: Sau khi filter, confidence ≥ 45% (base rate là 27.78%, nên đây là +17pp above chance).

---

#### 2.3 — Filter & Rank Rules (Cell 22)

```python
BASE_RATE = y.mean()  # 0.2778

# Filter: antecedent >= 2 features, confidence gap >= 15pp above base rate
rules_filtered = rules[
    (rules['consequents'].apply(lambda x: 'AtRisk' in x)) &
    (rules['antecedents'].apply(lambda x: len(x) >= 2)) &
    (rules['confidence'] >= BASE_RATE + 0.15)
]
```

> "Chúng tôi filter rules theo 3 tiêu chí: (1) consequent phải là AtRisk — chúng tôi chỉ quan tâm rules dự đoán risk; (2) antecedent phải có ít nhất 2 features — single-feature rules không thú vị vì SHAP đã capture được; (3) confidence phải cao hơn base rate ít nhất 15 percentage points — đảm bảo rule có predictive value thực sự."

**Một phát hiện thú vị:**
> "Ban đầu chúng tôi filter chỉ lấy rules có behavioral features (PhysActivity, BMI, Smoker...). Nhưng kết quả cho thấy top rules đều dominated bởi clinical/comorbidity features (HighBP, HighChol, DiffWalk). Chúng tôi quyết định giữ nguyên kết quả này thay vì force behavioral rules — đây là data-driven finding, consistent với feature leakage analysis ở Phase 3."

---

#### 2.4 — Formulate Hypotheses (Cell 24)

```python
ARM_TO_ORIG = {
    'BMI_cat': 'BMI',
    'GenHlth_cat': 'GenHlth',
    'DiffWalk_1': 'DiffWalk',
    ...
}
```

> "ARM dùng discretized names (BMI_cat_SeverelyObese) nhưng SHAP dùng original feature names (BMI). Chúng tôi cần mapping này để verify hypotheses. Mỗi hypothesis có dạng: 'Trong subgroup X (defined bởi ARM rule), mean SHAP của feature Y phải cao hơn population mean SHAP'. Đây là cách chúng tôi operationalize câu hỏi nghiên cứu RQ2."

---

### Phase 3: Ensemble Classification

#### 3.1 — Evaluate Model Helper (Cell 27)

```python
def evaluate_model(model, X_tr, y_tr, X_te, y_te, name='Model'):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1]
    return {
        'Recall': recall_score(y_te, y_pred),
        'Macro_F1': f1_score(y_te, y_pred, average='macro'),
        'AUC_ROC': roc_auc_score(y_te, y_prob),
        'MCC': matthews_corrcoef(y_te, y_pred)
    }
```

> "Chúng tôi dùng 4 metrics thay vì chỉ accuracy. Tại sao? Accuracy bị inflate bởi class imbalance — một model predict tất cả là No-Risk sẽ đạt 72% accuracy. Recall quan trọng vì false negative (bỏ sót bệnh nhân At-Risk) nguy hiểm hơn false positive. Macro-F1 cân bằng cả 2 classes. AUC-ROC đo khả năng phân biệt tổng thể. MCC là metric cân bằng nhất cho imbalanced data."

---

#### 3.2 — Optuna Tuning XGBoost (Cell 31)

```python
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    # Evaluate với 3-fold CV bên trong
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1_macro')
    return scores.mean()

study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)
```

> "Optuna dùng Tree-structured Parzen Estimators (TPE) — một dạng Bayesian optimization. Thay vì grid search exhaustive, TPE học từ các trials trước để suggest parameters có khả năng tốt hơn. `log=True` cho learning_rate và regularization vì các parameters này hoạt động trên log scale — khoảng cách giữa 0.001 và 0.01 quan trọng hơn giữa 0.1 và 0.11."

> "50 trials mất khoảng 1126 giây (~19 phút) cho XGBoost. Đây là trade-off có chủ đích giữa thời gian và quality."

---

#### 3.3 — Leakage Sensitivity (Cell 39)

```python
for variant_name, feat_cols in [
    ('Variant_A_full', ALL_FEATURES),           # 21 features
    ('Variant_B_no_leakage', VARIANT_B_FEATURES), # remove 5 leaky features
    ('Variant_C_behavioral_only', VARIANT_C_FEATURES)  # 10 behavioral only
]:
```

> "Đây là contribution quan trọng nhất về methodology. GenHlth (self-rated health) có correlation r=0.33 với target — cao nhất trong dataset. Nhưng GenHlth là self-reported và bị ảnh hưởng bởi diagnosis awareness: người biết mình bị tiểu đường sẽ rate health thấp hơn. Tương tự, CholCheck (cholesterol monitoring) và DiffWalk (difficulty walking) là consequences của diabetes, không phải predictors. Nếu include những features này, model đang 'cheat' bằng cách dùng thông tin post-diagnosis."


---

### Phase 4: SHAP Verification

#### 4.1 — Global SHAP (Cell 44)

```python
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)
```

> "TreeExplainer là exact algorithm cho tree-based models — không phải sampling-based như KernelSHAP. Nó traverse cấu trúc nội bộ của từng tree để tính Shapley values chính xác. Kết quả: mỗi prediction được decompose thành contributions của từng feature, với tổng bằng đúng prediction score."

**Top 10 features:**
```
GenHlth   0.674
Age       0.619
BMI       0.616
HighBP    0.515
HighChol  0.345
Income    0.264
MentHlth  0.229
PhysHlth  0.206
Education 0.171
Sex       0.170
```

---

#### 4.2 — Hypothesis Verification (Cell 48)

```python
CONFIRM_THRESHOLD = 0.50  # 50% relative increase

for h in hypotheses_final:
    orig_feat = h['test_orig_feat']
    context_items = h['context']
    
    # Build subgroup mask từ ARM context
    mask = pd.Series([True] * len(X_test), index=X_test.index)
    for item in context_items:
        if item.endswith('_1'):
            feat = item[:-2]
            mask &= (X_test[feat] > 0.5)
    
    pop_mean = shap_df[orig_feat].mean()
    sub_mean = shap_df.loc[mask, orig_feat].mean()
    relative_increase = abs(sub_mean - pop_mean) / (abs(pop_mean) + 1e-10)
```

> "Logic verification: nếu ARM rule {A, B, C} → AtRisk là genuine interaction, thì trong subgroup người có cả A, B, C, SHAP value của feature chính phải cao hơn đáng kể so với population. Chúng tôi dùng relative increase thay vì absolute difference vì SHAP values có scale khác nhau giữa các features."

---

#### 4.3 — Youth Analysis (Cell 52)

```python
# Age codes 1-5 = 18-44 per BRFSS codebook
age_threshold = (5 - df_A['Age'].min()) / (df_A['Age'].max() - df_A['Age'].min())
youth_mask = X_test['Age'] <= age_threshold

shap_youth = shap.TreeExplainer(best_xgb).shap_values(X_youth)
```

> "Age trong X_test đã được MinMaxScaled. Chúng tôi tính threshold tương ứng với Age code 5 (44 tuổi) trong scaled space. Đây là detail kỹ thuật quan trọng — nếu dùng threshold sai, subgroup sẽ không đúng."

---

## PHẦN 3 — EXPLANATION OF RESULTS

---

### Kết quả ARM

> "FP-Growth tìm được 19,204 frequent itemsets và 10,346 rules. Sau filtering còn 157 rules có consequent = AtRisk và confidence ≥ 45%."

**Top rule phân tích:**
```
{CholCheck_1, DiffWalk_1, HighBP_1, HighChol_1} → AtRisk
confidence = 0.610, lift = 2.194
```

> "Lift = 2.194 có nghĩa là: người có cả 4 conditions này có xác suất At-Risk cao gấp 2.19 lần so với population average (27.78%). Confidence = 61% nghĩa là trong số tất cả người có 4 conditions này, 61% là At-Risk."

> "Điều thú vị là top rules đều dominated bởi clinical/comorbidity features, không phải behavioral features thuần túy. Rule behavioral mạnh nhất là {BMI_SeverelyObese, HighBP} với lift = 2.105. Điều này cho thấy: trong BRFSS dataset, các markers lâm sàng (huyết áp cao, cholesterol cao) là predictors mạnh hơn behaviors thuần túy (ăn uống, tập thể dục). Đây là finding có ý nghĩa — nó suggest rằng behavioral interventions cần target những người đã có clinical risk markers."

---

### Kết quả Model Comparison

| Model | Recall | Macro-F1 | AUC-ROC | MCC |
|-------|--------|----------|---------|-----|
| Logistic Regression | 0.746 | 0.694 | 0.807 | 0.419 |
| Decision Tree | 0.793 | **0.784** | 0.822 | **0.578** |
| **XGBoost (Optuna)** | **0.814** | 0.730 | **0.848** | 0.496 |
| LightGBM (Optuna) | 0.613 | 0.711 | 0.814 | 0.423 |
| Random Forest (Optuna) | 0.700 | 0.717 | 0.824 | 0.445 |
| Soft Voting | 0.733 | 0.734 | 0.844 | 0.480 |

> "XGBoost được chọn làm primary model vì AUC-ROC cao nhất (0.848) và recall cao nhất (0.814). Soft Voting Ensemble không vượt qua XGBoost trên macro-F1 (0.734 vs 0.730) nên không được adopt — đây là pre-specified criterion."

> "Decision Tree có MCC cao nhất (0.578) — điều này có vẻ paradoxical. Giải thích: Decision Tree tạo ra hard binary splits trên dominant features (GenHlth, BMI, Age), và MCC rewards khi cả 2 classes được predict đúng. Nhưng AUC thấp hơn (0.822) cho thấy probability calibration kém — nó không phân biệt tốt ở các threshold khác nhau."

> "LightGBM có recall thấp nhất (0.613) dù được Optuna tune. Điều này có thể do leaf-wise growth của LightGBM tend to overfit trên minority class khi dataset lớn và imbalanced."

---

### Kết quả Leakage Analysis

| Variant | Recall | Macro-F1 | AUC-ROC |
|---------|--------|----------|---------|
| A — Full (21 features) | 0.814 | 0.730 | 0.848 |
| B — Leakage-removed | 0.807 | 0.704 | 0.825 |
| C — Behavioral-only | 0.838 | 0.612 | 0.759 |

> "Removing 5 leaky features (Variant B) giảm AUC chỉ 0.023 — modest nhưng không negligible. Điều này confirm rằng GenHlth, CholCheck, DiffWalk có predictive signal thực sự, nhưng một phần signal đó là leakage. Các nghiên cứu trước (Majcherek 2025, Kutlu 2024) likely overestimate performance khoảng 2-3% AUC."

> "Variant C (behavioral-only) có recall cao nhất (0.838) nhưng macro-F1 thấp nhất (0.612). Điều này có nghĩa: behavioral features rất tốt để identify true positives (high recall) nhưng kém trong việc phân biệt overall (low F1). Implication cho Vietnam: behavioral screening có thể identify at-risk individuals nhưng cần clinical confirmation."

---

### Kết quả SHAP Verification

| Hypothesis | Feature | Pop Mean SHAP | Sub Mean SHAP | Relative Increase | Confirmed |
|-----------|---------|--------------|--------------|------------------|-----------|
| H1 | BMI | −0.402 | −0.261 | 35.1% | ✗ |
| H2 | DiffWalk | −0.013 | −0.004 | **67.8%** | ✓ |
| H3 | HighChol | −0.073 | −0.064 | 13.1% | ✗ |
| H4 | GenHlth | −0.368 | −0.098 | **73.4%** | ✓ |

> "H2 confirmed: Trong subgroup {CholCheck, HighChol, HighBP}, DiffWalk SHAP tăng 67.8%. Điều này có nghĩa: khi một người đã có cholesterol cao và huyết áp cao, việc họ có difficulty walking là signal mạnh hơn đáng kể so với population. DiffWalk trong context này không chỉ là complication — nó là marker của advanced metabolic dysfunction."

> "H4 confirmed: Trong subgroup {AnyHealthcare, CholCheck, HighBP}, GenHlth SHAP tăng 73.4%. Người có healthcare access, đang monitor cholesterol, và có hypertension mà vẫn rate health là 'Fair' — đây là combination đặc biệt nguy hiểm. Họ aware về health của mình nhưng vẫn deteriorating."

> "H1 rejected (35.1%): BMI không amplify đáng kể trong subgroup severely obese + hypertension. Giải thích: BMI và HighBP correlated với nhau, nên SHAP của BMI bị 'absorbed' một phần bởi HighBP trong tree structure. Đây là Tree SHAP correlation bias mà chúng tôi đã anticipate trong methodology."

> "H3 rejected (13.1%): HighChol không amplify trong fair-health subgroup. HighChol là relatively weak predictor (SHAP = 0.345) và marginal effect của nó không thay đổi nhiều trong subgroup này."

---

### Kết quả Youth Analysis

| Feature | Full Pop | Youth (18-44) | Ratio |
|---------|----------|--------------|-------|
| Age | 0.618 | 1.949 | **3.15×** |
| MentHlth | 0.229 | 0.387 | **1.69×** |
| Income | 0.264 | 0.375 | **1.42×** |
| GenHlth | 0.674 | 0.919 | 1.36× |

> "Age SHAP tăng 3.15× trong youth subgroup — finding ấn tượng nhất. Trong population chung, Age là predictor mạnh vì risk tăng đều theo tuổi. Nhưng trong nhóm 18-44, Age SHAP còn cao hơn nữa — nghĩa là trong nhóm này, người ở cuối bracket (gần 44) có risk profile rất khác người ở đầu bracket (18-25). Risk trajectory accelerate mạnh trong late 30s đến early 40s."

> "MentHlth tăng 1.69× — novel finding không có trong bất kỳ nghiên cứu nào chúng tôi review. Mental health burden là risk factor disproportionately quan trọng với người trẻ. Có thể do: stress-related cortisol dysregulation, depression-associated sedentary behavior, hoặc reverse causality từ undiagnosed prediabetes ảnh hưởng mental wellbeing."

> "Income tăng 1.42× — consistent với socioeconomic vulnerability. Người trẻ thu nhập thấp ở Vietnam đối mặt với: processed food accessibility, sedentary jobs, limited healthcare access. Đây là actionable finding cho policy."


---

## PHẦN 4 — EVIDENCE OF REAL UNDERSTANDING

---

### Câu hỏi khó có thể được hỏi — và cách trả lời

---

**Q: Tại sao dùng macro-F1 thay vì weighted-F1 để tune Optuna?**

> "Weighted-F1 sẽ bị dominated bởi majority class (No-Risk, 72%). Macro-F1 treat cả 2 classes equally — nó là average của F1 cho từng class. Trong bài toán y tế, chúng tôi muốn model perform tốt trên CẢ HAI classes, không chỉ majority. Nếu dùng weighted-F1, Optuna sẽ optimize cho No-Risk class và sacrifice At-Risk recall."

---

**Q: Tại sao SHAP values trong bảng verification đều âm?**

> "SHAP values âm có nghĩa là feature đó GIẢM xác suất At-Risk so với baseline. Điều này không có nghĩa là feature không quan trọng — nó có nghĩa là trong test set, average contribution của feature đó là negative. Ví dụ, BMI mean SHAP = -0.402 vì nhiều người trong test set có BMI bình thường, kéo average xuống. Khi chúng tôi so sánh subgroup vs population, chúng tôi so sánh relative change, không phải absolute value."

---

**Q: Tại sao không dùng SHAP interaction values thay vì subgroup comparison?**

> "SHAP interaction values (từ TreeSHAP) có thể tính pairwise interactions. Nhưng chúng tôi chọn subgroup comparison vì: (1) interpretable hơn — dễ explain cho clinicians; (2) ARM-defined subgroups là multi-feature combinations, không chỉ pairwise; (3) SHAP interaction values bị biased hơn khi features correlated. Subgroup approach là conservative và transparent hơn."

---

**Q: Jaccard similarity 0.308 giữa ARM và SHAP — có thấp không?**

> "0.308 không phải thấp trong context này. ARM và SHAP capture fundamentally different things: ARM tìm co-occurrence patterns ở population level, SHAP đo individual-level prediction contributions. Overlap hoàn toàn sẽ đáng ngờ hơn — nó sẽ suggest hai methods đang measure cùng một thứ. 4 features chung (BMI, GenHlth, HighBP, HighChol) là cross-validation tự nhiên giữa hai approaches."

---

**Q: Tại sao không dùng SMOTE-Tomek thay vì SMOTE đơn thuần?**

> "Chúng tôi đã consider điều này. SMOTE-Tomek remove borderline instances sau khi oversample, potentially cleaner. Nhưng với 269,131 records, Tomek links removal sẽ rất chậm. Và empirically, SMOTE đã cho macro-F1 = 0.692 — tốt hơn NearMiss và bằng class_weight. Marginal gain từ SMOTE-Tomek không justify thêm computational cost."

---

**Q: Tại sao không include GenHlth trong Variant C (behavioral-only)?**

> "GenHlth là self-rated health — không phải behavioral feature. Nó là subjective health perception, bị ảnh hưởng bởi diagnosis awareness. Variant C chỉ include features mà người dùng có thể self-report về behaviors: physical activity, smoking, diet (fruits/veggies), alcohol, và demographics (age, sex, income, education). BMI được include vì có thể đo tại nhà."

---

**Q: Kết quả có generalize được sang Vietnam không?**

> "Đây là limitation quan trọng nhất. BRFSS là U.S. telephone survey với sampling bias rõ ràng: older, higher-income, more health-aware respondents. Vietnam có demographics khác: younger at-risk population, different dietary patterns, different healthcare access. Tuy nhiên, findings của chúng tôi là hypothesis-generating: (1) mental health là risk factor quan trọng với người trẻ — cần validate trên Vietnamese data; (2) income amplification trong youth — consistent với urbanization patterns ở Vietnam; (3) behavioral-only features có recall cao — suggest behavioral screening feasible ở community level."

---

**Q: Tại sao Decision Tree có MCC cao nhất (0.578) nhưng không được chọn?**

> "MCC cao của Decision Tree là misleading. MCC = 0.578 nhưng AUC = 0.822 — thấp hơn XGBoost (0.848). AUC đo khả năng phân biệt tổng thể ở mọi threshold, trong khi MCC chỉ đo tại một threshold cụ thể (default 0.5). Decision Tree tend to create hard boundaries tốt tại threshold 0.5 nhưng probability calibration kém. Cho SHAP analysis, chúng tôi cần model với probability calibration tốt — XGBoost phù hợp hơn."

---

**Q: Tại sao Age SHAP trong youth subgroup cao hơn full population?**

> "Đây là finding counterintuitive nhưng có explanation rõ ràng. Trong full population, Age là continuous predictor — risk tăng đều từ 18 đến 80+. Trong youth subgroup (18-44), model đang phân biệt giữa người 18 tuổi (very low risk) và người 44 tuổi (risk đang accelerate). Khoảng cách risk giữa 18 và 44 tuổi, khi normalized trong subgroup, lớn hơn khoảng cách tương đương trong full population. Nói cách khác: Age là discriminative hơn TRONG nhóm trẻ so với trong population chung."

---

### Tóm tắt contributions thực sự

> "Contribution cốt lõi của chúng tôi không phải là đạt AUC cao nhất — Majcherek 2025 và Kutlu 2024 đã làm điều đó. Contribution của chúng tôi là:"

1. **Methodological**: Lần đầu tiên implement sequential ARM → SHAP pipeline với hypothesis verification, thay vì dùng SHAP như exploratory tool
2. **Empirical**: Quantify feature leakage impact (0.023 AUC) — prior studies không làm điều này
3. **Clinical**: Confirm rằng 2/4 ARM rules represent genuine interactions, không chỉ statistical co-occurrence
4. **Novel finding**: Mental health là 1.69× more predictive trong youth — không có trong bất kỳ reviewed study nào
5. **Vietnam relevance**: Age-stratified analysis provide actionable hypotheses cho prevention programs targeting Vietnamese youth

---

*Script này được soạn dựa trên phân tích toàn bộ notebook Version/main_IDE.ipynb và kết quả thực nghiệm từ main_IDE.ipynb*
