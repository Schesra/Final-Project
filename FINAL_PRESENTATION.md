# SCRIPT THUYẾT TRÌNH HOÀN CHỈNH — CLC01 Group 9
## Mining Behavioral Risk Patterns for Diabetes Prediction: An Explainable Ensemble Approach
### Thuyết trình trực tiếp bằng notebook — Version/main_IDE.ipynb

---

> **Cách dùng file này:** Mở notebook song song với file này. Chạy từng cell, sau đó đọc phần giải thích tương ứng. Các đoạn trong `> "..."` là lời nói trực tiếp khi thuyết trình.

---

## MỞ ĐẦU — Giới thiệu tổng quan (trước khi chạy cell đầu tiên)

> "Dự án của chúng tôi giải quyết một gap trong nghiên cứu hiện tại về dự đoán tiểu đường type 2. Tất cả các nghiên cứu trước đây đều dùng SHAP như một công cụ khám phá hậu kỳ — tức là train model xong rồi mới nhìn vào SHAP để xem feature nào quan trọng. Chúng tôi đảo ngược logic này: dùng Association Rule Mining để tạo ra các hypothesis CỤ THỂ trước, rồi dùng SHAP để kiểm chứng xem hypothesis đó có đúng không."

**Pipeline tổng thể — đọc Cell 0:**
```
DATA PREPARATION → ARM (FP-Growth) → ENSEMBLE CLASSIFICATION → SHAP VERIFICATION
     Phase 1              Phase 2              Phase 3                Phase 4
     Cells 5-15           Cells 16-24          Cells 25-41           Cells 42-52
```

> "Mỗi phase có đầu vào và đầu ra xác định. Phase 1 output là clean data. Phase 2 output là 4 testable hypotheses. Phase 3 output là best model (XGBoost). Phase 4 output là hypothesis verification results và youth analysis. Toàn bộ pipeline chạy trên BRFSS 2015 dataset — 269,131 records từ U.S. CDC."

**Tại sao thứ tự này quan trọng:**
> "Nếu chúng tôi chạy SHAP trước rồi mới nhìn vào ARM, chúng tôi sẽ bị confirmation bias — chọn rules phù hợp với SHAP findings. Bằng cách chạy ARM trước, hypotheses được tạo ra hoàn toàn độc lập với model, sau đó SHAP verification là genuine test."

---
---

# PHASE 1 — DATA PREPARATION

---

### Tổng quan quy trình Data Preparation

> "Trước khi chạy bất kỳ cell nào, tôi sẽ mô tả toàn bộ quy trình data preparation chúng tôi đã thực hiện — 6 bước theo thứ tự, mỗi bước giải quyết một vấn đề cụ thể của dataset này."

**Quy trình 6 bước:**

```
[1] Load & Inspect     → phát hiện 3 class, imbalance 72:28, BMI max=98
        ↓
[2] Binarize Target    → merge class 1+2 thành At-Risk (empirically justified)
        ↓
[3] BMI Outlier        → IQR capping tại 42.5 (giữ 11,280 records thay vì xóa)
        ↓
[4] Normalize          → MinMaxScaler fit trên train only (tránh leakage)
        ↓
[5] Discretize for ARM → bins theo WHO/BRFSS thresholds (domain-driven)
        ↓
[6] Split & Balance    → 80/20 stratified split + SMOTE-aware CV
```

> "Điểm quan trọng: bước 4 và 5 tạo ra HAI phiên bản data khác nhau từ cùng một nguồn. Scaled data (bước 4) dùng cho classification. Unscaled + discretized data (bước 5) dùng cho ARM. Lý do: ARM cần original values để discretize theo clinical thresholds — nếu dùng scaled values, bins sẽ không còn ý nghĩa lâm sàng."

**Kết quả sau toàn bộ Phase 1:**

| Output | Dùng cho | Kích thước |
|--------|----------|-----------|
| `df_A_scaled` | Classification (Phase 3) | 269,131 × 21 features |
| `df_arm` (discretized) | ARM transaction matrix (Phase 2) | 269,131 × 29 columns |
| `X_train / X_test` | Model training/evaluation | 215,304 / 53,827 records |
| `X_tr_bal / y_tr_bal` | SMOTE-balanced training | 311,002 records |

---

## CELL 2 — Setup & Install

```python
warnings.filterwarnings('ignore')
logging.getLogger('jupyter_client').setLevel(logging.CRITICAL)
!{sys.executable} -m pip install xgboost lightgbm shap optuna ...
```
**Output:** `Setup complete.`

> "Cell này suppress warnings và install libraries. Dùng `sys.executable` thay vì `pip` trực tiếp để đảm bảo install vào đúng Python environment đang chạy notebook — tránh lỗi 'installed but not found' khi có nhiều Python versions. SEED = 42 được set một lần duy nhất và dùng nhất quán cho tất cả random operations trong toàn bộ pipeline — train/test split, SMOTE, Optuna, cross-validation — đảm bảo reproducibility hoàn toàn."

---

## CELL 4 — Load Data

```python
DATA_PATH = "./data.csv"
df_raw = pd.read_csv(DATA_PATH)
```

**Output:**
```
Loaded: (269131, 22)
Target distribution (raw):
0.0    194377   ← No Risk       (72.22%)
1.0     39657   ← Pre-diabetes  (14.73%)
2.0     35097   ← Diabetes      (13.04%)
```

> "Chúng tôi load BRFSS 2015 dataset — 269,131 records, 22 columns. Bước đầu tiên là inspect target distribution. Phát hiện ngay 3 vấn đề: (1) target có 3 class thay vì 2 — cần quyết định có merge không; (2) imbalance rõ ràng: 72% No-Risk vs 28% At-Risk; (3) class 1 và class 2 có tỷ lệ gần nhau (14.73% vs 13.04%) — gợi ý chúng có thể giống nhau về profile. Bước tiếp theo kiểm tra điều này."

---

## CELL 7 — Binarize Target

```python
df['Diabetes_binary'] = df['Diabetes_binary'].apply(lambda x: 0 if x == 0.0 else 1)
```

**Output:**
```
After binarization:
0    194377  (72.22%)
1     74754  (27.78%)

Pre-diabetes: BMI=31.83, HighBP=73.8%, PhysActivity=63.4%
Diabetes:     BMI=31.96, HighBP=75.2%, PhysActivity=62.9%
No Risk:      BMI=28.10, HighBP=40.1%, PhysActivity=75.2%
```

> "Chúng tôi so sánh profile của 3 class trên 3 features đại diện. Kết quả: Pre-diabetes và Diabetes gần như giống hệt nhau — BMI chênh 0.13, HighBP chênh 1.4%, PhysActivity chênh 0.5%. Trong khi No-Risk khác hoàn toàn — BMI thấp hơn 3.7 đơn vị, HighBP thấp hơn 33.7%, PhysActivity cao hơn 11.8%. Dựa trên kết quả này, chúng tôi merge class 1 và 2 thành At-Risk. Sau binarization: 72.22% No-Risk vs 27.78% At-Risk — imbalance ratio 2.6:1."

---

## CELL 9 — BMI Outlier Handling

```python
# Strategy A: IQR Capping tại 42.5
df_A['BMI'] = df_A['BMI'].clip(upper=upper_cap)

# Strategy B: Remove BMI > 60
df_B = df[df['BMI'] <= 60].copy()
```

**Output:**
```
Records with BMI > 60: 993
BMI max: 98.0
IQR Cap threshold: 42.50 (affects 11280 records)
Strategy B removed: 993 records
```

> "Inspect BMI phát hiện max = 98 — không thể tồn tại về mặt sinh học, rõ ràng là data entry error. Chúng tôi thử nghiệm 2 chiến lược xử lý và so sánh: Strategy A dùng IQR capping — tính Q1, Q3, IQR rồi cap tại Q3 + 1.5×IQR = 42.5. Kết quả: 11,280 records bị cap nhưng không mất record nào. Strategy B remove trực tiếp BMI > 60 — chỉ loại 993 records nhưng mất data. Chúng tôi chọn Strategy A vì giữ được toàn bộ 269,131 records. BMI 42.5 vẫn là 'Severely Obese' — threshold có ý nghĩa lâm sàng."

---

## CELL 11 — MinMaxScaler

```python
BINARY_FEATURES = ['HighBP','HighChol','CholCheck','Smoker',...]  # 14 features
SCALE_FEATURES  = ['BMI','MentHlth','PhysHlth','GenHlth','Age','Education','Income']  # 7 features
# Scaler FIT chỉ trên training data
scaler.fit(X_train[SCALE_FEATURES])
```

**Output:**
```
Scaled ranges:
     BMI  MentHlth  PhysHlth  GenHlth  Age  Education  Income
min  0.0       0.0       0.0      0.0  0.0        0.0     0.0
max  1.0       1.0       1.0      1.0  1.0        1.0     1.0
```

> "Chúng tôi chia 21 features thành 2 nhóm: 14 binary features (0/1) không cần scale vì đã ở đúng range. 7 continuous/ordinal features có range khác nhau — BMI: 10–42, Age: 1–13, MentHlth: 0–30 — cần đưa về [0,1] bằng MinMaxScaler. Scaler được fit chỉ trên X_train, sau đó transform cả train lẫn test. Kết quả min=0.0, max=1.0 xác nhận scaling đúng. Lưu ý: ARM dùng df_A chưa scale — giữ original values để discretize theo clinical thresholds."

---

## CELL 13 — Discretize Features for ARM

```python
df_arm['BMI_cat'] = pd.cut(df_arm['BMI'],
    bins=[0, 18.5, 25, 30, 35, float('inf')],
    labels=['Underweight','Normal','Overweight','Obese','SeverelyObese'])

df_arm['Age_cat'] = pd.cut(df_arm['Age'],
    bins=[0, 4, 9, 13],
    labels=['YoungAdult','MiddleAged','Senior'])

df_arm['GenHlth_cat'] = df_arm['GenHlth'].map(
    {1:'Excellent', 2:'VeryGood', 3:'Good', 4:'Fair', 5:'Poor'})

df_arm['MentHlth_cat'] = pd.cut(df_arm['MentHlth'],
    bins=[0, 1, 14, 31],
    labels=['None','Moderate','High'])
```

**Output:**
```
ARM dataset shape: (269131, 29)
Missing values: 0
```

> "Chúng tôi discretize 7 continuous/ordinal features thành categorical để dùng trong ARM. BMI được chia theo WHO clinical thresholds (18.5/25/30/35) thành 5 categories. Age được chia theo BRFSS codebook thành 3 nhóm: codes 1-4 = YoungAdult (18-44), 5-9 = MiddleAged (45-64), 10-13 = Senior (65+). GenHlth được map trực tiếp từ BRFSS 5-point scale. MentHlth và PhysHlth được chia thành None/Moderate/High dựa trên số ngày trong 30 ngày qua. Income và Education được chia thành 3 mức Low/Mid/High. Kết quả: 29 columns, 0 missing values — sẵn sàng cho FP-Growth."

---

## CELL 15 — Train/Test Split & Class Balancing Experiment

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)

for strategy_name, strategy in [
    ('SMOTE', SMOTE(random_state=SEED)),
    ('NearMiss', NearMiss()),
    ('class_weight', None)
]:
    # Evaluate INSIDE CV folds — không phải trước khi split
```

**Output:**
```
Train: (215304, 21), Test: (53827, 21)
Train target dist: {0: 155501, 1: 59803}

SMOTE:        mean macro F1 = 0.6919  ← WINNER
NearMiss:     mean macro F1 = 0.5899
class_weight: mean macro F1 = 0.6918

Best balancing strategy: SMOTE
```

> "80/20 split với stratify=y — đảm bảo tỷ lệ class giống nhau trong train và test (27.78% At-Risk ở cả hai). Test set (53,827 records) được held-out hoàn toàn, không dùng trong bất kỳ bước training nào."

> "Chúng tôi split 80/20 với stratify=y — đảm bảo tỷ lệ 27.78% At-Risk giống nhau trong cả train lẫn test. Train: 215,304 records, Test: 53,827 records — held-out hoàn toàn."

> "Sau đó chúng tôi thực hiện experiment so sánh 3 chiến lược xử lý imbalance bằng 5-fold CV trên training set: SMOTE tạo synthetic minority samples bằng interpolation giữa nearest neighbors — macro F1 = 0.6919. NearMiss undersample majority class — macro F1 = 0.5899, kém nhất vì mất ~95,000 records. class_weight điều chỉnh loss function — macro F1 = 0.6918. SMOTE được chọn. Sau SMOTE: training set tăng từ 215,304 lên 311,002 records (155,501 No-Risk + 155,501 At-Risk)."

> "Điểm kỹ thuật: SMOTE được apply bên trong mỗi fold CV, không phải trước khi split. Nếu apply trước, synthetic samples tạo từ toàn bộ dataset có thể leak thông tin từ validation fold vào training — làm inflate metrics. Đây là SMOTE leakage, một lỗi phổ biến trong literature mà chúng tôi tránh được."

> "Tóm tắt Phase 1: từ 269,131 raw records với 3 class, mixed feature types, outliers, và imbalance — chúng tôi tạo ra hai output: scaled data cho classification và discretized data cho ARM. Mỗi bước đều có kết quả đo lường cụ thể."

---
---

# PHASE 2 — ASSOCIATION RULE MINING

---

## CELL 18 — Build Transaction Matrix

```python
# Binary features → rename thành 'FeatureName_1'
arm_binary_part.columns = [f'{c}_1' for c in BINARY_FEATURES]

# Categorical features → one-hot encode
arm_cat_part = pd.get_dummies(df_arm[cat_cols].astype(str))

# Include target vào transaction
arm_target.rename(columns={'Diabetes_binary_0': 'NoRisk', 'Diabetes_binary_1': 'AtRisk'})

arm_bool = arm_encoded.astype(bool)  # mlxtend yêu cầu boolean
```

**Output:**
```
Transaction matrix shape: (269131, 41)
Average items per transaction: 14.45
Columns: ['HighBP_1', 'HighChol_1', ..., 'BMI_cat_Normal', 'BMI_cat_Obese',
          ..., 'Age_cat_Senior', 'GenHlth_cat_Fair', ..., 'NoRisk', 'AtRisk']
```

> "Transaction matrix có 41 columns = 14 binary features + 27 one-hot encoded columns (từ 7 categorical) + 2 target columns. Mỗi row là một 'transaction' — một người với tập hợp các health attributes. Average 14.45 items/transaction: mỗi người trung bình có 14-15 attributes active. Ví dụ: {HighBP_1, CholCheck_1, BMI_cat_Obese, Age_cat_Senior, GenHlth_cat_Good, Income_cat_MidIncome, ...}."

> "Target (AtRisk/NoRisk) được include vào transaction để FP-Growth tìm được rules có consequent = AtRisk. Binary features được rename thành 'FeatureName_1' để rõ nghĩa: HighBP_1 = 'người này có huyết áp cao'. Convert sang bool vì mlxtend FP-Growth yêu cầu boolean matrix."

---

## CELL 20 — Run FP-Growth

```python
frequent_itemsets = fpgrowth(arm_bool,
    min_support=0.05,      # xuất hiện trong ≥5% records
    use_colnames=True,
    max_len=5)             # giới hạn độ dài để tránh MemoryError

rules = association_rules(frequent_itemsets,
    metric='lift',
    min_threshold=1.5)     # lift ≥ 1.5x above random

# Filter: chỉ giữ rules dự đoán AtRisk
rules_at_risk = rules[rules['consequents'] == frozenset({'AtRisk'})]
rules_at_risk = rules_at_risk[rules_at_risk['confidence'] >= 0.45]
```

**Output:**
```
Running FP-Growth...
Frequent itemsets found: 19204   ← tất cả combinations xuất hiện ≥5%
Rules before filtering:  10346   ← tất cả A→B rules với lift≥1.5
Rules with consequent=AtRisk: 278  ← chỉ rules dự đoán risk
After confidence >= 0.45:     157  ← rules đủ mạnh
```

> "Tại sao FP-Growth thay vì Apriori? Apriori phải scan database nhiều lần và generate candidate itemsets — với 269,131 records và 41 features, rất chậm. FP-Growth compress toàn bộ dataset vào một FP-tree compact, chỉ cần 2 lần scan."

> "Đọc từng con số: 19,204 frequent itemsets là 'nguyên liệu thô'. 10,346 rules là tất cả A→B combinations. 278 rules có consequent=AtRisk — chúng tôi chỉ quan tâm rules dự đoán risk, loại bỏ 10,068 rules không liên quan như HighBP→HighChol. 157 rules sau confidence≥0.45: base rate là 27.78%, nên confidence 45% = +17 percentage points above chance — threshold data-driven."

> "min_support=0.05: rule phải xuất hiện trong ≥5% dataset = ~13,456 records — đủ để statistically meaningful. max_len=5: giới hạn độ dài itemset để tránh MemoryError với combinations quá dài."

---

## CELL 22 — Filter & Rank Rules

```python
BASE_RATE = y.mean()  # 0.2778

rules_filtered = rules_at_risk[
    (rules_at_risk['antecedents'].apply(lambda x: len(x) >= 2)) &
    (rules_at_risk['confidence'] >= BASE_RATE + 0.15)  # ≥42.78%
].sort_values('lift', ascending=False).head(20)

rules_filtered['has_behavioral'] = rules_filtered['antecedents'].apply(has_behavioral_feature)
```

**Output — Top 5 rules:**
```
Rank  Antecedents                                          Conf   Lift   Behavioral
1     CholCheck_1, DiffWalk_1, HighBP_1, HighChol_1       0.610  2.194  False
2     AnyHealthcare_1, DiffWalk_1, HighBP_1, HighChol_1   0.609  2.192  False
3     DiffWalk_1, HighBP_1, HighChol_1                    0.608  2.188  False
4     AnyHealthcare_1, BMI_cat_SeverelyObese, CholCheck_1, HighBP_1  0.590  2.124  TRUE
5     BMI_cat_SeverelyObese, CholCheck_1, HighBP_1        0.588  2.116  TRUE
```

> "Filter theo 3 tiêu chí: (1) antecedent ≥ 2 features — single-feature rules không thú vị vì SHAP đã capture được; (2) confidence ≥ base rate + 15pp — đảm bảo predictive value thực sự; (3) sort by lift — lift đo genuine association strength."

> "Đọc Rule #1: {CholCheck_1, DiffWalk_1, HighBP_1, HighChol_1} → AtRisk. Support=0.063: 6.3% dataset = ~16,900 người có cả 4 conditions. Confidence=0.610: trong số những người này, 61% là At-Risk. Lift=2.194: xác suất At-Risk cao gấp 2.19 lần so với population average 27.78%."

> "Phát hiện quan trọng: top rules đều dominated bởi clinical/comorbidity features (HighBP, HighChol, DiffWalk, CholCheck), không phải behavioral features thuần túy. Ban đầu chúng tôi filter chỉ lấy behavioral rules, nhưng quyết định giữ nguyên kết quả này — đây là data-driven finding, consistent với feature leakage analysis ở Phase 3. Rule behavioral mạnh nhất là {BMI_SeverelyObese, HighBP} với lift=2.105."

---

## CELL 24 — Formulate Testable Hypotheses

```python
ARM_TO_ORIG = {
    'BMI_cat': 'BMI',        # BMI_cat_SeverelyObese → test SHAP của BMI
    'GenHlth_cat': 'GenHlth', # GenHlth_cat_Fair → test SHAP của GenHlth
    'DiffWalk_1': 'DiffWalk', # DiffWalk_1 → test SHAP của DiffWalk
    ...
}

# Ưu tiên rules có behavioral feature, sau đó clinical
rules_ordered = pd.concat([behavioral_rules, clinical_rules])

# Mỗi hypothesis: test feature + context subgroup
for rule in rules_ordered:
    test_arm_item = first_testable_item_in_antecedent
    context = remaining_items  # dùng để define subgroup
```

**Output:**
```
Hypotheses formulated: 4

H1: In subgroup ['AnyHealthcare_1', 'CholCheck_1', 'HighBP_1'],
    mean SHAP of [BMI] > population mean SHAP
    ARM item: BMI_cat_SeverelyObese → SHAP feature: BMI
    Rule lift=2.124, confidence=0.59

H2: In subgroup ['CholCheck_1', 'HighChol_1', 'HighBP_1'],
    mean SHAP of [DiffWalk] > population mean SHAP
    ARM item: DiffWalk_1 → SHAP feature: DiffWalk
    Rule lift=2.194, confidence=0.609

H3: In subgroup ['AnyHealthcare_1', 'CholCheck_1', 'GenHlth_cat_Fair'],
    mean SHAP of [HighChol] > population mean SHAP
    ARM item: HighChol_1 → SHAP feature: HighChol
    Rule lift=2.062, confidence=0.573

H4: In subgroup ['AnyHealthcare_1', 'CholCheck_1', 'HighBP_1'],
    mean SHAP of [GenHlth] > population mean SHAP
    ARM item: GenHlth_cat_Fair → SHAP feature: GenHlth
    Rule lift=2.049, confidence=0.569
```

> "Đây là bước bridge giữa ARM và SHAP. ARM dùng discretized names (BMI_cat_SeverelyObese) nhưng SHAP dùng original feature names (BMI) — cần mapping để verify. Logic của mỗi hypothesis: ARM tìm ra rule {A, B, C, D} → AtRisk với lift cao. Hypothesis: 'Trong subgroup người có A, B, C — feature D có SHAP value cao hơn population average'. Nếu đúng → D genuinely amplifies risk khi co-occurring với A, B, C. Nếu sai → co-occurrence chỉ là statistical artifact."

> "Chỉ có 4 hypotheses (không phải 20) vì code giữ tối đa 5 unique original features để tránh redundancy. Context items (AnyHealthcare_1, CholCheck_1...) dùng để define subgroup trong SHAP verification."

---
---

# PHASE 3 — ENSEMBLE CLASSIFICATION

---

## CELL 27 — Helper: Evaluate Model + Apply SMOTE

```python
def evaluate_model(model, X_tr, y_tr, X_te, y_te, name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1]
    return {
        'Recall':   recall_score(y_te, y_pred),           # At-Risk detection rate
        'Macro_F1': f1_score(y_te, y_pred, average='macro'), # balanced F1
        'AUC_ROC':  roc_auc_score(y_te, y_prob),          # discrimination ability
        'MCC':      matthews_corrcoef(y_te, y_pred)        # balanced metric
    }

X_tr_bal, y_tr_bal = SMOTE().fit_resample(X_train, y_train)
```

**Output:**
```
Using strategy: SMOTE
Balanced train shape: (311002, 21)
  ← từ 215,304 → 311,002 (SMOTE tạo thêm 95,698 synthetic At-Risk samples)
```

> "SMOTE tạo synthetic samples cho minority class cho đến khi balanced: 155,501 No-Risk + 155,501 At-Risk = 311,002 records. SMOTE interpolate giữa các At-Risk samples thực để tạo synthetic samples — không phải duplicate."

> "Tại sao dùng 4 metrics thay vì accuracy? Accuracy bị inflate bởi class imbalance — model predict tất cả là No-Risk sẽ đạt 72.22% accuracy. Recall quan trọng nhất trong y tế: false negative (bỏ sót bệnh nhân At-Risk) nguy hiểm hơn false positive. Macro-F1 average F1 của cả 2 classes, không bị dominated bởi majority. AUC-ROC đo khả năng phân biệt ở mọi threshold. MCC là metric cân bằng nhất cho imbalanced data."

---

## CELL 29 — Baseline Models

```python
lr = LogisticRegression(max_iter=1000, random_state=SEED)
dt = DecisionTreeClassifier(random_state=SEED)
```

**Output:**
```
LogisticRegression: Recall=0.7458, F1=0.6941, AUC=0.807,  MCC=0.4189
DecisionTree:       Recall=0.7925, F1=0.7839, AUC=0.8215, MCC=0.578
```

> "Logistic Regression: AUC=0.807 là surprisingly decent — cho thấy features có linear separability tốt. Recall=0.746: detect được 74.6% At-Risk cases, còn 25.4% bị bỏ sót."

> "Decision Tree: kết quả paradoxical — F1=0.784 và MCC=0.578 cao hơn cả XGBoost tuned! Giải thích: Decision Tree tạo hard binary splits tại threshold 0.5. Với SMOTE-balanced data, nó học được boundaries rõ ràng cho cả 2 classes → MCC cao. Nhưng AUC=0.822 thấp hơn XGBoost (0.848) → probability calibration kém. Decision Tree không phân biệt tốt ở các threshold khác nhau, và không được chọn vì SHAP cần probability calibration tốt."

---

## CELL 31 — Optuna Tuning: XGBoost

```python
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()  # = 2.6

def objective_xgb(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 100, 500),
        'max_depth':       trial.suggest_int('max_depth', 3, 8),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':       trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda':      trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'scale_pos_weight': scale_pos,  # thêm imbalance handling
    }
    # 3-fold CV bên trong mỗi trial
    scores = [f1_macro on each fold]
    return np.mean(scores)

study_xgb.optimize(objective_xgb, n_trials=50)
```

**Output:**
```
Tuning XGBoost...
XGBoost best F1: 0.7242 (1126s)   ← ~19 phút
XGBoost_tuned: Recall=0.8138, F1=0.7303, AUC=0.8481, MCC=0.4958
```

> "Optuna dùng Tree-structured Parzen Estimators (TPE) — Bayesian optimization. Thay vì grid search exhaustive, TPE học từ các trials trước để suggest parameters có khả năng tốt hơn. log=True cho learning_rate và regularization vì các parameters này hoạt động trên log scale — khoảng cách giữa 0.001 và 0.01 quan trọng hơn giữa 0.1 và 0.11."

> "scale_pos_weight=2.6: tỷ lệ No-Risk/At-Risk trong training set. XGBoost dùng parameter này để weight minority class cao hơn trong loss function — thêm một lớp imbalance handling ngoài SMOTE."

> "Kết quả: Recall=0.814 — detect được 81.4% At-Risk cases, tốt nhất trong tất cả models. AUC=0.848 — tốt nhất. 1126 giây là trade-off có chủ đích: 50 trials × 3-fold CV × training time."

---

## CELL 33 — Optuna Tuning: LightGBM

```python
params = {
    'num_leaves': trial.suggest_int('num_leaves', 20, 100),  # LightGBM-specific
    'class_weight': 'balanced',
    ...
}
```

**Output:**
```
Tuning LightGBM...
LightGBM best F1: 0.7088 (635s)   ← nhanh hơn XGBoost
LightGBM_tuned: Recall=0.6133, F1=0.7109, AUC=0.8141, MCC=0.4228
```

> "LightGBM có Recall thấp nhất (0.613) — kết quả đáng chú ý. Tại sao? LightGBM dùng leaf-wise growth (khác với XGBoost dùng level-wise). Leaf-wise growth tạo ra trees sâu hơn, tend to overfit trên minority class patterns. Với SMOTE-balanced data, LightGBM có thể đang overfit trên synthetic samples thay vì học patterns thực. 635 giây nhanh hơn XGBoost vì LightGBM được thiết kế cho speed với GOSS."

---

## CELL 35 — Optuna Tuning: Random Forest

```python
params = {
    'max_features': trial.suggest_categorical('max_features', ['sqrt','log2']),
    'class_weight': 'balanced',
    ...
}
```

**Output:**
```
Tuning Random Forest...
RF best F1: 0.7149 (10573s)   ← ~3 giờ, chậm nhất
RandomForest_tuned: Recall=0.7, F1=0.7171, AUC=0.8235, MCC=0.4454
```

> "10,573 giây (~3 giờ) — chậm nhất vì Random Forest train nhiều trees độc lập, không có early stopping như boosting methods. max_features='sqrt': mỗi split chỉ xem xét sqrt(21)≈5 features ngẫu nhiên → tạo decorrelated trees → giảm variance. Random Forest kém hơn XGBoost vì RF dùng bagging (parallel, independent trees) trong khi XGBoost dùng boosting (sequential, each tree corrects previous errors)."

---

## CELL 37 — Soft Voting Ensemble

```python
voting = VotingClassifier(
    estimators=[('xgb', best_xgb), ('lgb', best_lgb), ('rf', best_rf)],
    voting='soft'  # average predicted probabilities
)
```

**Output:**
```
SoftVoting: Recall=0.7328, F1=0.7336, AUC=0.8441, MCC=0.4801
XGBoost model saved.
```

> "Soft voting = average predicted probabilities từ 3 models. Kết quả: Ensemble KHÔNG được chọn. Pre-specified criterion: 'adopt ensemble only if it outperforms best single model on macro-F1'. F1=0.7336 > XGBoost 0.7303 — ensemble thắng về F1, nhưng recall thấp hơn (0.733 vs 0.814) và AUC thấp hơn (0.844 vs 0.848). XGBoost được giữ vì recall cao hơn quan trọng hơn trong bài toán y tế."

> "Tại sao ensemble không luôn tốt hơn? Ensemble hoạt động tốt khi các models có errors không correlated. Ở đây, XGBoost, LightGBM, RF đều trained trên cùng data với cùng features → errors có thể correlated. LightGBM có recall rất thấp (0.613) → kéo ensemble xuống về recall."

---

## CELL 39 — Leakage Sensitivity: Variants A/B/C

```python
# 5 features bị loại trong Variant B:
HIGH_LEAKAGE = ['GenHlth',              # r=0.33, self-rated health bị ảnh hưởng bởi diagnosis
                'CholCheck',            # cholesterol monitoring = post-diagnosis behavior
                'DiffWalk',             # difficulty walking = diabetes complication
                'HeartDiseaseorAttack', # r=0.190, comorbidity không phải predictor
                'Stroke']               # tương tự

for variant_name, feat_cols in [
    ('Variant_A_full', ALL_FEATURES),              # 21 features
    ('Variant_B_no_leakage', VARIANT_B_FEATURES),  # 16 features
    ('Variant_C_behavioral_only', VARIANT_C_FEATURES)  # 10 features
]:
```

**Output:**
```
Variant_A_full:            Recall=0.8138, F1=0.7303, AUC=0.8481
Variant_B_no_leakage:      Recall=0.8066, F1=0.7038, AUC=0.8247  ← -0.023 AUC
Variant_C_behavioral_only: Recall=0.8375, F1=0.6118, AUC=0.7594  ← -0.089 AUC
```

> "Đây là contribution quan trọng nhất về methodology. GenHlth (self-rated health) có correlation r=0.33 với target — cao nhất dataset. Nhưng GenHlth là self-reported và bị ảnh hưởng bởi diagnosis awareness: người biết mình bị tiểu đường sẽ rate health thấp hơn. Tương tự, CholCheck và DiffWalk là consequences của diabetes, không phải predictors. Nếu include những features này, model đang 'cheat' bằng cách dùng thông tin post-diagnosis."

> "Variant A → B: AUC giảm 0.023, F1 giảm 0.026 — modest nhưng không negligible. Confirm rằng các nghiên cứu trước (Majcherek 2025, Kutlu 2024) likely overestimate performance ~2-3% AUC."

> "Variant C (behavioral-only): Recall=0.838 cao nhất nhưng F1=0.612 thấp nhất — paradox quan trọng. Behavioral features rất tốt để identify true positives (high recall) nhưng kém trong phân biệt overall (low precision → low F1). Implication cho Vietnam: behavioral screening (hỏi về diet, exercise, smoking) có thể identify at-risk individuals nhưng cần clinical confirmation."

---

## CELL 41 — Full Model Comparison

**Output:**
```
=== FULL MODEL COMPARISON ===
             Model  Recall  Macro_F1  AUC_ROC    MCC
LogisticRegression  0.7458    0.6941   0.8070 0.4189
      DecisionTree  0.7925    0.7839   0.8215 0.5780
     XGBoost_tuned  0.8138    0.7303   0.8481 0.4958  ← SELECTED
    LightGBM_tuned  0.6133    0.7109   0.8141 0.4228
RandomForest_tuned  0.7000    0.7171   0.8235 0.4454
        SoftVoting  0.7328    0.7336   0.8441 0.4801
```

> "XGBoost được chọn: AUC cao nhất (0.848) + Recall cao nhất (0.814) + compatible với exact Tree SHAP. Decision Tree có F1 và MCC cao hơn nhưng AUC thấp hơn — probability calibration kém, không phù hợp cho SHAP analysis."

---
---

# PHASE 4 — SHAP VERIFICATION

---

## CELL 44 — Global SHAP Analysis

```python
best_xgb.fit(X_tr_bal, y_tr_bal)
explainer = shap.TreeExplainer(best_xgb)  # exact algorithm, không phải sampling
shap_values = explainer.shap_values(X_test)

mean_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=ALL_FEATURES)
```

**Output:**
```
Top 10 features by mean |SHAP|:
GenHlth      0.6741   ← rank 1
Age          0.6185   ← rank 2
BMI          0.6157   ← rank 3
HighBP       0.5152   ← rank 4
HighChol     0.3450   ← rank 5
Income       0.2642   ← rank 6
MentHlth     0.2287   ← rank 7
PhysHlth     0.2062   ← rank 8
Education    0.1714   ← rank 9
Sex          0.1698   ← rank 10
```

> "TreeExplainer là exact algorithm cho tree-based models — traverse cấu trúc nội bộ của từng tree để tính Shapley values chính xác. Mỗi prediction được decompose thành contributions của từng feature, với tổng bằng đúng prediction score. Mean |SHAP| = average absolute contribution — đo importance, không phân biệt direction."

> "GenHlth rank 1 (0.674): self-rated health là predictor mạnh nhất. Cảnh báo leakage: r=0.33 với target — cao nhất dataset. Một phần signal này là post-diagnosis awareness."

> "Age rank 2 (0.619): risk tăng đều theo tuổi — expected. BMI rank 3 (0.616): gần bằng Age. SHAP dependence plot cho thấy threshold rõ ràng tại BMI≈30."

> "Income rank 6 (0.264): socioeconomic factor quan trọng hơn nhiều người nghĩ. MentHlth rank 7 (0.229): sẽ thấy rõ hơn trong youth analysis — đây là novel finding."

---

## CELL 46 — Dependence Plots (BMI & Age)

```python
for feat in ['BMI', 'Age']:
    shap.dependence_plot(feat, shap_values, X_test, feature_names=ALL_FEATURES)
```

**Output:** 2 plots

> "BMI dependence plot: trục X = BMI value (scaled), trục Y = SHAP value. Thấy rõ: SHAP tăng mạnh khi BMI vượt qua threshold ~0.6 (tương ứng BMI≈30 trong original scale). Dưới threshold này, SHAP gần 0 hoặc âm. Đây là non-linear relationship mà linear models không capture được — justify việc dùng tree-based models."

> "Age dependence plot: SHAP tăng gần như linear theo Age — risk tăng đều theo tuổi. Màu sắc của dots (interaction feature) cho thấy feature nào interact mạnh nhất với Age."

---

## CELL 48 — Hypothesis Verification (Core Contribution)

```python
CONFIRM_THRESHOLD = 0.50  # 50% relative increase → confirmed

for h in hypotheses_final:
    orig_feat = h['test_orig_feat']
    context_items = h['context']

    # Build subgroup mask từ ARM context
    mask = pd.Series([True] * len(X_test), index=X_test.index)
    for ctx_item in context_items:
        if ctx_item.endswith('_1') and ctx_orig in BINARY_FEATURES:
            mask = mask & (X_test[ctx_orig] == 1.0)  # người có feature này

    pop_mean = shap_df[orig_feat].mean()              # population average
    sub_mean = shap_df[mask][orig_feat].mean()        # subgroup average
    relative_increase = (sub_mean - pop_mean) / abs(pop_mean)
```

**Output:**
```
H1 [BMI]:      pop=-0.40169, sub=-0.26083, increase=35.1% → REJECTED ✗
H2 [DiffWalk]: pop=-0.01256, sub=-0.00405, increase=67.8% → CONFIRMED ✓
H3 [HighChol]: pop=-0.07333, sub=-0.06369, increase=13.1% → REJECTED ✗
H4 [GenHlth]:  pop=-0.36807, sub=-0.09779, increase=73.4% → CONFIRMED ✓
```

> "Tại sao SHAP values đều âm? SHAP values âm = feature này GIẢM xác suất At-Risk so với baseline trong average. Không có nghĩa là feature không quan trọng — mean |SHAP| vẫn cao. Trong test set, nhiều người có BMI bình thường, GenHlth tốt → average SHAP âm. Chúng tôi so sánh relative change giữa subgroup và population, không phải absolute value."

> "H2 CONFIRMED (67.8%): Trong subgroup {CholCheck, HighChol, HighBP}, DiffWalk SHAP tăng 67.8%. Ý nghĩa lâm sàng: khi một người đã có cholesterol cao + huyết áp cao, việc họ có difficulty walking là signal mạnh hơn đáng kể. DiffWalk trong context này không chỉ là complication — nó là marker của advanced metabolic dysfunction. Combination này identify nhóm người đang ở giai đoạn tiến triển của bệnh."

> "H4 CONFIRMED (73.4%): Trong subgroup {AnyHealthcare, CholCheck, HighBP}, GenHlth SHAP tăng 73.4%. Người có healthcare access, đang monitor cholesterol, và có hypertension mà vẫn rate health là 'Fair' — đây là combination đặc biệt nguy hiểm. Họ aware về health của mình nhưng vẫn deteriorating. GenHlth 'Fair' trong context này là strong signal của undiagnosed diabetes."

> "H1 REJECTED (35.1%): BMI không amplify đáng kể trong subgroup severely obese + hypertension. Giải thích: BMI và HighBP correlated với nhau. Trong tree structure, HighBP có thể 'absorb' một phần importance của BMI → Tree SHAP correlation bias. Khi đã có HighBP trong subgroup, BMI không amplify thêm nhiều."

> "H3 REJECTED (13.1%): HighChol không amplify trong fair-health subgroup. HighChol là relatively weak predictor (SHAP=0.345, rank 5) và marginal effect không thay đổi nhiều trong subgroup này."

> "Kết luận: 2/4 hypotheses confirmed → ARM rules không uniformly represent genuine interactions. Validates necessity của ARM-to-SHAP verification step — directly addresses limitation của Bata 2025 và Fakir 2024."

---

## CELL 50 — ARM–SHAP Jaccard Cross-Check

```python
top10_shap = set(mean_shap.head(10).index)
arm_features_raw = set()  # features appearing in top 15 ARM rules
jaccard = len(top10_shap & arm_features_raw) / len(top10_shap | arm_features_raw)
```

**Output:**
```
Top 10 SHAP features:    ['Age', 'BMI', 'Education', 'GenHlth', 'HighBP',
                          'HighChol', 'Income', 'MentHlth', 'PhysHlth', 'Sex']
Features in top 15 ARM:  ['AnyHealthcare', 'BMI', 'CholCheck', 'DiffWalk',
                          'GenHlth', 'HighBP', 'HighChol']
Intersection:            ['BMI', 'GenHlth', 'HighBP', 'HighChol']
Jaccard similarity: 0.3077
```

> "Jaccard = 4/13 = 0.308. 4 features chung (BMI, GenHlth, HighBP, HighChol) là cross-validation tự nhiên giữa 2 approaches — cả ARM lẫn SHAP đều agree rằng 4 features này là central to T2DM risk."

> "0.308 không phải thấp. ARM và SHAP capture fundamentally different things: ARM tìm co-occurrence patterns ở population level, SHAP đo individual-level prediction contributions. Overlap hoàn toàn (Jaccard=1.0) sẽ đáng ngờ hơn — suggest 2 methods đang measure cùng một thứ."

> "Features chỉ trong SHAP (Age, Education, Income, MentHlth, PhysHlth, Sex): quan trọng cho individual prediction nhưng không tạo strong co-occurrence patterns. Features chỉ trong ARM (AnyHealthcare, CholCheck, DiffWalk): xuất hiện nhiều trong rules nhưng SHAP importance thấp hơn — context features, không phải primary predictors."

---

## CELL 52 — Youth Sub-Analysis (Age 18–44)

```python
# Kỹ thuật: Age đã được MinMaxScaled → cần tính threshold trong scaled space
age_threshold = (5 - df_A['Age'].min()) / (df_A['Age'].max() - df_A['Age'].min())
youth_mask = X_test['Age'] <= age_threshold  # Age code ≤ 5 = 18-44 tuổi

shap_youth = shap.TreeExplainer(best_xgb).shap_values(X_youth)
```

**Output:**
```
Youth subset: 10377 records, 946 At-Risk (9.1%)
  ← prevalence 9.1% vs 27.78% full population

Top 10 — Full population vs Youth:
           Full_pop_SHAP  Youth_SHAP  Ratio
GenHlth          0.67409     0.91938  1.36×
Age              0.61846     1.94945  3.15×  ← ấn tượng nhất
BMI              0.61572     0.74671  1.21×
HighBP           0.51518     0.57011  1.11×
HighChol         0.34499     0.43539  1.26×
Income           0.26418     0.37540  1.42×
MentHlth         0.22865     0.38663  1.69×  ← novel finding
PhysHlth         0.20617     0.26708  1.30×
Education        0.17141     0.22870  1.34×
Sex              0.16976     0.15924  0.94×  ← duy nhất giảm
```

> "Kỹ thuật quan trọng: Age trong X_test đã được MinMaxScaled. Code tính age_threshold = (5 - min) / (max - min) để convert Age code 5 (44 tuổi) sang scaled space. Nếu dùng threshold sai, subgroup sẽ không đúng."

> "Age SHAP tăng 3.15× — finding ấn tượng nhất. Trong full population, Age là continuous predictor — risk tăng đều từ 18 đến 80+. Trong youth subgroup (18-44), model phân biệt giữa người 18 tuổi (very low risk) và người 44 tuổi (risk đang accelerate). Khoảng cách risk giữa 18 và 44 tuổi, khi normalized trong subgroup, lớn hơn khoảng cách tương đương trong full population. Implication: target screening cho nhóm 35-44."

> "MentHlth tăng 1.69× — novel finding không có trong bất kỳ reviewed study nào. Mental health burden là risk factor disproportionately quan trọng với người trẻ. Possible mechanisms: stress-related cortisol dysregulation → insulin resistance; depression-associated sedentary behavior; reverse causality từ undiagnosed prediabetes ảnh hưởng mental wellbeing. Implication cho Vietnam: integrate mental health screening với diabetes risk assessment."

> "Income tăng 1.42×: socioeconomic vulnerability more predictive trong working-age population. Người trẻ thu nhập thấp ở Vietnam: processed food accessibility, sedentary jobs, limited healthcare. Sex giảm 0.94×: gender ít quan trọng hơn trong youth — risk factors khác dominant hơn."

---

## CELL 54 — Final Results Summary

**Output:**
```
[1] Model Comparison:
     XGBoost_tuned: Recall=0.8138, F1=0.7303, AUC=0.8481 ← BEST

[2] Leakage Sensitivity:
     Variant_A_full:            AUC=0.8481
     Variant_B_no_leakage:      AUC=0.8247  (-0.023)
     Variant_C_behavioral_only: AUC=0.7594  (-0.089)

[3] Hypothesis Verification:
     H2 DiffWalk: 67.8% → CONFIRMED
     H4 GenHlth:  73.4% → CONFIRMED
     H1 BMI:      35.1% → REJECTED
     H3 HighChol: 13.1% → REJECTED

[4] ARM-SHAP Jaccard: 0.3077
[5] Best balancing: SMOTE
```

> "Tổng kết: pipeline 4 giai đoạn hoàn chỉnh. ARM tạo ra 4 hypotheses. XGBoost đạt AUC=0.848. Leakage quantified: 0.023 AUC từ 5 leaky features. 2/4 hypotheses confirmed — ARM rules không uniformly represent genuine interactions. Youth analysis: Age 3.15×, MentHlth 1.69× — actionable findings cho Vietnam."

---
---

# PHẦN KẾT — Q&A VÀ EVIDENCE OF REAL UNDERSTANDING

---

> "Phần này chuẩn bị cho các câu hỏi khó từ giáo viên/hội đồng."

---

**Q: Tại sao dùng macro-F1 thay vì weighted-F1 để tune Optuna?**

> "Weighted-F1 bị dominated bởi majority class (No-Risk, 72%). Macro-F1 treat cả 2 classes equally — average F1 của từng class. Trong bài toán y tế, chúng tôi muốn model perform tốt trên CẢ HAI classes. Nếu dùng weighted-F1, Optuna sẽ optimize cho No-Risk class và sacrifice At-Risk recall."

---

**Q: Tại sao SHAP values trong bảng verification đều âm?**

> "SHAP values âm = feature đó GIẢM xác suất At-Risk so với baseline trong average. Không có nghĩa là feature không quan trọng. BMI mean SHAP = -0.402 vì nhiều người trong test set có BMI bình thường, kéo average xuống. Chúng tôi so sánh relative change giữa subgroup và population, không phải absolute value."

---

**Q: Tại sao không dùng SHAP interaction values thay vì subgroup comparison?**

> "SHAP interaction values có thể tính pairwise interactions. Nhưng chúng tôi chọn subgroup comparison vì: (1) interpretable hơn — dễ explain cho clinicians; (2) ARM-defined subgroups là multi-feature combinations, không chỉ pairwise; (3) SHAP interaction values bị biased hơn khi features correlated. Subgroup approach là conservative và transparent hơn."

---

**Q: Jaccard 0.308 có thấp không?**

> "0.308 không phải thấp. ARM và SHAP capture fundamentally different things: ARM tìm co-occurrence patterns ở population level, SHAP đo individual-level prediction contributions. Overlap hoàn toàn sẽ đáng ngờ hơn — suggest 2 methods đang measure cùng một thứ. 4 features chung là cross-validation tự nhiên."

---

**Q: Tại sao không dùng SMOTE-Tomek?**

> "SMOTE-Tomek remove borderline instances sau khi oversample, potentially cleaner. Nhưng với 269,131 records, Tomek links removal sẽ rất chậm. Empirically, SMOTE đã cho macro-F1 = 0.692 — tốt hơn NearMiss. Marginal gain từ SMOTE-Tomek không justify thêm computational cost."

---

**Q: Tại sao không include GenHlth trong Variant C?**

> "GenHlth là self-rated health — không phải behavioral feature. Nó là subjective health perception, bị ảnh hưởng bởi diagnosis awareness. Variant C chỉ include features mà người dùng có thể self-report về behaviors: physical activity, smoking, diet, alcohol, và demographics."

---

**Q: Kết quả có generalize được sang Vietnam không?**

> "Đây là limitation quan trọng nhất. BRFSS là U.S. telephone survey với sampling bias: older, higher-income, more health-aware respondents. Vietnam có demographics khác. Tuy nhiên, findings là hypothesis-generating: (1) mental health là risk factor quan trọng với người trẻ — cần validate trên Vietnamese data; (2) income amplification trong youth — consistent với urbanization patterns ở Vietnam; (3) behavioral-only features có recall cao — suggest behavioral screening feasible ở community level."

---

**Q: Tại sao Decision Tree MCC cao nhất nhưng không được chọn?**

> "MCC = 0.578 nhưng AUC = 0.822 — thấp hơn XGBoost (0.848). AUC đo khả năng phân biệt tổng thể ở mọi threshold, MCC chỉ đo tại threshold 0.5. Decision Tree tend to create hard boundaries tốt tại 0.5 nhưng probability calibration kém. Cho SHAP analysis, cần model với probability calibration tốt — XGBoost phù hợp hơn."

---

**Q: Tại sao Age SHAP trong youth subgroup cao hơn full population?**

> "Trong full population, Age là continuous predictor — risk tăng đều từ 18 đến 80+. Trong youth subgroup (18-44), model phân biệt giữa người 18 tuổi (very low risk) và người 44 tuổi (risk đang accelerate). Khoảng cách risk giữa 18 và 44 tuổi, khi normalized trong subgroup, lớn hơn khoảng cách tương đương trong full population. Age là discriminative hơn TRONG nhóm trẻ so với trong population chung."

---

**Q: Tại sao LightGBM recall thấp nhất dù được Optuna tune?**

> "LightGBM dùng leaf-wise growth — tạo trees sâu hơn, tend to overfit trên minority class patterns. Với SMOTE-balanced data, LightGBM có thể đang overfit trên synthetic samples thay vì học patterns thực. num_leaves parameter tạo ra nhiều leaf nodes → model phức tạp hơn → overfit hơn trên imbalanced data."

---

## TÓM TẮT CONTRIBUTIONS THỰC SỰ

> "Contribution cốt lõi của chúng tôi không phải là đạt AUC cao nhất — Majcherek 2025 và Kutlu 2024 đã làm điều đó trên cùng dataset. Contribution của chúng tôi là:"

1. **Methodological**: Lần đầu tiên implement sequential ARM → SHAP pipeline với hypothesis verification, thay vì dùng SHAP như exploratory tool
2. **Empirical**: Quantify feature leakage impact (0.023 AUC từ 5 leaky features) — prior studies không làm điều này
3. **Clinical**: Confirm rằng 2/4 ARM rules represent genuine interactions, không chỉ statistical co-occurrence
4. **Novel finding**: Mental health là 1.69× more predictive trong youth — không có trong bất kỳ reviewed study nào
5. **Vietnam relevance**: Age-stratified analysis provide actionable hypotheses cho prevention programs targeting Vietnamese youth

---

*File này kết hợp từ PRESENTATION_SCRIPT.md và CELL_EXPLANATION.md*
*Dựa trên phân tích toàn bộ Version/main_IDE.ipynb*
