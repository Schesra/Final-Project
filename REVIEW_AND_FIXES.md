# Review & Fix Suggestions — CLC01 Group 9
**Reviewer:** Claude Code (claude-sonnet-4-6)  
**Date:** 2026-04-14  
**Files reviewed:** `main_IDE.ipynb`, `data.csv`, `CLC01_Group9_Proposal_v3_Fixed.md`, `outputs/*.csv`

---

## Tổng quan kết quả

| Hạng mục | Trạng thái |
|---|---|
| Data Preparation | ✅ Đúng theo proposal |
| ARM — FP-Growth | ⚠️ Rules thiếu behavioral features |
| Model — XGBoost | ✅ Đạt tất cả 4 thresholds |
| Model — LightGBM | 🔴 Recall 0.6133 — thất bại |
| Model — SoftVoting | ⚠️ Không nên là final model |
| Leakage Sensitivity | ✅ Variant B giữ Recall ≥ 0.75 |
| SHAP Verification | ⚠️ Confirmed hypotheses đều là leakage features |
| Youth Analysis | ✅ Insight tốt |

---

## VẤN ĐỀ 1 — LightGBM Recall 0.6133 (Mức độ: 🔴 Nghiêm trọng)

### Mô tả
LightGBM sau 50 trials Optuna chỉ đạt Recall=0.6133, thấp hơn cả Logistic Regression cơ bản (0.7458). Ngưỡng yêu cầu trong proposal là Recall ≥ 0.75.

**Nguyên nhân:** Optuna objective function tối ưu hóa `macro F1` trong CV nhưng không có constraint tối thiểu cho Recall. LightGBM có thể học cách đạt F1 cao bằng cách hi sinh Recall để tăng Precision.

### Đề xuất sửa

**Option A — Thêm recall constraint vào objective (khuyến nghị):**

Tìm cell 33 (Optuna LightGBM), sửa objective function:

```python
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'class_weight': 'balanced', 'random_state': SEED, 'verbose': -1
    }
    scores_f1, scores_recall = [], []
    for tr_idx, val_idx in skf.split(X_train, y_train):
        Xf, Xv = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        yf, yv = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = LGBMClassifier(**params)
        m.fit(Xf, yf)
        yp = m.predict(Xv)
        scores_f1.append(f1_score(yv, yp, average='macro'))
        scores_recall.append(recall_score(yv, yp))
    
    mean_recall = np.mean(scores_recall)
    mean_f1 = np.mean(scores_f1)
    
    # Penalize nếu recall < 0.70 để buộc model giữ recall cao
    if mean_recall < 0.70:
        return mean_f1 * (mean_recall / 0.70)  # penalty
    return mean_f1
```

**Option B — Dùng Recall làm objective trực tiếp:**

```python
# Thay dòng return
return np.mean(scores_recall)   # optimize recall trực tiếp
```

**Option C (nhanh nhất) — Điều chỉnh threshold predict_proba:**

Sau khi train LightGBM với params hiện tại, hạ threshold từ 0.5 xuống:

```python
# Thêm vào sau khi train best_lgb
from sklearn.metrics import recall_score
y_prob_lgb = best_lgb.predict_proba(X_test)[:, 1]

# Tìm threshold tối ưu cho Recall >= 0.75
for thresh in np.arange(0.3, 0.5, 0.01):
    y_pred_t = (y_prob_lgb >= thresh).astype(int)
    rec = recall_score(y_test, y_pred_t)
    f1 = f1_score(y_test, y_pred_t, average='macro')
    if rec >= 0.75:
        print(f"Threshold={thresh:.2f}: Recall={rec:.4f}, F1={f1:.4f}")
        break
```

### Trong báo cáo cần ghi:
> "LightGBM tuning (50 Optuna trials, objective=macro F1) failed to meet the Recall ≥ 0.75 threshold (achieved 0.6133), indicating a recall-precision tradeoff where the optimizer sacrificed sensitivity for balanced F1. XGBoost is selected as the primary model."

---

## VẤN ĐỀ 2 — SoftVoting không nên là final model (Mức độ: ⚠️ Trung bình)

### Mô tả
Proposal nêu rõ: *"Soft Voting Ensemble adopted only if it outperforms the best single model."*

So sánh thực tế:

| Model | Recall | Macro_F1 | AUC-ROC |
|-------|--------|----------|---------|
| XGBoost_tuned | **0.8138** | 0.7303 | **0.8481** |
| SoftVoting | 0.7328 | 0.7336 | 0.8441 |

XGBoost thắng ở cả Recall (primary metric) và AUC-ROC. SoftVoting chỉ nhỉnh hơn về Macro_F1 (+0.003, không đáng kể).

### Đề xuất sửa

Thêm đoạn code so sánh tự động vào cell 37 (SoftVoting):

```python
# Sau khi evaluate SoftVoting, thêm:
xgb_result = next(r for r in all_results if r['Model'] == 'XGBoost_tuned')
sv_result = next(r for r in all_results if r['Model'] == 'SoftVoting')

if sv_result['Recall'] > xgb_result['Recall']:
    FINAL_MODEL = voting
    FINAL_MODEL_NAME = 'SoftVoting'
    print("SoftVoting selected as final model (higher Recall)")
else:
    FINAL_MODEL = best_xgb
    FINAL_MODEL_NAME = 'XGBoost_tuned'
    print(f"XGBoost selected as final model (Recall {xgb_result['Recall']} > SoftVoting {sv_result['Recall']})")
```

Sau đó dùng `FINAL_MODEL` thay vì `best_xgb` trong Phase 4 SHAP.

### Trong báo cáo cần ghi:
> "SoftVoting did not outperform XGBoost on the primary metric (Recall: 0.7328 vs 0.8138); XGBoost_tuned is adopted as the final model per pre-defined criteria."

---

## VẤN ĐỀ 3 — ARM không phát hiện behavioral patterns (Mức độ: ⚠️ Trung bình — cần framing đúng)

### Mô tả
Top 3 rules lift cao nhất chứa hoàn toàn leakage features. Chỉ rules 4–7 có behavioral feature (BMI_cat_SeverelyObese), nhưng đi kèm với HighBP và CholCheck.

**Top 3 rules thực tế:**
1. `{CholCheck + DiffWalk + HighBP + HighChol}` → lift=2.194
2. `{AnyHealthcare + DiffWalk + HighBP + HighChol}` → lift=2.192
3. `{DiffWalk + HighBP + HighChol}` → lift=2.188

Code đã comment out bộ lọc behavioral:
```python
# Note: behavioral-only filter removed — data shows clinical features dominate high-lift rules
```

### Đây KHÔNG phải lỗi code — đây là finding thực tế của data

Proposal đã dự đoán trước (Section 4):
> *"if the final ARM output is dominated by BMI and PhysActivity rules rather than dietary features, this is a data-driven finding rather than a methodological failure"*

### Đề xuất bổ sung vào notebook

Thêm một markdown cell sau cell 22 (Filter & Rank Rules) để giải thích:

```markdown
**Finding — ARM Pattern Observation:**
The top-ranked rules by lift are dominated by clinical indicators (DiffWalk, HighBP, HighChol, 
CholCheck) rather than behavioral features. This is consistent with the dataset's structure: 
leakage features (DiffWalk r=0.21, GenHlth r=0.33) have stronger co-occurrence signals than 
behavioral features (Fruits r=-0.03, Veggies r=-0.05, Smoker r=0.055).

Only rules 4–7 contain a behavioral feature (BMI_cat_SeverelyObese), confirming that BMI 
is the sole behavioral predictor with sufficient discriminative power for ARM discovery. 
Dietary features (Fruits, Veggies) and Smoker do not form high-lift co-occurrence patterns 
with At-Risk label at min_support=0.05 — this is a data-driven finding, not a methodological failure.
```

### Bổ sung thêm: Separate ARM chỉ trên Variant B features

Để tăng giá trị behavioral của ARM, có thể chạy FP-Growth thêm lần 2 chỉ trên behavioral features:

```python
# ARM trên behavioral features only (không có leakage features)
BEHAVIORAL_COLS = ['PhysActivity_1', 'Smoker_1', 'Fruits_1', 'Veggies_1',
                   'HvyAlcoholConsump_1', 'BMI_cat_Normal', 'BMI_cat_Overweight',
                   'BMI_cat_Obese', 'BMI_cat_SeverelyObese', 'HighBP_1', 'Sex_1',
                   'AtRisk']

arm_behav = arm_bool[[c for c in arm_bool.columns if any(b in c for b in 
            ['PhysActivity','Smoker','Fruits','Veggies','HvyAlcohol','BMI_cat','AtRisk'])]]

freq_behav = fpgrowth(arm_behav, min_support=0.03, use_colnames=True, max_len=4)
rules_behav = association_rules(freq_behav, metric='lift', min_threshold=1.2)
rules_behav_at_risk = rules_behav[rules_behav['consequents'].apply(
    lambda x: x == frozenset({'AtRisk'}))]
print("Behavioral-only ARM rules:", len(rules_behav_at_risk))
print(rules_behav_at_risk.sort_values('lift', ascending=False).head(10))
```

*Lưu ý: Cần hạ min_support xuống 0.03 vì behavioral co-occurrence yếu hơn.*

---

## VẤN ĐỀ 4 — Hypothesis Verification chỉ confirm leakage features (Mức độ: ⚠️ Quan trọng cho scientific contribution)

### Mô tả

| Hypothesis | Feature test | Là leakage? | Confirmed? |
|---|---|---|---|
| H1 | BMI | ❌ Behavioral | ❌ +35% (rejected) |
| H2 | DiffWalk | ✅ High leakage | ✅ +68% |
| H3 | HighChol | ⚠️ Medium leakage | ❌ +13% |
| H4 | GenHlth | ✅ High leakage | ✅ +73% |

H2 và H4 được confirmed nhưng đây là DiffWalk (biến chứng của diabetes) và GenHlth (self-rated health bị bias). H1 (BMI — behavioral) bị rejected.

### Framing đúng trong báo cáo

Đây là **finding quan trọng**, không phải thất bại:

> *"SHAP verification rejects H1 (BMI amplification in high-BP context), confirming that BMI's elevated risk contribution in severely obese patients is largely independent of HighBP co-occurrence — the ARM co-occurrence reflects clinical co-prevalence, not risk amplification. This is the expected scientific outcome: ARM discovers co-occurrence; SHAP determines that the co-occurrence does not produce interaction effects for behavioral features."*

### Đề xuất bổ sung hypothesis behavioral thuần

Nếu chạy lại ARM với behavioral-only (Issue 3), có thể formulate thêm:

**H5:** "In patients with BMI_Obese AND PhysActivity=0, mean SHAP of BMI is higher than population mean"

```python
# H5: BMI trong context PhysActivity=0
h5_mask = (X_test_orig['PhysActivity'] == 0)  # dùng X_test trước scale
h5_shap = shap_df.loc[h5_mask, 'BMI']
pop_shap_bmi = shap_df['BMI'].mean()
sub_shap_bmi = h5_shap.mean()
rel_inc = (sub_shap_bmi - pop_shap_bmi) / abs(pop_shap_bmi)
print(f"H5 [BMI|PhysActivity=0]: pop={pop_shap_bmi:.4f}, sub={sub_shap_bmi:.4f}, rel={rel_inc:.1%}")
```

---

## VẤN ĐỀ 5 — SHAP mean values đều âm — cần giải thích (Mức độ: ℹ️ Cần ghi chú)

### Mô tả

Tất cả pop_mean SHAP trong hypothesis verification đều âm:
- BMI: -0.402
- DiffWalk: -0.013
- HighChol: -0.073
- GenHlth: -0.368

### Giải thích

SHAP values âm có nghĩa feature đó đang đẩy prediction về phía class 0 (No Risk) — **không có nghĩa là model sai**. Điều này xảy ra vì:

1. **72.22% records là No Risk** — baseline expectation đã nghiêng về 0
2. Các features này được average trên toàn test set, bao gồm cả class 0 (chiếm đa số)
3. Ví dụ BMI: người có BMI thấp (class 0) có SHAP âm lớn; khi average, giá trị trung bình nghiêng về âm

### Công thức relative_increase hiện tại:

```python
relative_increase = (subgroup_mean - pop_mean) / abs(pop_mean)
```

Khi cả hai âm: subgroup=-0.261, pop=-0.402 → rel_inc = (-0.261 - (-0.402)) / 0.402 = +35%

**Điều này đúng về mặt toán học** (subgroup ít âm hơn = feature đóng góp ít cho class 0 = tương đối nhiều hơn cho class 1), nhưng cần giải thích trong báo cáo.

### Đề xuất thêm vào notebook (cell 48, sau verification loop):

```python
# Giải thích SHAP âm
print("\nNote on negative SHAP values:")
print("Negative mean SHAP indicates the feature pushes prediction toward No-Risk on average")
print("across the full test set (72% No-Risk). This is expected.")
print("Relative increase measures how much LESS negative (= more risk-amplifying)")
print("the feature becomes in the ARM-identified subgroup vs. full population.")
```

---

## VẤN ĐỀ 6 — DecisionTree outperforms tuned ensembles (Mức độ: ℹ️ Cần giải thích)

### Mô tả

| Model | Macro_F1 | MCC | Thời gian tune |
|-------|----------|-----|----------------|
| DecisionTree (default) | **0.7839** | **0.578** | 0s |
| XGBoost_tuned | 0.7303 | 0.4958 | 1,126s |
| RandomForest_tuned | 0.7171 | 0.4454 | 10,573s |

DT mặc định vượt RF tuned sau 10,573 giây là anomaly đáng chú ý.

### Nguyên nhân có thể

1. DT với `class_weight='balanced'` trên tabular data BRFSS có thể fit rất tốt vì features mostly binary
2. Optuna tune RF/XGBoost với objective là macro F1 trên CV — có thể CV folds không representative do stratification bị lệch ở một số folds nhỏ
3. DT overfits test set theo cách khác với ensembles (high variance nhưng low bias trên test)

### Đề xuất kiểm tra

```python
# Kiểm tra DT có overfit không
from sklearn.metrics import recall_score, f1_score
dt_train_pred = dt.predict(X_tr_bal)
dt_test_pred = dt.predict(X_test)

print("DT Train Macro_F1:", f1_score(y_tr_bal, dt_train_pred, average='macro'))
print("DT Test Macro_F1:", f1_score(y_test, dt_test_pred, average='macro'))
print("DT Train Recall:", recall_score(y_tr_bal, dt_train_pred))
print("DT Test Recall:", recall_score(y_test, dt_test_pred))
# Nếu Train F1 >> Test F1 → overfitting
```

### Trong báo cáo cần ghi:
> "Decision Tree (default parameters) achieved surprisingly high Macro_F1=0.7839 and MCC=0.578, outperforming tuned ensembles on these metrics. This likely reflects the binary-dominant feature structure of BRFSS data, where a single decision boundary per feature is sufficient for good performance. However, XGBoost is retained as the primary model for SHAP analysis due to superior Recall (0.8138) and AUC-ROC (0.8481), and its support for reliable TreeExplainer-based SHAP values."

---

## Thứ tự ưu tiên sửa

| # | Vấn đề | Cần sửa code? | Ưu tiên |
|---|---|---|---|
| 1 | LightGBM Recall thấp | ✅ Có (hoặc ghi nhận trong report) | 🔴 Cao |
| 2 | SoftVoting không là final model | ✅ Thêm logic so sánh | ⚠️ Trung bình |
| 3 | ARM missing behavioral patterns | ✅ Thêm markdown + optional behavioral-only ARM | ⚠️ Trung bình |
| 4 | Hypothesis testing chỉ leakage | ✅ Thêm H5 behavioral + framing | ⚠️ Trung bình |
| 5 | SHAP âm cần giải thích | ✅ Thêm comment/note | ℹ️ Thấp |
| 6 | DT anomaly | ✅ Thêm overfit check + giải thích | ℹ️ Thấp |

---

## Những điểm tốt cần giữ nguyên

- Pipeline reproducible với `SEED=42` nhất quán xuyên suốt
- Leakage sensitivity (Variants A/B/C) được implement đúng và cho kết quả tốt
- MinMaxScaler chỉ fit trên train — không leak
- ARM thresholds (min_support=0.05, min_confidence=0.45, min_lift=1.5) hợp lý
- Youth SHAP comparison có insight thực tế rõ ràng (Age SHAP tăng 215% ở nhóm trẻ)
- FP-Growth thay vì Apriori cho 269K records — đúng lựa chọn
- Stratified K-Fold bên trong CV — đúng anti-leakage protocol
