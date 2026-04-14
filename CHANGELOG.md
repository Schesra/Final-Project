# Code Changelog — Mining Behavioral Risk Patterns for Diabetes Prediction
**CLC01 — Group 9**

Tài liệu này ghi lại các thay đổi code theo từng phiên bản, lý do thay đổi, và kết quả quan sát được. Dùng để tham khảo khi viết báo cáo nghiên cứu.

---

## v1.0 — Khởi tạo pipeline (normalization.ipynb)

**File:** `normalization.ipynb`

**Nội dung:**
- Binarize target: Class 0 → No Risk, Class 1+2 → At Risk
- BMI outlier: IQR Capping (Strategy A) và Record Removal BMI>60 (Strategy B)
- MinMaxScaler cho continuous + ordinal features
- Discretize features cho ARM

**Vấn đề phát hiện:**
- `MentHlth_cat` bị 66% null và `PhysHlth_cat` bị 58% null
- Nguyên nhân: `pd.cut` với `bins=[-1, 0, 13, 30]` không capture được giá trị `0.0` (178,365 records có MentHlth=0)

**Fix:**
- Đổi sang `bins=[0, 1, 14, 31]` với `right=False, include_lowest=True`
- Kết quả sau fix: 0 missing values trên tất cả 7 cột categorical

---

## v2.0 — Tạo main.ipynb (pipeline đầy đủ)

**File:** `main.ipynb`

**Nội dung:**
- Gộp toàn bộ 4 phases vào một file duy nhất (yêu cầu nộp bài)
- Phase 1: Data Preparation
- Phase 2: Association Rule Mining (FP-Growth)
- Phase 3: Ensemble Classification (XGBoost + LightGBM + RF + Soft Voting)
- Phase 4: SHAP Verification

**Quyết định thiết kế:**
- Dùng Google Colab + Drive làm môi trường chạy chính
- SMOTE/NearMiss/class_weight được so sánh trong CV folds (không apply trước split)
- Optuna 50 trials cho mỗi model, objective = macro F1
- XGBoost là primary SHAP target (TreeExplainer exact, không approximate)

---

## v2.1 — Fix setup cell (pip install)

**Vấn đề:** `!pip install` không chạy được trên Windows Jupyter — `pip` không có trong PATH

**Fix:** Đổi sang `!{sys.executable} -m pip install ...` để trỏ đúng Python interpreter

**Thêm:** `matplotlib`, `seaborn`, `scikit-learn` vào danh sách install (bị thiếu ở v2.0)

---

## v2.2 — Fix transaction matrix cho ARM (Phase 2, Cell 2.1)

**Vấn đề:** Binary features bị `astype(str)` rồi `get_dummies` tạo ra 2 columns mỗi feature (`_0` và `_1`) thay vì 1 — làm transaction matrix phình từ ~35 lên 55 columns, gây MemoryError ở `association_rules()`

**Quan sát:** FP-Growth tìm được 1,842,683 frequent itemsets → `association_rules()` bị MemoryError

**Fix:**
- Binary features: giữ nguyên 0/1, chỉ rename thành `{feature}_1`
- Categorical features: `get_dummies` bình thường
- Transaction matrix giảm từ 55 → 41 columns

**Kết quả sau fix:** 19,204 frequent itemsets, không còn MemoryError

---

## v2.3 — Điều chỉnh FP-Growth thresholds

**Vấn đề ban đầu:** `min_support=0.10, min_lift=2.0, max_len=4` → chỉ 3,788 itemsets, 2 rules, 0 rules với consequent=AtRisk

**Nguyên nhân:** `min_support=0.10` quá cao — itemset `{X, AtRisk}` cần support ≥ 0.10 nhưng AtRisk chỉ chiếm 27.78%

**Fix:** Đổi sang `min_support=0.05, min_lift=1.5, max_len=5`

**Kết quả:** 19,204 itemsets, 10,346 rules, 278 rules với consequent=AtRisk

---

## v2.4 — Điều chỉnh confidence threshold và filter behavioral

**Vấn đề:** Filter `confidence >= 0.60` + bắt buộc có behavioral feature → 0 rules sau filter

**Phân tích 3 rules còn lại (conf≥0.60):**
```
['DiffWalk_1', 'HighBP_1', 'HighChol_1'] → AtRisk  conf=0.608, lift=2.188
['CholCheck_1', 'DiffWalk_1', 'HighBP_1', 'HighChol_1'] → AtRisk  conf=0.609, lift=2.194
['AnyHealthcare_1', 'DiffWalk_1', 'HighBP_1', 'HighChol_1'] → AtRisk  conf=0.609, lift=2.192
```
→ Tất cả đều là clinical features, không có behavioral feature nào

**Data-driven finding (nhất quán với proposal Section 4):**
- Behavioral features thuần túy (PhysActivity r=-0.120, Smoker r=0.055, Fruits r=-0.030) có correlation quá yếu để tạo rules có confidence cao
- Clinical features (HighBP r=0.289, DiffWalk r=0.223) dominate high-confidence rules
- Đây là finding có giá trị khoa học: ARM xác nhận leakage risk đã được dự báo trong proposal

**Thử nghiệm các mức confidence:**
| Confidence | Total rules | Rules có behavioral feature |
|---|---|---|
| ≥ 0.55 | 31 | 9 |
| ≥ 0.50 | 55 | 10 |
| ≥ 0.45 | 153 | 55 |
| ≥ 0.40 | 272 | 112 |

**Quyết định:** Hạ xuống `confidence >= 0.45` (vẫn cao hơn base rate 27.78% là 62%)

**Fix:**
- Bỏ filter bắt buộc behavioral feature
- Thêm cột `has_behavioral` để tag và báo cáo
- Hạ confidence gap từ 20pp xuống 15pp (0.2778 + 0.15 = 0.4278)

**Kết quả:** 153 rules với consequent=AtRisk, 55 rules có behavioral feature

---

## v2.5 — Chuyển sang local IDE (main_IDE.ipynb)

**Lý do:** VS Code Jupyter kernel timeout do `pyzmq` port communication issue

**Thay đổi so với main.ipynb:**
- Bỏ `from google.colab import drive` và `drive.mount()`
- Đổi `DATA_PATH = '/content/drive/MyDrive/data.csv'` → `'./data.csv'`
- Đổi `SAVE_PATH` → `'./outputs/'`
- Thêm `warnings.filterwarnings("ignore")` ở load data cell
- Xóa `device='cuda'` và `device='gpu'` (không cần cho Colab)
- Clear tất cả outputs để notebook sạch

---

## v2.6 — Suppress DeprecationWarning spam

**Vấn đề:** `jupyter_client/session.py:203: DeprecationWarning: datetime.datetime.utcnow()` spam liên tục trong output

**Nguyên nhân:** Warning đến từ subprocess của Jupyter infrastructure, không phải Python process chính — `warnings.filterwarnings` không chặn được

**Fix (4 tầng):**
```python
os.environ['PYTHONWARNINGS'] = 'ignore'          # môi trường
warnings.filterwarnings('ignore')                 # Python warnings
logging.getLogger('jupyter_client').setLevel(logging.CRITICAL)  # jupyter_client logs
logging.getLogger('tornado').setLevel(logging.CRITICAL)         # tornado logs
!pip install --upgrade jupyter_client 2>/dev/null  # fix source
```

**Lưu ý:** Warning này là Colab infrastructure dùng `jupyter_client` cũ, không ảnh hưởng kết quả

---

## Kết quả Phase 1 & 2 (đã verify)

### Phase 1
| Cell | Kết quả |
|---|---|
| 1.1 Binarize | No Risk: 194,377 (72.22%), At Risk: 74,754 (27.78%) |
| 1.2 BMI Outlier | IQR cap=42.50 (11,280 records), Strategy B removed 993 records |
| 1.3 Normalize | MinMaxScaler: tất cả 7 features về [0.0, 1.0] |
| 1.4 Discretize | ARM dataset: 269,131 × 29 cols, 0 missing values |
| 1.5 Balancing | SMOTE=0.6919, NearMiss=0.5899, class_weight=0.6918 → **Best: SMOTE** |

### Phase 2
| Cell | Kết quả |
|---|---|
| 2.1 Transaction matrix | 269,131 × 41 cols, avg 14.45 items/transaction |
| 2.2 FP-Growth | 19,204 itemsets, 10,346 rules, 278 rules AtRisk, 153 rules conf≥0.45 |
| 2.3 Filter | Đang chạy lại sau fix confidence threshold |

---

## v2.7 — Fix cell 2.4: Hypothesis formulation logic

**Vấn đề:** Chỉ tạo được 1 hypothesis dù có 20 rules

**Nguyên nhân:**
- Logic tìm behavioral feature chỉ match tên gốc (`BMI`, `PhysActivity`...) nhưng trong ARM rules, BMI đã được discretize thành `BMI_cat_SeverelyObese` — chỉ match được 1 rule duy nhất
- Chỉ scan top 10 rules, không ưu tiên rules có behavioral feature

**Fix:**
1. Thêm `ARM_TO_ORIG` mapping đầy đủ cho tất cả ARM items → original feature names:
   ```python
   'HighBP_1': 'HighBP', 'HighChol_1': 'HighChol', 'DiffWalk_1': 'DiffWalk', ...
   ```
2. Mở rộng `ALL_TESTABLE` để include cả clinical features (không chỉ behavioral)
3. Ưu tiên behavioral rules trước khi lấy clinical rules:
   ```python
   rules_ordered = pd.concat([behavioral_rules, clinical_rules])
   ```
4. Scan toàn bộ rules thay vì chỉ top 10

**Kết quả:** 4 hypotheses (tăng từ 1)
- H1: BMI (behavioral) — context: HighBP + CholCheck + AnyHealthcare, lift=2.124
- H2: HighChol (clinical) — context: DiffWalk + HighBP, lift=2.194
- H3: GenHlth (clinical/leakage) — context: HighBP + CholCheck, lift=2.049
- H4: DiffWalk (clinical/leakage) — context: HighBP + CholCheck, lift=2.014

**Finding quan trọng cho báo cáo:** Chỉ H1 là behavioral hypothesis thực sự. H2–H4 dominated bởi clinical features — xác nhận empirically phân tích feature leakage trong proposal Section 4.

---

## v2.8 — Fix cell 4.3: KeyError 'test_feature'

**Vấn đề:** `KeyError: 'test_feature'` khi chạy hypothesis verification

**Nguyên nhân:** Key mismatch giữa cell 2.4 và 4.3:
- Cell 2.4 (v2.7) đã đổi sang key `test_arm_item` và `test_orig_feat`
- Cell 4.3 vẫn dùng key cũ `test_feature` — không tồn tại

**Fix cell 4.3:**
- Đổi `h['test_feature']` → `h['test_orig_feat']` cho SHAP lookup
- Dùng `ARM_TO_ORIG` để map context items về original feature names khi build subgroup mask
- Thêm cột `ARM_Item` và `Is_Behavioral` vào verification results để phân biệt behavioral vs clinical hypotheses
- Fix subgroup mask logic: dùng `ctx_item.startswith(prefix)` thay vì string contains

---

## v2.9 — Xóa create_ide_notebook.py

**Lý do:** Script này clear toàn bộ outputs khi tạo lại `main_IDE.ipynb`, làm mất kết quả đã chạy

**Quyết định:** Từ nay sửa trực tiếp vào từng file, không dùng script generate nữa
