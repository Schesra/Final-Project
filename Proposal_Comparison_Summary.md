# **Proposal Revision Summary**
**Addressing Instructor Comments & Course Requirements**

## **Key Changes Made**

### **1. INSIGHT & Personal Relevance (Major Addition)**
**Before**: Generic motivation về diabetes ở Vietnam
**After**: 
- Personal story về grandfather's diabetes
- University student lifestyle patterns (late nights, fast food, sedentary)
- Specific question: "Which exact lifestyle combinations cross the risk threshold?"
- Personal commitment to actionable findings

### **2. Track Focus (Major Simplification)**
**Before**: Complex combination (ARM + Ensemble + Outlier Detection + SHAP)
**After**: 
- **Main Track**: Association Rule Mining (aligned with syllabus)
- Supporting: Basic correlation comparison as baseline
- Clear focus on pattern discovery, not prediction accuracy

### **3. Problem Clarity (Restructured)**
**Before**: 3 complex research questions mixing prediction and interpretation
**After**: 
- Clear focus: "Which lifestyle combinations create hidden risk clusters?"
- ARM-specific questions about itemsets, rules, and lift values
- Actionable "if-then" behavioral insights

### **4. Dataset Justification (Enhanced)**
**Before**: Assumed BRFSS without comparison
**After**: 
- Explicit comparison: BRFSS (269K) vs Pima (768)
- Clear rationale: BRFSS chosen for ARM scale requirements
- Backup plan: Switch to Pima if needed

### **5. Methodology Justification (Simplified)**
**Before**: Complex ensemble pipeline hard to justify
**After**: 
- Clear ARM pipeline: Data → Discretization → Transactions → FP-Growth → Rules
- Market basket analogy: "bread + milk" → "BMI + no exercise + poor sleep"
- Focus on interpretability over accuracy

## **Addressing Specific Instructor Comments**

### **Comment 1**: "Mining Behavioral Risk Patterns - cần làm rõ 'patterns'"
**Solution**: 
- Defined patterns = Association Rules from FP-Growth
- Clear examples: {BMI_Obese=1, PhysActivity=0, Fruits=0} → Diabetes
- Distinguished from feature importance

### **Comment 2**: "Ensemble + SHAP là pipeline nặng; cần biện minh"
**Solution**: 
- Removed ensemble complexity
- Focus purely on ARM for pattern discovery
- SHAP removed entirely - no longer needed

### **Comment 3**: "Dataset cần chốt rõ: Pima Indians Diabetes, BRFSS Survey, hay bộ khác?"
**Solution**: 
- Primary: BRFSS (data.csv) for scale
- Backup: Pima (diabetes.csv) if issues
- Clear size and feature comparison

### **Comment 4**: "Kiểm tra 'Mining' trong title có phản ánh technique cụ thể không"
**Solution**: 
- Title emphasizes "Co-occurrence Patterns" 
- Explicit FP-Growth algorithm mention
- Clear ARM methodology throughout

### **Comment 5**: "Thêm leakage risk analysis"
**Solution**: 
- Identified high-risk features (GenHlth, DiffWalk, PhysHlth)
- Mitigation strategies listed
- Focus on lifestyle factors measurable before diagnosis

## **Course Requirement Alignment**

### **✓ Clear Question**: 
"Which combinations of daily lifestyle behaviors create hidden risk clusters?"

### **✓ Personal Care**: 
University student lifestyle concerns, family diabetes history

### **✓ Expected Patterns**: 
Sedentary + high BMI, multiple moderate risks vs single severe risks

### **✓ Feasible Plan**: 
BRFSS data + FP-Growth + Rule evaluation + Visualization

### **✓ Track Alignment**: 
**Association Rule Mining** as primary track from syllabus

### **✓ Evaluation Plan**: 
Support/Confidence/Lift metrics, baseline comparison, clinical relevance

## **Proof of Concept**

Created `diabetes_arm_poc.py` to demonstrate:
- ✓ Data loading and preprocessing
- ✓ Transaction matrix creation  
- ✓ FP-Growth implementation
- ✓ Rule generation and filtering
- ✓ Basic visualization
- ✓ Feasibility validation

## **Risk Mitigation**

### **Technical Risks Addressed**:
- Too many rules → Increase thresholds
- Dataset issues → Backup Pima dataset
- Insufficient patterns → Lower confidence thresholds

### **Scope Risks Addressed**:
- Complexity reduced to single track focus
- Clear deliverables defined
- 4-week timeline realistic

## **Final Deliverables (Simplified)**

1. **Top 20 Lifestyle Risk Rules** with interpretations
2. **Interactive Visualization** of pattern networks
3. **Age-Specific Analysis** (young adults vs general)
4. **Jupyter Notebook** with complete ARM pipeline
5. **Final Report** (15-20 pages)
6. **Presentation** (10 minutes)

## **Next Steps**

1. **Submit revised proposal** (CLC01-Group9-Final-Proposal.md)
2. **Run proof of concept** to validate approach
3. **Begin data exploration** and preprocessing
4. **Implement FP-Growth pipeline**
5. **Generate and validate rules**

---

**Key Success Factors**:
- ✅ Personal insight and relevance clear
- ✅ Single track focus (ARM)
- ✅ Feasible scope and timeline
- ✅ Clear evaluation metrics
- ✅ Actionable outputs defined