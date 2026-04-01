"""
Diabetes Association Rule Mining - Proof of Concept
CLC01 Group 9 - Demonstrating feasibility of ARM approach
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    """Load both datasets and compare characteristics"""
    print("=== DATASET COMPARISON ===")
    
    # Load BRFSS dataset
    try:
        brfss = pd.read_csv('data.csv')
        print(f"BRFSS Dataset: {brfss.shape[0]} records, {brfss.shape[1]} features")
        print(f"Target distribution:\n{brfss['Diabetes_binary'].value_counts()}")
        print(f"Class balance: {brfss['Diabetes_binary'].value_counts(normalize=True)}")
    except:
        print("BRFSS dataset not found")
        brfss = None
    
    # Load Pima dataset  
    try:
        pima = pd.read_csv('diabetes.csv')
        print(f"\nPima Dataset: {pima.shape[0]} records, {pima.shape[1]} features")
        print(f"Target distribution:\n{pima['Outcome'].value_counts()}")
        print(f"Class balance: {pima['Outcome'].value_counts(normalize=True)}")
    except:
        print("Pima dataset not found")
        pima = None
        
    return brfss, pima

def preprocess_for_arm(df, dataset_type='brfss'):
    """Convert dataset to binary transaction format for ARM"""
    
    if dataset_type == 'brfss':
        # Binarize target: combine pre-diabetes and diabetes
        df['At_Risk'] = (df['Diabetes_binary'] >= 1).astype(int)
        
        # Create binary features for ARM
        transactions = []
        
        for _, row in df.head(1000).iterrows():  # Sample for POC
            transaction = []
            
            # Add lifestyle factors
            if row['HighBP'] == 1: transaction.append('HighBP')
            if row['HighChol'] == 1: transaction.append('HighChol') 
            if row['BMI'] >= 30: transaction.append('BMI_Obese')
            elif row['BMI'] >= 25: transaction.append('BMI_Overweight')
            if row['Smoker'] == 1: transaction.append('Smoker')
            if row['PhysActivity'] == 0: transaction.append('No_Exercise')
            if row['Fruits'] == 0: transaction.append('No_Fruits')
            if row['Veggies'] == 0: transaction.append('No_Veggies')
            if row['HvyAlcoholConsump'] == 1: transaction.append('Heavy_Alcohol')
            if row['GenHlth'] >= 4: transaction.append('Poor_Health')
            if row['Age'] >= 9: transaction.append('Senior')
            elif row['Age'] <= 4: transaction.append('Young_Adult')
            
            # Add target
            if row['At_Risk'] == 1: transaction.append('DIABETES_RISK')
                
            transactions.append(transaction)
            
    else:  # Pima dataset
        transactions = []
        
        for _, row in df.iterrows():
            transaction = []
            
            # Add risk factors
            if row['Glucose'] >= 140: transaction.append('High_Glucose')
            if row['BloodPressure'] >= 90: transaction.append('High_BP')
            if row['BMI'] >= 30: transaction.append('BMI_Obese')
            elif row['BMI'] >= 25: transaction.append('BMI_Overweight')
            if row['Age'] >= 40: transaction.append('Age_40Plus')
            if row['Pregnancies'] >= 3: transaction.append('Multi_Pregnancy')
            if row['Insulin'] > 0: transaction.append('Insulin_Treatment')
            
            # Add target
            if row['Outcome'] == 1: transaction.append('DIABETES_RISK')
                
            transactions.append(transaction)
    
    return transactions

def run_association_rules(transactions, min_support=0.05, min_confidence=0.6):
    """Run FP-Growth and generate association rules"""
    
    # Convert to transaction encoder format
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"\n=== TRANSACTION MATRIX ===")
    print(f"Shape: {df_encoded.shape}")
    print(f"Items: {list(df_encoded.columns)}")
    
    # Find frequent itemsets
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    print(f"\nFrequent itemsets found: {len(frequent_itemsets)}")
    
    if len(frequent_itemsets) == 0:
        print("No frequent itemsets found. Lowering min_support...")
        frequent_itemsets = fpgrowth(df_encoded, min_support=0.02, use_colnames=True)
    
    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        # Filter rules that predict diabetes risk
        diabetes_rules = rules[rules['consequents'].astype(str).str.contains('DIABETES_RISK')]
        
        print(f"\n=== TOP DIABETES RISK RULES ===")
        if len(diabetes_rules) > 0:
            # Sort by lift and display top rules
            top_rules = diabetes_rules.nlargest(10, 'lift')
            
            for idx, rule in top_rules.iterrows():
                antecedent = ', '.join(list(rule['antecedents']))
                consequent = ', '.join(list(rule['consequents']))
                print(f"\nRule: {antecedent} → {consequent}")
                print(f"  Support: {rule['support']:.3f}")
                print(f"  Confidence: {rule['confidence']:.3f}")  
                print(f"  Lift: {rule['lift']:.3f}")
                
            return top_rules
        else:
            print("No rules predicting diabetes risk found")
            return None
    else:
        print("No frequent itemsets found")
        return None

def visualize_results(rules):
    """Create basic visualizations of ARM results"""
    if rules is None or len(rules) == 0:
        print("No rules to visualize")
        return
        
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Support vs Confidence
    plt.subplot(2, 2, 1)
    plt.scatter(rules['support'], rules['confidence'], s=rules['lift']*20, alpha=0.7)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (size = lift)')
    
    # Plot 2: Lift distribution
    plt.subplot(2, 2, 2)
    plt.hist(rules['lift'], bins=10, alpha=0.7)
    plt.xlabel('Lift')
    plt.ylabel('Frequency')
    plt.title('Distribution of Lift Values')
    
    # Plot 3: Top rules by lift
    plt.subplot(2, 1, 2)
    top_5 = rules.nlargest(5, 'lift')
    rule_labels = [f"Rule {i+1}" for i in range(len(top_5))]
    plt.barh(rule_labels, top_5['lift'])
    plt.xlabel('Lift')
    plt.title('Top 5 Rules by Lift')
    
    plt.tight_layout()
    plt.savefig('arm_results_poc.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("DIABETES ASSOCIATION RULE MINING - PROOF OF CONCEPT")
    print("=" * 60)
    
    # Load and explore datasets
    brfss, pima = load_and_explore_data()
    
    # Choose dataset based on availability
    if brfss is not None:
        print("\n=== USING BRFSS DATASET ===")
        transactions = preprocess_for_arm(brfss, 'brfss')
        print(f"Sample transactions: {transactions[:3]}")
        
        # Run ARM with optimal thresholds (discovered via debug)
        rules = run_association_rules(transactions, min_support=0.05, min_confidence=0.5)
        
    elif pima is not None:
        print("\n=== USING PIMA DATASET ===")
        transactions = preprocess_for_arm(pima, 'pima')
        print(f"Sample transactions: {transactions[:3]}")
        
        # Run ARM with lower thresholds for smaller dataset
        rules = run_association_rules(transactions, min_support=0.1, min_confidence=0.7)
        
    else:
        print("No datasets available for analysis")
        return
    
    # Visualize results
    if rules is not None:
        visualize_results(rules)
        
        print(f"\n=== PROOF OF CONCEPT SUMMARY ===")
        print(f"✓ Successfully loaded and preprocessed data")
        print(f"✓ Created transaction format for ARM")
        print(f"✓ Generated {len(rules)} diabetes risk rules")
        print(f"✓ Identified actionable lifestyle patterns")
        print(f"✓ Demonstrated feasibility of approach")
    else:
        print("\n=== ADJUSTMENTS NEEDED ===")
        print("- Lower support/confidence thresholds")
        print("- Different feature discretization")
        print("- Larger sample size")

if __name__ == "__main__":
    main()