"""
Diabetes ARM - Debug Version
Fix: No diabetes rules issue
"""

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

def debug_diabetes_distribution():
    """Check diabetes distribution in sample"""
    df = pd.read_csv('data.csv')
    
    # Check original distribution
    print("=== ORIGINAL TARGET DISTRIBUTION ===")
    print(df['Diabetes_binary'].value_counts())
    print(df['Diabetes_binary'].value_counts(normalize=True))
    
    # Check first 1000 records (what we're using)
    sample = df.head(1000)
    print(f"\n=== SAMPLE (1000 records) DISTRIBUTION ===")
    print(sample['Diabetes_binary'].value_counts())
    print(sample['Diabetes_binary'].value_counts(normalize=True))
    
    # Binarize and check
    sample['At_Risk'] = (sample['Diabetes_binary'] >= 1).astype(int)
    print(f"\n=== AFTER BINARIZATION ===")
    print(sample['At_Risk'].value_counts())
    print(f"At Risk percentage: {sample['At_Risk'].mean():.3f}")
    
    return sample

def create_transactions_debug(df):
    """Create transactions with debug info"""
    transactions = []
    diabetes_count = 0
    
    for idx, row in df.iterrows():
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
        
        # Add target - CRITICAL FIX
        at_risk = (row['Diabetes_binary'] >= 1)
        if at_risk: 
            transaction.append('DIABETES_RISK')
            diabetes_count += 1
            
        transactions.append(transaction)
    
    print(f"\n=== TRANSACTION DEBUG ===")
    print(f"Total transactions: {len(transactions)}")
    print(f"Transactions with DIABETES_RISK: {diabetes_count}")
    print(f"DIABETES_RISK percentage: {diabetes_count/len(transactions):.3f}")
    
    # Show sample transactions with diabetes
    diabetes_transactions = [t for t in transactions if 'DIABETES_RISK' in t]
    print(f"\nSample diabetes transactions:")
    for i, t in enumerate(diabetes_transactions[:3]):
        print(f"  {i+1}: {t}")
    
    return transactions

def run_arm_with_lower_thresholds(transactions):
    """Run ARM with progressively lower thresholds"""
    
    # Convert to transaction encoder format
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"\n=== ENCODED MATRIX DEBUG ===")
    print(f"Shape: {df_encoded.shape}")
    print(f"DIABETES_RISK column exists: {'DIABETES_RISK' in df_encoded.columns}")
    if 'DIABETES_RISK' in df_encoded.columns:
        diabetes_support = df_encoded['DIABETES_RISK'].mean()
        print(f"DIABETES_RISK support: {diabetes_support:.3f}")
    
    # Try different support thresholds
    support_levels = [0.05, 0.03, 0.02, 0.01]
    
    for min_support in support_levels:
        print(f"\n=== TRYING SUPPORT = {min_support} ===")
        
        try:
            # Find frequent itemsets
            frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
            print(f"Frequent itemsets found: {len(frequent_itemsets)}")
            
            if len(frequent_itemsets) > 0:
                # Check if DIABETES_RISK is in frequent itemsets
                diabetes_itemsets = frequent_itemsets[
                    frequent_itemsets['itemsets'].astype(str).str.contains('DIABETES_RISK')
                ]
                print(f"Itemsets containing DIABETES_RISK: {len(diabetes_itemsets)}")
                
                if len(diabetes_itemsets) > 0:
                    print("Sample diabetes itemsets:")
                    for idx, row in diabetes_itemsets.head(3).iterrows():
                        print(f"  {row['itemsets']} (support: {row['support']:.3f})")
                
                # Try to generate rules with lower confidence
                confidence_levels = [0.6, 0.5, 0.4, 0.3]
                
                for min_confidence in confidence_levels:
                    try:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                        
                        if len(rules) > 0:
                            # Filter diabetes rules
                            diabetes_rules = rules[rules['consequents'].astype(str).str.contains('DIABETES_RISK')]
                            
                            if len(diabetes_rules) > 0:
                                print(f"\n🎉 SUCCESS! Found {len(diabetes_rules)} diabetes rules")
                                print(f"Support: {min_support}, Confidence: {min_confidence}")
                                
                                # Show top rules
                                top_rules = diabetes_rules.nlargest(5, 'lift')
                                for idx, rule in top_rules.iterrows():
                                    antecedent = ', '.join(list(rule['antecedents']))
                                    print(f"\nRule: {antecedent} → DIABETES_RISK")
                                    print(f"  Support: {rule['support']:.3f}")
                                    print(f"  Confidence: {rule['confidence']:.3f}")
                                    print(f"  Lift: {rule['lift']:.3f}")
                                
                                return diabetes_rules
                            else:
                                print(f"  No diabetes rules with confidence >= {min_confidence}")
                        else:
                            print(f"  No rules generated with confidence >= {min_confidence}")
                    except Exception as e:
                        print(f"  Error with confidence {min_confidence}: {e}")
            else:
                print(f"  No frequent itemsets with support >= {min_support}")
                
        except Exception as e:
            print(f"Error with support {min_support}: {e}")
    
    print("\n❌ No diabetes rules found with any threshold combination")
    return None

def main():
    print("DIABETES ARM - DEBUG VERSION")
    print("=" * 50)
    
    # Debug diabetes distribution
    sample_df = debug_diabetes_distribution()
    
    # Create transactions with debug
    transactions = create_transactions_debug(sample_df)
    
    # Run ARM with multiple thresholds
    rules = run_arm_with_lower_thresholds(transactions)
    
    if rules is not None:
        print(f"\n✅ SOLUTION FOUND: Use lower thresholds in main implementation")
    else:
        print(f"\n🔍 NEED INVESTIGATION: Check feature encoding or sample size")

if __name__ == "__main__":
    main()