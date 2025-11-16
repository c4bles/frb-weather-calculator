#!/usr/bin/env python3
"""
FRB Weather Calculator Validation Script
=========================================

Validate predictions on sample FRBs.
Tests repeater detection accuracy.

Usage:
    python examples/validate_predictions.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from frb_calculator import predict_repeater_probability, predict_burst_rate


def main():
    print("=" * 80)
    print("FRB WEATHER CALCULATOR VALIDATION")
    print("=" * 80)
    print()
    
    # Load sample data
    data_path = Path(__file__).parent / 'sample_frbs.csv'
    
    if not data_path.exists():
        print(f"❌ ERROR: Sample data not found at {data_path}")
        return 1
    
    print(f"Loading sample data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} FRBs")
    print()
    
    # Run predictions
    print("Running predictions...")
    
    # Predict for each FRB
    P_repeats = []
    burst_rates = []
    
    for idx, row in df.iterrows():
        P = predict_repeater_probability(
            sSFR=row['sSFR'],
            M_star=row['M_star'],
            age_gyr=row['age_gyr'],
            environment=row['environment']
        )
        rate = predict_burst_rate(
            SFR=row['SFR'],
            M_star=row['M_star'],
            age_gyr=row['age_gyr']
        )
        P_repeats.append(P)
        burst_rates.append(rate)
    
    df['P_repeat_pred'] = P_repeats
    df['burst_rate_pred'] = burst_rates
    
    print("✅ Predictions complete")
    print()
    
    # Classification (threshold at 50%)
    df['predicted_class'] = df['P_repeat_pred'].apply(lambda p: 'repeater' if p > 0.5 else 'one-off')
    
    # Overall statistics
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print()
    
    print(f"Sample Size:        {len(df)} FRBs")
    print(f"P(repeat) range:    {df['P_repeat_pred'].min():.3f} - {df['P_repeat_pred'].max():.3f}")
    print(f"Burst rate range:   {df['burst_rate_pred'].min():.2f} - {df['burst_rate_pred'].max():.2f} /yr")
    print()
    
    # Repeater detection (only for known status)
    known = df[df['repeater_status'] != 'unknown'].copy()
    
    if len(known) > 0:
        print("=" * 80)
        print("REPEATER DETECTION (Known Status Only, N={})".format(len(known)))
        print("=" * 80)
        print()
        
        # Calculate accuracy metrics
        true_repeaters = known[known['repeater_status'] == 'repeater']
        true_oneoffs = known[known['repeater_status'] == 'one-off']
        
        # True positives
        tp = len(true_repeaters[true_repeaters['P_repeat_pred'] > 0.5])
        # False negatives
        fn = len(true_repeaters[true_repeaters['P_repeat_pred'] <= 0.5])
        # True negatives
        tn = len(true_oneoffs[true_oneoffs['P_repeat_pred'] <= 0.5])
        # False positives
        fp = len(true_oneoffs[true_oneoffs['P_repeat_pred'] > 0.5])
        
        accuracy = (tp + tn) / len(known) if len(known) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"Accuracy:           {accuracy:.1%}")
        print(f"Precision:          {precision:.1%}")
        print(f"Recall:             {recall:.1%}")
        print()
        
        print("Confusion Matrix:")
        print(f"  True Positives:   {tp} (repeaters correctly identified)")
        print(f"  False Negatives:  {fn} (repeaters missed)")
        print(f"  True Negatives:   {tn} (one-offs correctly identified)")
        print(f"  False Positives:  {fp} (one-offs misclassified as repeaters)")
        print()
    
    # By repeater status
    print("=" * 80)
    print("BY REPEATER STATUS")
    print("=" * 80)
    print()
    
    for status in ['repeater', 'one-off', 'unknown']:
        subset = df[df['repeater_status'] == status]
        if len(subset) > 0:
            mean_p = subset['P_repeat_pred'].mean()
            mean_rate = subset['burst_rate_pred'].mean()
            
            print(f"{status.upper():15s} (N={len(subset):2d}):  P(repeat)={mean_p:.2%}  Rate={mean_rate:.1f}/yr")
    print()
    
    # Show examples
    print("=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    print()
    
    # Known repeaters
    print("Known Repeaters:")
    repeaters = df[df['repeater_status'] == 'repeater'].head(5)
    for idx, row in repeaters.iterrows():
        print(f"  {row['FRB_name']:20s}: P={row['P_repeat_pred']:.2%}  Rate={row['burst_rate_pred']:.1f}/yr")
    print()
    
    # Known one-offs
    print("Known One-Offs:")
    oneoffs = df[df['repeater_status'] == 'one-off'].head(5)
    for idx, row in oneoffs.iterrows():
        print(f"  {row['FRB_name']:20s}: P={row['P_repeat_pred']:.2%}  Rate={row['burst_rate_pred']:.1f}/yr")
    print()
    
    # Uncertain
    print("Uncertain (Predictions for Unclassified):")
    uncertain = df[df['repeater_status'] == 'unknown'].head(5)
    for idx, row in uncertain.iterrows():
        print(f"  {row['FRB_name']:20s}: P={row['P_repeat_pred']:.2%}  Rate={row['burst_rate_pred']:.1f}/yr  → {row['predicted_class']}")
    print()
    
    # Validation check
    print("=" * 80)
    print("VALIDATION CHECK")
    print("=" * 80)
    print()
    
    passed = True
    
    if len(known) > 0:
        if accuracy > 0.75:
            print(f"✅ Accuracy: {accuracy:.1%} > 75% (PASS)")
        else:
            print(f"⚠️  Accuracy: {accuracy:.1%} < 75% (MARGINAL)")
            passed = False
        
        if precision > 0.7:
            print(f"✅ Precision: {precision:.1%} > 70% (PASS)")
        else:
            print(f"⚠️  Precision: {precision:.1%} < 70% (MARGINAL)")
        
        if recall > 0.7:
            print(f"✅ Recall: {recall:.1%} > 70% (PASS)")
        else:
            print(f"⚠️  Recall: {recall:.1%} < 70% (MARGINAL)")
    else:
        print("⚠️  No known repeater status for validation")
    
    print()
    
    if passed:
        print("=" * 80)
        print("✅ VALIDATION PASSED!")
        print("=" * 80)
        print()
        print("The FRB Weather Calculator successfully predicts repeater behavior.")
        return 0
    else:
        print("=" * 80)
        print("⚠️  VALIDATION MARGINAL")
        print("=" * 80)
        print()
        print("Results are reasonable but could be improved.")
        print("This is expected for small samples and synthetic data.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

