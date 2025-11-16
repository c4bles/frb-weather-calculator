#!/usr/bin/env python3
"""
FRB Weather Calculator - Basic Usage Examples
==============================================

Simple examples demonstrating how to predict FRB activity.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from frb_calculator import (
    predict_repeater_probability,
    predict_burst_rate,
    predict_duty_cycle,
    comprehensive_frb_prediction,
)


print("=" * 80)
print("FRB WEATHER CALCULATOR - BASIC USAGE EXAMPLES")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: Known Repeater (FRB 20121102A-like)
# ============================================================================

print("EXAMPLE 1: FRB 20121102A-like host (known repeater)")
print("-" * 80)

P_repeat = predict_repeater_probability(
    sSFR=8e-10,           # Active star formation
    M_star=1e9,           # 1 billion solar masses
    age_gyr=5.0,          # 5 Gyr old
    environment='field'   # Field galaxy
)

burst_rate = predict_burst_rate(
    SFR=0.8,              # 0.8 M☉/yr
    M_star=1e9,
    age_gyr=5.0
)

duty_cycle = predict_duty_cycle(
    sSFR=8e-10,
    galaxy_type='green_valley'  # Based on sSFR
)

print(f"Host properties:")
print(f"  M_star  = 10^9 M☉")
print(f"  SFR     = 0.8 M☉/yr")
print(f"  sSFR    = 8×10^-10 yr^-1")
print(f"  Age     = 5 Gyr")
print()
print(f"FRB Predictions:")
print(f"  P(repeat):     {P_repeat:.1%}")
print(f"  Burst rate:    {burst_rate:.1f} bursts/year")
print(f"  Duty cycle:    {duty_cycle:.1%}")
print()
print(f"Interpretation: HIGH repeater probability (green valley regime)")
print()
print()

# ============================================================================
# EXAMPLE 2: Ancient Elliptical (No FRBs Expected)
# ============================================================================

print("EXAMPLE 2: Ancient elliptical (no FRBs expected)")
print("-" * 80)

P_repeat = predict_repeater_probability(
    sSFR=1e-13,           # Quiescent
    M_star=1e11,          # Massive elliptical
    age_gyr=12.0,         # Ancient
    environment='field'
)

burst_rate = predict_burst_rate(
    SFR=0.0001,           # Negligible SF
    M_star=1e11,
    age_gyr=12.0
)

print(f"Host properties:")
print(f"  M_star  = 10^11 M☉")
print(f"  SFR     = 0.0001 M☉/yr")
print(f"  sSFR    = 10^-13 yr^-1")
print(f"  Age     = 12 Gyr")
print()
print(f"FRB Predictions:")
print(f"  P(repeat):     {P_repeat:.1%}")
print(f"  Burst rate:    {burst_rate:.3f} bursts/year")
print()
print(f"Interpretation: Near-ZERO probability (all magnetars dead)")
print()
print()

# ============================================================================
# EXAMPLE 3: Active Star-Forming Galaxy
# ============================================================================

print("EXAMPLE 3: Active star-forming galaxy")
print("-" * 80)

P_repeat = predict_repeater_probability(
    sSFR=1e-9,            # Active SF
    M_star=1e10,
    age_gyr=3.0,          # Young
    environment='field'
)

burst_rate = predict_burst_rate(
    SFR=10.0,             # High SFR
    M_star=1e10,
    age_gyr=3.0
)

print(f"Host properties:")
print(f"  M_star  = 10^10 M☉")
print(f"  SFR     = 10 M☉/yr")
print(f"  sSFR    = 10^-9 yr^-1")
print(f"  Age     = 3 Gyr")
print()
print(f"FRB Predictions:")
print(f"  P(repeat):     {P_repeat:.1%}")
print(f"  Burst rate:    {burst_rate:.1f} bursts/year")
print()
print(f"Interpretation: MODERATE probability (young magnetars forming)")
print()
print()

# ============================================================================
# EXAMPLE 4: All Predictions at Once
# ============================================================================

print("EXAMPLE 4: All predictions for one galaxy")
print("-" * 80)

# FRB 20180916B-like host
sSFR_4 = 7.5e-10
M_star_4 = 2e9
SFR_4 = sSFR_4 * M_star_4
age_4 = 4.5

P_4 = predict_repeater_probability(sSFR_4, M_star_4, age_4, 'field')
rate_4 = predict_burst_rate(SFR=SFR_4, M_star=M_star_4, age_gyr=age_4)
dc_4 = predict_duty_cycle(sSFR=sSFR_4, galaxy_type='green_valley')

print(f"Host: M=2×10^9 M☉, SFR=1.5 M☉/yr, sSFR=7.5×10^-10 yr^-1, age=4.5 Gyr")
print()
print(f"Predictions:")
print(f"  P(repeat):       {P_4:.1%}")
print(f"  Burst rate:      {rate_4:.1f} /yr")
print(f"  Duty cycle:      {dc_4:.1%}")
print()
print()

# ============================================================================
# EXAMPLE 5: Batch Processing
# ============================================================================

print("EXAMPLE 5: Batch prediction for multiple FRB hosts")
print("-" * 80)

# Properties for 5 FRB hosts
sSFRs_batch = np.array([8e-10, 5e-11, 1e-13, 2e-9, 3e-11])
M_stars_batch = np.array([1e9, 1e10, 1e11, 5e9, 2e10])
ages_batch = np.array([5.0, 5.0, 12.0, 2.0, 6.0])
SFRs_batch = sSFRs_batch * M_stars_batch

# Predict for each
print(f"{'FRB':<6s} {'M_star':<12s} {'sSFR':<12s} {'Age':>6s} {'P(repeat)':>12s} {'Rate':>10s}")
print("-" * 80)
for i in range(len(sSFRs_batch)):
    P_i = predict_repeater_probability(sSFRs_batch[i], M_stars_batch[i], ages_batch[i], 'field')
    rate_i = predict_burst_rate(SFR=SFRs_batch[i], M_star=M_stars_batch[i], age_gyr=ages_batch[i])
    m_str = f"{M_stars_batch[i]:.1e}"
    s_str = f"{sSFRs_batch[i]:.1e}"
    print(f"FRB {i+1:<3d} {m_str:<12s} {s_str:<12s} {ages_batch[i]:>6.1f} {P_i:>11.1%} {rate_i:>9.1f}/yr")
print()
print()

# ============================================================================
# EXAMPLE 6: Different Galaxy Regimes (Evolutionary Stages)
# ============================================================================

print("EXAMPLE 6: FRB activity across galaxy evolutionary stages")
print("-" * 80)

regimes = [
    ("Active SF", 1e-9, 3.0, "Young magnetars forming"),
    ("Green Valley ⭐", 5e-11, 5.0, "PEAK activity (dying magnetars)"),
    ("Post-Green", 5e-12, 7.0, "Declining activity"),
    ("Ancient", 1e-13, 12.0, "All magnetars dead"),
]

print(f"Same galaxy mass (M=10^10 M☉) at different stages:")
print()

for regime_name, sSFR_reg, age_reg, description in regimes:
    P_reg = predict_repeater_probability(sSFR_reg, 1e10, age_reg, 'field')
    rate_reg = predict_burst_rate(SFR=sSFR_reg * 1e10, M_star=1e10, age_gyr=age_reg)
    
    print(f"  {regime_name:<20s}: P={P_reg:>6.1%}  Rate={rate_reg:>6.1f}/yr  ({description})")

print()
print(f"⭐ Green Valley = Optimal zone for FRB production!")
print()

# ============================================================================
# DONE
# ============================================================================

print("=" * 80)
print("✅ Examples complete!")
print("=" * 80)
print()
print("For validation, run:")
print("  python examples/validate_predictions.py")
print()

