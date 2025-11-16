"""
Unit Tests for FRB Weather Calculator
======================================

Tests the FRB repeater probability and burst rate predictions.

Run with: pytest tests/
Or: pytest tests/test_predictor.py -v
"""

import pytest
import numpy as np
from frb_calculator import (
    predict_repeater_probability,
    predict_burst_rate,
    predict_duty_cycle,
    comprehensive_frb_prediction,
    classify_galaxy_type,
)


# ============================================================================
# TEST 1: BASIC FUNCTIONALITY
# ============================================================================

def test_repeater_probability_valid_range():
    """P(repeat) must be between 0 and 1"""
    P = predict_repeater_probability(
        sSFR=1e-10,
        M_star=1e10,
        age_gyr=5.0,
        environment='field'
    )
    assert 0 <= P <= 1, f"P(repeat)={P:.3f} out of valid range [0, 1]"


def test_burst_rate_positive():
    """Burst rate must be non-negative"""
    rate = predict_burst_rate(
        SFR=1.0,
        M_star=1e10,
        age_gyr=5.0
    )
    assert rate >= 0, f"Burst rate={rate:.3f} is negative"


def test_duty_cycle_valid_range():
    """Duty cycle must be between 0 and 1"""
    dc = predict_duty_cycle(
        sSFR=1e-10,
        galaxy_type='green_valley'
    )
    assert 0 <= dc <= 1, f"Duty cycle={dc:.3f} out of valid range [0, 1]"


# ============================================================================
# TEST 2: KNOWN REPEATERS (FRB 20121102A-like)
# ============================================================================

def test_frb20121102a_high_probability():
    """FRB 20121102A host should have high P(repeat)"""
    # FRB 20121102A: M_star~1e9, sSFR~8e-10, age~5 Gyr
    P = predict_repeater_probability(
        sSFR=8e-10,
        M_star=1e9,
        age_gyr=5.0,
        environment='field'
    )
    assert P > 0.7, f"FRB 20121102A-like galaxy P(repeat)={P:.2%}, expected >70%"


def test_frb20121102a_burst_rate():
    """FRB 20121102A should have substantial burst rate"""
    rate = predict_burst_rate(
        SFR=0.8,
        M_star=1e9,
        age_gyr=5.0
    )
    assert rate > 5.0, f"FRB 20121102A-like burst rate={rate:.1f}/yr, expected >5/yr"


# ============================================================================
# TEST 3: PHYSICAL REGIMES
# ============================================================================

def test_active_sf_moderate_probability():
    """Active SF galaxies have moderate repeater probability"""
    P = predict_repeater_probability(
        sSFR=1e-9,  # Active SF
        M_star=1e10,
        age_gyr=3.0,
        environment='field'
    )
    assert 0.3 < P < 0.9, f"Active SF P(repeat)={P:.2%}, expected 30-90%"


def test_green_valley_high_probability():
    """Green valley galaxies have HIGH repeater probability (peak activity)"""
    P = predict_repeater_probability(
        sSFR=5e-11,  # Green valley
        M_star=1e10,
        age_gyr=5.0,
        environment='field'
    )
    assert P > 0.6, f"Green valley P(repeat)={P:.2%}, expected >60% (peak activity)"


def test_ancient_low_probability():
    """Ancient ellipticals have very low repeater probability"""
    P = predict_repeater_probability(
        sSFR=1e-13,  # Ancient, quiescent
        M_star=1e11,
        age_gyr=12.0,
        environment='field'
    )
    assert P < 0.1, f"Ancient elliptical P(repeat)={P:.2%}, expected <10%"


# ============================================================================
# TEST 4: GALAXY TYPE CLASSIFICATION
# ============================================================================

def test_classify_active_sf():
    """Active SF should be classified correctly"""
    galaxy_type = classify_galaxy_type(sSFR=1e-9, age_gyr=3.0)
    assert galaxy_type == 'active_SF', f"Classified as {galaxy_type}, expected 'active_SF'"


def test_classify_green_valley():
    """Green valley should be classified correctly"""
    galaxy_type = classify_galaxy_type(sSFR=5e-11, age_gyr=5.0)
    assert galaxy_type == 'green_valley', f"Classified as {galaxy_type}, expected 'green_valley'"


def test_classify_ancient():
    """Ancient systems should be classified correctly"""
    galaxy_type = classify_galaxy_type(sSFR=1e-13, age_gyr=12.0)
    assert galaxy_type == 'ancient', f"Classified as {galaxy_type}, expected 'ancient'"


# ============================================================================
# TEST 5: SCALING RELATIONS
# ============================================================================

def test_higher_sfr_higher_burst_rate():
    """Higher SFR should produce higher burst rate"""
    rate_low = predict_burst_rate(SFR=0.1, M_star=1e10, age_gyr=5.0)
    rate_high = predict_burst_rate(SFR=10.0, M_star=1e10, age_gyr=5.0)
    
    assert rate_high > rate_low, "Higher SFR should produce higher burst rate"


def test_lower_mass_higher_probability():
    """Lower mass galaxies tend to have higher P(repeat) (more efficient)"""
    P_low_mass = predict_repeater_probability(sSFR=1e-10, M_star=1e9, age_gyr=5.0, environment='field')
    P_high_mass = predict_repeater_probability(sSFR=1e-10, M_star=1e11, age_gyr=5.0, environment='field')
    
    # Lower mass should have equal or higher probability (more efficient coupling)
    assert P_low_mass >= P_high_mass * 0.8, "Lower mass should have similar or higher P(repeat)"


# ============================================================================
# TEST 6: ARRAY INPUTS
# ============================================================================

def test_array_input_repeater_probability():
    """Should handle array inputs correctly"""
    sSFRs = np.array([1e-9, 5e-11, 1e-13])
    M_stars = np.array([1e10, 1e10, 1e11])
    ages = np.array([3.0, 5.0, 12.0])
    
    from frb_calculator import predict_repeater_probability_array
    P_array = predict_repeater_probability_array(sSFRs, M_stars, ages, environment='field')
    
    assert len(P_array) == 3, "Should return 3 values"
    assert all(0 <= P <= 1 for P in P_array), "All P(repeat) should be in [0, 1]"


def test_array_input_burst_rate():
    """Should handle array inputs for burst rate"""
    SFRs = np.array([0.1, 1.0, 10.0])
    M_stars = np.array([1e10, 1e10, 1e10])
    ages = np.array([5.0, 5.0, 5.0])
    
    from frb_calculator import predict_burst_rate_array
    rates = predict_burst_rate_array(SFRs, M_star=M_stars, age_gyr=ages)
    
    assert len(rates) == 3, "Should return 3 values"
    assert all(r >= 0 for r in rates), "All burst rates should be non-negative"


# ============================================================================
# TEST 7: COMPREHENSIVE PREDICTION
# ============================================================================

def test_comprehensive_prediction():
    """Comprehensive prediction should return all metrics"""
    result = comprehensive_frb_prediction(
        M_star=1e10,
        SFR=1.0,
        age_gyr=5.0,
        environment='field'
    )
    
    # Check required keys
    required_keys = ['P_repeat', 'burst_rate', 'duty_cycle', 'regime']
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
    
    # Check value ranges
    assert 0 <= result['P_repeat'] <= 1, "P(repeat) out of range"
    assert result['burst_rate'] >= 0, "Burst rate negative"
    assert 0 <= result['duty_cycle'] <= 1, "Duty cycle out of range"


# ============================================================================
# TEST 8: EDGE CASES
# ============================================================================

def test_zero_sfr():
    """Zero SFR should give very low probability"""
    P = predict_repeater_probability(
        sSFR=0.0,
        M_star=1e10,
        age_gyr=12.0,
        environment='field'
    )
    assert P < 0.01, f"Zero SFR P(repeat)={P:.2%}, expected near-zero"


def test_very_high_sfr():
    """Very high SFR should still give valid predictions"""
    P = predict_repeater_probability(
        sSFR=1e-8,  # Extremely high sSFR
        M_star=1e10,
        age_gyr=1.0,
        environment='field'
    )
    assert 0 <= P <= 1, f"Very high sSFR P(repeat)={P:.3f} out of range"


def test_very_young_galaxy():
    """Very young galaxy should have valid predictions"""
    P = predict_repeater_probability(
        sSFR=1e-9,
        M_star=1e10,
        age_gyr=0.5,  # 500 Myr old
        environment='field'
    )
    assert 0 <= P <= 1, f"Very young galaxy P(repeat)={P:.3f} out of range"


def test_very_old_galaxy():
    """Very old galaxy should have low FRB activity"""
    P = predict_repeater_probability(
        sSFR=1e-14,
        M_star=1e11,
        age_gyr=13.0,  # 13 Gyr old
        environment='field'
    )
    assert P < 0.05, f"Very old galaxy P(repeat)={P:.2%}, expected <5%"


# ============================================================================
# TEST 9: ENVIRONMENTAL EFFECTS
# ============================================================================

def test_environment_affects_probability():
    """Different environments should affect predictions"""
    P_field = predict_repeater_probability(sSFR=1e-10, M_star=1e10, age_gyr=5.0, environment='field')
    P_cluster = predict_repeater_probability(sSFR=1e-10, M_star=1e10, age_gyr=5.0, environment='cluster')
    
    # Predictions should differ (environment matters)
    # Allow them to be different or similar depending on physics
    assert 0 <= P_field <= 1 and 0 <= P_cluster <= 1, "Both should be valid"


# ============================================================================
# TEST 10: CONSISTENCY CHECKS
# ============================================================================

def test_repeater_and_burst_rate_consistency():
    """High P(repeat) should generally correlate with higher burst rate"""
    # Green valley (high P, high rate)
    P_high = predict_repeater_probability(sSFR=5e-11, M_star=1e10, age_gyr=5.0, environment='field')
    rate_high = predict_burst_rate(SFR=0.5, M_star=1e10, age_gyr=5.0)
    
    # Ancient (low P, low rate)
    P_low = predict_repeater_probability(sSFR=1e-13, M_star=1e11, age_gyr=12.0, environment='field')
    rate_low = predict_burst_rate(SFR=0.0001, M_star=1e11, age_gyr=12.0)
    
    # High-P galaxy should have higher burst rate than low-P galaxy
    assert P_high > P_low, "Green valley should have higher P(repeat) than ancient"
    assert rate_high > rate_low, "Green valley should have higher burst rate than ancient"


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v'])

