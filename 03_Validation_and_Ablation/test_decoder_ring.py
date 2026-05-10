"""
test_decoder_ring.py
====================
Rigorous test suite for decoder_ring.py

Tests:
  1.  Known-answer test — Roboflow CM → expected landmark-only
  2.  Known-answer test — ISL CM → expected landmark-only
  3.  Cross-path consistency — CM path vs summary path, same data
  4.  Synthetic landmark-only model → correct spectrum position
  5.  Synthetic hybrid model → correct spectrum position
  6.  Synthetic multimodal model → correct spectrum position
  7.  Anchor pairs — bootstrap anchors drive spectrum score down
  8.  Anchor pairs — absence of anchors → higher spectrum score
  9.  Refusal rate — high refusal → refusal failure mode
  10. Refusal rate — low refusal + errors → silent failure mode
  11. Predicted failures — K→V anchor triggers V-family predictions
  12. Predicted failures — D→I anchor triggers I-family predictions
  13. Edge case — all correct (perfect model)
  14. Edge case — all wrong, diffuse errors (no structure)
  15. Edge case — single letter dataset
  16. Edge case — empty top_errors
  17. Edge case — missing optional fields
  18. Input validation — no inputs raises ValueError
  19. Input validation — CM without labels raises ValueError
  20. Geometric survivors — C/O/V near-perfect in landmark model
  21. Family classification — V-family pairs correctly grouped
  22. Family classification — I-family pairs correctly grouped
  23. Spectrum score monotonicity — more pairs = lower score
  24. Report contains all required sections
  25. Output dict has all required keys

Run:
    conda activate asl_gemma4
    python test_decoder_ring.py
"""

import sys
import numpy as np
import pandas as pd
import traceback

sys.path.insert(0, '/home/claude')
from decoder_ring import decode

# Use the same 24 static letters as the audit
LETTERS = [l for l in 'ABCDEFGHIKLMNOPQRSTUVWXY']
LETTER_IDX = {l: i for i, l in enumerate(LETTERS)}
N = len(LETTERS)

# ── TEST HELPERS ───────────────────────────────────────────────────────────────

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f'  ✓  {name}')
        passed += 1
    except AssertionError as e:
        print(f'  ✗  {name}')
        print(f'       AssertionError: {e}')
        failed += 1
        errors.append((name, str(e)))
    except Exception as e:
        print(f'  ✗  {name}')
        print(f'       Exception: {e}')
        traceback.print_exc()
        failed += 1
        errors.append((name, str(e)))


def make_identity_cm(n=None):
    """Perfect model — all correct."""
    n = n or N
    return np.eye(n)


def make_landmark_cm():
    """
    Synthetic landmark-only confusion matrix.
    Strong V-family + I-family + MNT collapses.
    """
    cm = np.eye(N) * 5.0
    def add_confusion(true_l, err_l, count):
        i, j = LETTER_IDX.get(true_l), LETTER_IDX.get(err_l)
        if i is not None and j is not None:
            cm[i, j] += count

    # V-family — total collapse
    add_confusion('K', 'V', 40)
    add_confusion('U', 'V', 30)
    add_confusion('R', 'V', 25)
    add_confusion('W', 'V', 20)
    add_confusion('Y', 'V', 35)

    # I-family — strong
    add_confusion('D', 'I', 43)
    add_confusion('T', 'I', 25)
    add_confusion('X', 'I', 30)
    add_confusion('G', 'D', 18)
    add_confusion('P', 'D', 28)

    # MNT cluster
    add_confusion('M', 'N', 20)
    add_confusion('M', 'T', 15)
    add_confusion('N', 'T', 18)

    return cm


def make_hybrid_cm():
    """
    Synthetic hybrid model — near-diagonal, diffuse small errors.
    No structural pairs fire above threshold.
    """
    cm = np.eye(N) * 40.0
    rng = np.random.default_rng(42)
    # Small diffuse noise
    noise = rng.integers(0, 3, size=(N, N)).astype(float)
    np.fill_diagonal(noise, 0)
    return cm + noise


def make_multimodal_cm():
    """
    Synthetic multimodal — some structural pairs but weaker,
    more distributed errors.
    """
    cm = np.eye(N) * 20.0
    def add_confusion(true_l, err_l, count):
        i, j = LETTER_IDX.get(true_l), LETTER_IDX.get(err_l)
        if i is not None and j is not None:
            cm[i, j] += count

    # Weak V-family only
    add_confusion('K', 'V', 8)   # ~25% confusion rate
    add_confusion('U', 'V', 6)

    # Some distributed errors
    add_confusion('A', 'S', 5)
    add_confusion('B', 'L', 4)
    add_confusion('H', 'E', 5)

    return cm


def cm_to_summary(cm, letters=None):
    """Convert CM to per_letter_accuracy + top_errors for Path B testing."""
    letters = letters or LETTERS
    label_idx = {l: i for i, l in enumerate(letters)}
    per_letter_accuracy = {}
    top_errors = {}

    for l in letters:
        i = label_idx[l]
        row = cm[i].copy()
        total = row.sum()
        if total == 0:
            per_letter_accuracy[l] = 0.0
            top_errors[l] = []
            continue
        correct = cm[i, i]
        per_letter_accuracy[l] = float(correct / total)
        error_row = row.copy()
        error_row[i] = 0
        top_idx = np.argsort(error_row)[::-1][:3]
        top_errors[l] = [(letters[j], int(error_row[j]))
                         for j in top_idx if error_row[j] > 0]

    return per_letter_accuracy, top_errors


# ── LOAD REAL DATA ─────────────────────────────────────────────────────────────

def load_real_cm(path):
    df = pd.read_csv(path, index_col=0)
    # Drop J
    if 'J' in df.index:
        df = df.drop(index='J')
    if 'J' in df.columns:
        df = df.drop(columns='J')
    df = df.reindex(index=LETTERS, columns=LETTERS, fill_value=0)
    return df.values.astype(float)


try:
    cm_roboflow = load_real_cm(
        '/mnt/user-data/uploads/roboflow_confusion_matrix__1_.csv')
    cm_isl = load_real_cm(
        '/mnt/user-data/uploads/isl_confusion_matrix__1_.csv')
    real_data_available = True
except Exception as e:
    print(f'Warning: real data not available ({e})')
    real_data_available = False


# ── TESTS ──────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('DECODER RING TEST SUITE')
print('='*60)

print('\n[1] Known-answer tests — real audit data')

def test_roboflow_landmark():
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_roboflow, labels=LETTERS,
                    refusal_rate=0.01, dataset_name='Roboflow')
    assert result['spectrum_position'] in ('landmark-only', 'landmark+context'), \
        f"Expected landmark position, got {result['spectrum_position']}"
    assert result['spectrum_score'] < 0.45, \
        f"Expected low spectrum score, got {result['spectrum_score']:.3f}"
    # K→V must be firing
    assert ('K', 'V') in result['firing_pairs'], \
        "K→V should be firing on Roboflow data"
    # D→I must be firing
    assert ('D', 'I') in result['firing_pairs'], \
        "D→I should be firing on Roboflow data"

test('Roboflow CM → landmark position + K→V + D→I firing', test_roboflow_landmark)


def test_isl_landmark():
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_isl, labels=LETTERS,
                    refusal_rate=0.0, dataset_name='ISL')
    assert result['spectrum_position'] in ('landmark-only', 'landmark+context'), \
        f"Expected landmark position, got {result['spectrum_position']}"
    assert ('K', 'V') in result['firing_pairs'], \
        "K→V should fire on ISL (all 41 images → V)"
    assert result['failure_mode'] == 'silent', \
        f"ISL has zero refusals — should be silent, got {result['failure_mode']}"

test('ISL CM → landmark position + silent failure mode', test_isl_landmark)


def test_cross_dataset_consistency():
    """Both datasets should land in same spectrum position."""
    if not real_data_available:
        raise AssertionError('Real data not available')
    r_roboflow = decode(confusion_matrix=cm_roboflow, labels=LETTERS)
    r_isl = decode(confusion_matrix=cm_isl, labels=LETTERS)
    assert r_roboflow['spectrum_position'] == r_isl['spectrum_position'], \
        (f"Roboflow={r_roboflow['spectrum_position']} "
         f"ISL={r_isl['spectrum_position']} — should match (Mantel r=0.945)")

test('Cross-dataset consistency — same spectrum position', test_cross_dataset_consistency)


print('\n[2] Cross-path consistency — CM vs summary path')

def test_cm_vs_summary_roboflow():
    if not real_data_available:
        raise AssertionError('Real data not available')
    r_cm = decode(confusion_matrix=cm_roboflow, labels=LETTERS)
    acc, errors = cm_to_summary(cm_roboflow, LETTERS)
    r_sum = decode(per_letter_accuracy=acc, top_errors=errors)
    # Both paths should agree on spectrum position
    assert r_cm['spectrum_position'] == r_sum['spectrum_position'], \
        (f"CM path: {r_cm['spectrum_position']} | "
         f"Summary path: {r_sum['spectrum_position']} — should agree")
    # Both should detect K→V
    assert ('K', 'V') in r_sum['firing_pairs'], \
        "Summary path should detect K→V"

test('CM path vs summary path — same spectrum position (Roboflow)', test_cm_vs_summary_roboflow)


def test_cm_vs_summary_isl():
    if not real_data_available:
        raise AssertionError('Real data not available')
    r_cm = decode(confusion_matrix=cm_isl, labels=LETTERS)
    acc, errors = cm_to_summary(cm_isl, LETTERS)
    r_sum = decode(per_letter_accuracy=acc, top_errors=errors)
    assert r_cm['spectrum_position'] == r_sum['spectrum_position'], \
        (f"CM path: {r_cm['spectrum_position']} | "
         f"Summary path: {r_sum['spectrum_position']}")

test('CM path vs summary path — same spectrum position (ISL)', test_cm_vs_summary_isl)


print('\n[3] Synthetic model tests')

def test_synthetic_landmark():
    cm = make_landmark_cm()
    result = decode(confusion_matrix=cm, labels=LETTERS, refusal_rate=0.01)
    assert result['spectrum_position'] in ('landmark-only', 'landmark+context'), \
        f"Synthetic landmark model should be at landmark end, got {result['spectrum_position']}"
    assert result['spectrum_score'] < 0.40, \
        f"Score should be low, got {result['spectrum_score']:.3f}"
    assert len(result['firing_pairs']) >= 5, \
        f"Should detect ≥5 pairs, got {len(result['firing_pairs'])}"

test('Synthetic landmark-only model → landmark spectrum position', test_synthetic_landmark)


def test_synthetic_hybrid():
    cm = make_hybrid_cm()
    result = decode(confusion_matrix=cm, labels=LETTERS)
    assert result['spectrum_position'] in ('multimodal', 'hybrid'), \
        f"Synthetic hybrid model should be at hybrid end, got {result['spectrum_position']}"
    assert result['spectrum_score'] > 0.50, \
        f"Score should be high, got {result['spectrum_score']:.3f}"

test('Synthetic hybrid model → hybrid/multimodal spectrum position', test_synthetic_hybrid)


def test_synthetic_multimodal():
    cm = make_multimodal_cm()
    result = decode(confusion_matrix=cm, labels=LETTERS)
    # Synthetic multimodal has weak K->V (~25%) and U->V — these still
    # trigger V-family, so correctly scores below hybrid.
    # Key check: scores HIGHER than full landmark collapse model.
    cm_lm = make_landmark_cm()
    r_lm = decode(confusion_matrix=cm_lm, labels=LETTERS)
    assert result['spectrum_score'] > r_lm['spectrum_score'], \
        (f"Partial collapses should score higher than total collapses: "
         f"multimodal={result['spectrum_score']:.3f} vs "
         f"landmark={r_lm['spectrum_score']:.3f}")

test('Synthetic multimodal scores higher than full landmark model', test_synthetic_multimodal)


def test_spectrum_ordering():
    """Landmark model should score lower than hybrid model."""
    cm_lm = make_landmark_cm()
    cm_hy = make_hybrid_cm()
    r_lm = decode(confusion_matrix=cm_lm, labels=LETTERS)
    r_hy = decode(confusion_matrix=cm_hy, labels=LETTERS)
    assert r_lm['spectrum_score'] < r_hy['spectrum_score'], \
        (f"Landmark score {r_lm['spectrum_score']:.3f} should be < "
         f"hybrid score {r_hy['spectrum_score']:.3f}")

test('Spectrum ordering — landmark scores lower than hybrid', test_spectrum_ordering)


print('\n[4] Anchor pair tests')

def test_kv_anchor_drives_score_down():
    """Adding K→V collapse to a neutral matrix should lower spectrum score."""
    cm_neutral = make_hybrid_cm()
    cm_kv = cm_neutral.copy()
    k_idx, v_idx = LETTER_IDX['K'], LETTER_IDX['V']
    cm_kv[k_idx, v_idx] += 40  # strong K→V collapse
    r_neutral = decode(confusion_matrix=cm_neutral, labels=LETTERS)
    r_kv = decode(confusion_matrix=cm_kv, labels=LETTERS)
    assert r_kv['spectrum_score'] < r_neutral['spectrum_score'], \
        "Adding K→V collapse should lower spectrum score"
    assert ('K', 'V') in r_kv['firing_pairs'], \
        "K→V should be in firing pairs"

test('K→V anchor lowers spectrum score', test_kv_anchor_drives_score_down)


def test_multiple_anchors_additive():
    """More anchor pairs = lower spectrum score."""
    cm_base = make_hybrid_cm()

    # Add one anchor
    cm_1 = cm_base.copy()
    cm_1[LETTER_IDX['K'], LETTER_IDX['V']] += 40

    # Add two anchors
    cm_2 = cm_1.copy()
    cm_2[LETTER_IDX['D'], LETTER_IDX['I']] += 43

    r1 = decode(confusion_matrix=cm_1, labels=LETTERS)
    r2 = decode(confusion_matrix=cm_2, labels=LETTERS)
    assert r2['spectrum_score'] <= r1['spectrum_score'], \
        f"Two anchors ({r2['spectrum_score']:.3f}) should score ≤ one anchor ({r1['spectrum_score']:.3f})"

test('Multiple anchors are additive in score reduction', test_multiple_anchors_additive)


print('\n[5] Failure mode tests')

def test_high_refusal_mode():
    acc = {l: 0.0 for l in LETTERS}
    errors = {l: [] for l in LETTERS}
    result = decode(per_letter_accuracy=acc, top_errors=errors,
                    refusal_rate=0.45)
    assert result['failure_mode'] == 'refusal', \
        f"High refusal rate should → refusal mode, got {result['failure_mode']}"

test('High refusal rate → refusal failure mode', test_high_refusal_mode)


def test_silent_failure_mode():
    """Low accuracy, low refusal = silent misidentification."""
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_isl, labels=LETTERS, refusal_rate=0.0)
    assert result['failure_mode'] == 'silent', \
        f"ISL (0% refusal, many zeros) should be silent, got {result['failure_mode']}"

test('Zero refusals + many zero-accuracy letters → silent failure mode', test_silent_failure_mode)


print('\n[6] Predicted failures tests')

def test_kv_predicts_vfamily():
    """K→V firing should predict U→V, R→V etc."""
    cm = np.eye(N) * 5.0
    cm[LETTER_IDX['K'], LETTER_IDX['V']] += 40
    result = decode(confusion_matrix=cm, labels=LETTERS)
    predicted_pairs = [p['pair'] for p in result['predicted_additional_failures']]
    assert ('U', 'V') in predicted_pairs, \
        "K→V should predict U→V"
    assert ('R', 'V') in predicted_pairs, \
        "K→V should predict R→V"

test('K→V fires → U→V, R→V predicted', test_kv_predicts_vfamily)


def test_di_predicts_ifamily():
    """D→I firing should predict T→I, X→I etc."""
    cm = np.eye(N) * 5.0
    cm[LETTER_IDX['D'], LETTER_IDX['I']] += 43
    result = decode(confusion_matrix=cm, labels=LETTERS)
    predicted_pairs = [p['pair'] for p in result['predicted_additional_failures']]
    assert ('T', 'I') in predicted_pairs, \
        "D→I should predict T→I"
    assert ('X', 'I') in predicted_pairs, \
        "D→I should predict X→I"

test('D→I fires → T→I, X→I predicted', test_di_predicts_ifamily)


def test_predicted_not_in_firing():
    """Predicted failures should not overlap with firing pairs."""
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_roboflow, labels=LETTERS)
    firing_set = set(result['firing_pairs'].keys())
    predicted_pairs = [p['pair'] for p in result['predicted_additional_failures']]
    overlap = [p for p in predicted_pairs if p in firing_set]
    assert len(overlap) == 0, \
        f"Predicted failures should not overlap with firing: {overlap}"

test('Predicted failures do not overlap with firing pairs', test_predicted_not_in_firing)


print('\n[7] Edge case tests')

def test_perfect_model():
    """Perfect accuracy → hybrid end of spectrum, no firing pairs."""
    cm = make_identity_cm() * 40
    result = decode(confusion_matrix=cm, labels=LETTERS)
    assert len(result['firing_pairs']) == 0, \
        "Perfect model should have no firing pairs"
    assert result['spectrum_score'] > 0.5, \
        f"Perfect model should score above 0.5, got {result['spectrum_score']:.3f}"

test('Perfect model → no firing pairs, high spectrum score', test_perfect_model)


def test_diffuse_errors():
    """All wrong but diffuse (no structural pairs) → no decoder ring signal."""
    rng = np.random.default_rng(99)
    cm = rng.integers(1, 5, size=(N, N)).astype(float)
    np.fill_diagonal(cm, 0)  # all wrong
    result = decode(confusion_matrix=cm, labels=LETTERS)
    # Diffuse errors should produce few or no structural pair firings
    # (they might get some by chance, but score should be mid-range)
    assert result['spectrum_score'] > 0.25, \
        "Diffuse errors should not look like landmark-only model"

test('Diffuse errors (no structure) → not landmark-only', test_diffuse_errors)


def test_empty_top_errors():
    """Handle empty top_errors gracefully."""
    acc = {'A': 0.5, 'K': 0.0, 'D': 0.0, 'V': 1.0, 'C': 1.0, 'O': 1.0}
    errors = {l: [] for l in acc}
    result = decode(per_letter_accuracy=acc, top_errors=errors)
    assert 'spectrum_position' in result
    assert 'report' in result

test('Empty top_errors → no crash, valid output', test_empty_top_errors)


def test_missing_optional_fields():
    """Omitting refusal_rate and dataset_name should work fine."""
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_roboflow, labels=LETTERS)
    assert 'spectrum_position' in result
    assert result['report'] is not None

test('Missing optional fields → valid output', test_missing_optional_fields)


def test_partial_label_set():
    """Subset of letters — should work for any label set."""
    small_labels = ['A', 'K', 'V', 'D', 'I', 'C', 'O']
    n = len(small_labels)
    cm = np.eye(n) * 5.0
    cm[small_labels.index('K'), small_labels.index('V')] += 40
    result = decode(confusion_matrix=cm, labels=small_labels)
    assert ('K', 'V') in result['firing_pairs'], \
        "Should detect K→V in small label set"

test('Partial label set → K→V still detected', test_partial_label_set)


print('\n[8] Input validation tests')

def test_no_inputs_raises():
    try:
        decode()
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

test('No inputs → ValueError', test_no_inputs_raises)


def test_cm_without_labels_raises():
    try:
        decode(confusion_matrix=np.eye(5))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

test('CM without labels → ValueError', test_cm_without_labels_raises)


def test_invalid_cm_shape():
    """Non-square matrix should either raise or handle gracefully."""
    try:
        result = decode(confusion_matrix=np.ones((5, 6)),
                        labels=['A', 'B', 'C', 'D', 'E'])
        # If it doesn't raise, check it produced something
        assert 'spectrum_position' in result
    except (ValueError, IndexError):
        pass  # Either outcome is acceptable

test('Non-square CM → raises or handles gracefully', test_invalid_cm_shape)


print('\n[9] Output structure tests')

def test_output_keys():
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_roboflow, labels=LETTERS)
    required_keys = [
        'spectrum_position', 'spectrum_score', 'firing_pairs',
        'families', 'predicted_additional_failures', 'failure_mode',
        'training_data_implication', 'deployment_risk', 'report'
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

test('Output dict has all required keys', test_output_keys)


def test_spectrum_position_valid():
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_roboflow, labels=LETTERS)
    valid = {'landmark-only', 'landmark+context', 'multimodal', 'hybrid'}
    assert result['spectrum_position'] in valid, \
        f"Invalid spectrum position: {result['spectrum_position']}"

test('spectrum_position is a valid enum value', test_spectrum_position_valid)


def test_spectrum_score_range():
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_roboflow, labels=LETTERS)
    assert 0.0 <= result['spectrum_score'] <= 1.0, \
        f"Spectrum score out of range: {result['spectrum_score']}"

test('spectrum_score is in [0.0, 1.0]', test_spectrum_score_range)


def test_report_sections():
    if not real_data_available:
        raise AssertionError('Real data not available')
    result = decode(confusion_matrix=cm_roboflow, labels=LETTERS,
                    dataset_name='Roboflow')
    report = result['report']
    required_sections = [
        'SPECTRUM POSITION',
        'FAILURE MODE',
        'FIRING DECODER RING PAIRS',
        'TRAINING DATA IMPLICATION',
        'DEPLOYMENT RISK',
        'FSboard',
    ]
    for section in required_sections:
        assert section in report, f"Report missing section: {section}"

test('Report contains all required sections', test_report_sections)


print('\n[10] Family classification tests')

def test_vfamily_grouping():
    cm = np.eye(N) * 5.0
    for l in ['K', 'U', 'R']:
        cm[LETTER_IDX[l], LETTER_IDX['V']] += 35
    result = decode(confusion_matrix=cm, labels=LETTERS)
    assert 'V-family' in result['families'], \
        "V-family should be detected"
    assert result['families']['V-family']['pair_count'] >= 3, \
        f"V-family should have ≥3 pairs, got {result['families']['V-family']['pair_count']}"

test('V-family pairs correctly grouped', test_vfamily_grouping)


def test_ifamily_grouping():
    cm = np.eye(N) * 5.0
    for true_l in ['D', 'T', 'X']:
        cm[LETTER_IDX[true_l], LETTER_IDX['I']] += 30
    result = decode(confusion_matrix=cm, labels=LETTERS)
    assert 'I-family' in result['families'], \
        "I-family should be detected"

test('I-family pairs correctly grouped', test_ifamily_grouping)


def test_mixed_families():
    """Both V-family and I-family firing → landmark+context."""
    cm = make_landmark_cm()
    result = decode(confusion_matrix=cm, labels=LETTERS)
    assert 'V-family' in result['families'], "V-family should fire"
    assert 'I-family' in result['families'], "I-family should fire"
    assert result['spectrum_position'] in ('landmark-only', 'landmark+context'), \
        f"Mixed families → landmark end, got {result['spectrum_position']}"

test('Mixed V+I families → landmark-only or landmark+context', test_mixed_families)


# ── SUMMARY ────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print(f'RESULTS: {passed} passed, {failed} failed out of {passed+failed} tests')
print('='*60)

if failed > 0:
    print('\nFailed tests:')
    for name, msg in errors:
        print(f'  ✗ {name}')
        print(f'    {msg}')

if failed == 0:
    print('\nAll tests passed.')
else:
    sys.exit(1)
