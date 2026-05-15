"""
decoder_ring.py
===============
ASL Confusion Matrix Decoder Ring
Diagnostic function for auditing vision model training data
from confusion structure alone.

Based on:
  - FSboard: Georg et al. 2024 (CC BY 4.0, arXiv:2407.15806)
  - Gemma 4 E4B audit: Roboflow ASL + ISL datasets
  - Mantel test validation: r=0.945 cross-dataset, p<0.001

Two input paths converge on the same diagnostic output:

  Path A — Confusion matrix:
      result = decode(confusion_matrix=cm, labels=labels)

  Path B — Per-letter summary (output of any inference run):
      result = decode(
          per_letter_accuracy={'A': 0.25, 'K': 0.0, ...},
          top_errors={'K': [('V', 41)], 'D': [('I', 43)], ...},
          refusal_rate=0.05,          # optional
          dataset_name='MyDataset'    # optional
      )

Output:
  {
    'spectrum_position': str,
    'spectrum_score': float,          # 0.0=landmark-only, 1.0=hybrid
    'firing_pairs': list,
    'families': dict,
    'predicted_additional_failures': list,
    'failure_mode': str,
    'training_data_implication': str,
    'deployment_risk': str,
    'report': str
  }

Dependencies: numpy, pandas (optional for CM path)
"""

import textwrap
import numpy as np
from typing import Optional, Dict, List, Tuple, Union


# ── KNOWLEDGE BASE ─────────────────────────────────────────────────────────────
# All thresholds calibrated against Gemma 4 E4B audit data:
# Roboflow ASL (1,185 images) + ISL (1,185 images), 24 static letters.
# Mantel r=0.945, p<0.001 confirms cross-dataset stability.

# Decoder ring pairs organised by family.
# Each entry: (true_letter, error_target, family, landmark_percentile, mechanism)
#
# landmark_percentile: position in MediaPipe pairwise distance distribution
#   Low (<10th) = geometrically similar = PREDICTED by landmark models
#   High (>50th) = geometrically distant = NOT predicted, implies signer-context
#
DECODER_RING_PAIRS = [
    # ── V-family: geometrically predicted ─────────────────────────────────────
    # Letters that share two-finger extension geometry in landmark space
    ('K', 'V', 'V-family',  2.3,  'geometric'),
    ('U', 'V', 'V-family',  3.8,  'geometric'),
    ('R', 'V', 'V-family',  4.1,  'geometric'),
    ('W', 'V', 'V-family',  5.2,  'geometric'),
    ('Y', 'V', 'V-family',  9.7,  'geometric'),

    # ── I-family: signer-context driven ───────────────────────────────────────
    # Letters geometrically distant but confused — implies training on full
    # signer context where these look similar in full-body framing
    ('D', 'I', 'I-family',  52.0, 'signer-context'),
    ('T', 'I', 'I-family',  96.7, 'signer-context'),
    ('X', 'I', 'I-family',  71.0, 'signer-context'),
    ('G', 'I', 'I-family',  75.7, 'signer-context'),
    ('P', 'I', 'I-family',  44.0, 'signer-context'),
    ('G', 'D', 'I-family',  60.0, 'signer-context'),  # G collapses to D en route to I
    ('P', 'D', 'I-family',  48.0, 'signer-context'),  # P collapses to D
    ('T', 'D', 'I-family',  88.0, 'signer-context'),  # T→D documented in both datasets

    # ── M/N/T cluster: occlusion-driven ───────────────────────────────────────
    # Thumb placement and finger occlusion — FSboard documented
    ('M', 'N', 'MNT-cluster', 35.0, 'occlusion'),
    ('M', 'T', 'MNT-cluster', 42.0, 'occlusion'),
    ('N', 'T', 'MNT-cluster', 38.0, 'occlusion'),
    ('M', 'A', 'MNT-cluster', 28.0, 'occlusion'),  # fist family bleed

    # ── A/S cluster: closed-fist variations ───────────────────────────────────
    ('A', 'S', 'AS-cluster',  22.0, 'geometric'),
    ('S', 'A', 'AS-cluster',  22.0, 'geometric'),
    ('A', 'E', 'AS-cluster',  31.0, 'geometric'),
    ('E', 'W', 'AS-cluster',  55.0, 'signer-context'),  # E→W documented in audit
]

# Anchor pairs — confirmed at 100% persistence in bootstrap jiggle test
# These are the strongest signal; presence alone is diagnostic
BOOTSTRAP_ANCHORS = {('K', 'V'), ('D', 'I'), ('G', 'D'), ('E', 'W'), ('P', 'D')}

# Family weights for spectrum scoring
# Higher = more diagnostic of landmark-only training
FAMILY_WEIGHTS = {
    'V-family':    1.0,   # strongest landmark signal
    'I-family':    0.85,  # signer-context, also landmark-adjacent
    'MNT-cluster': 0.70,  # occlusion, FSboard documented
    'AS-cluster':  0.50,  # weaker signal, also appears in full models
}

# Spectrum thresholds (0.0 = landmark-only, 1.0 = hybrid)
SPECTRUM_THRESHOLDS = {
    'landmark-only':       (0.0,  0.30),
    'landmark+context':    (0.30, 0.55),
    'multimodal':          (0.55, 0.80),
    'hybrid':              (0.80, 1.01),
}

# Firing thresholds
# Path A (confusion matrix): fraction of row that lands on error target
CM_FIRING_THRESHOLD = 0.20      # ≥20% of predictions for true_letter → error_target
CM_STRONG_THRESHOLD = 0.50      # ≥50% = strong/dominant collapse
CM_TOTAL_COLLAPSE   = 0.85      # ≥85% = total collapse (K→V territory)

# Path B (per-letter summary): error count relative to total predictions
ACCURACY_COLLAPSE_THRESHOLD = 0.10   # acc ≤ 10% triggers pair check
ACCURACY_ZERO_THRESHOLD     = 0.02   # acc ≤ 2% = near-total failure

# Refusal rate interpretation
REFUSAL_HIGH = 0.30    # ≥30% = resolution failure (Sign MNIST pattern)
REFUSAL_LOW  = 0.05    # ≤5%  = confident misidentification pattern

# Letters that should be near-perfect in any model with real vision
# (geometrically isolated: C, O, V)
GEOMETRIC_SURVIVORS = {'C', 'O', 'V'}

# Predicted additional failures given anchor pair firing
# If anchor fires → these related pairs are likely also present
PREDICTED_FROM_ANCHOR = {
    ('K', 'V'): [('U', 'V'), ('R', 'V'), ('W', 'V'), ('Y', 'V')],
    ('D', 'I'): [('T', 'I'), ('X', 'I'), ('G', 'I'), ('T', 'D'), ('P', 'D')],
    ('G', 'D'): [('D', 'I'), ('G', 'I')],
    ('P', 'D'): [('D', 'I'), ('P', 'I')],
    ('E', 'W'): [('F', 'W'), ('H', 'E')],
    ('M', 'N'): [('M', 'T'), ('N', 'T')],
    ('A', 'S'): [('A', 'E'), ('S', 'E')],
}

# Deployment risk descriptions
DEPLOYMENT_RISK_TEXT = {
    'landmark-only': (
        "HIGH — Silent systematic failure. The model will confidently "
        "misidentify specific letters at near-zero accuracy with no refusal signal. "
        "K, D, G, P, T, X, and Y are at highest risk. "
        "Not suitable for clinical, educational, or assistive technology deployment "
        "without explicit user warnings and per-letter performance documentation."
    ),
    'landmark+context': (
        "MODERATE-HIGH — Structured failure with signer-context dependency. "
        "Performance will degrade significantly on signers whose presentation "
        "differs from training distribution. Specific letter pairs will fail "
        "systematically. Deployment requires per-letter accuracy disclosure."
    ),
    'multimodal': (
        "MODERATE — Errors are less letter-specific but still present. "
        "Failure mode shifts from structural pairs toward sequence-level ambiguity. "
        "More robust to signer variation but temporal/contextual errors will occur. "
        "Deployment feasible with appropriate accuracy documentation."
    ),
    'hybrid': (
        "LOWER — Near-diagonal confusion structure. Classic confusion pairs "
        "largely resolved. Remaining errors are context-driven and distributed. "
        "Deployment appropriate with standard accuracy disclosure."
    ),
}

TRAINING_IMPLICATION_TEXT = {
    'landmark-only': (
        "Model confusion structure matches FSboard landmark-only fingerprint "
        "(Georg et al. 2024). Strongly implies training on 21-point hand skeleton "
        "data (e.g. MediaPipe keypoints) rather than raw image pixels. "
        "V-family collapses (K→V) are geometrically inevitable for any model "
        "operating on hand landmark coordinates. I-family collapses (D→I, T→I) "
        "additionally suggest training on full-signer context images where these "
        "letters appear visually similar."
    ),
    'landmark+context': (
        "Model shows mixed landmark and signer-context failure signatures. "
        "Likely trained on a combination of landmark data and full-signer images, "
        "or on images where both geometric and contextual cues were available. "
        "The I-family collapses (geometrically distant but confused) point to "
        "signer-context training data as a significant component."
    ),
    'multimodal': (
        "Model confusion structure is less blocky than landmark-only systems. "
        "Consistent with full visual or multimodal training (hands + face + body "
        "or raw RGB). Classic confusion pairs are reduced but not eliminated. "
        "Errors suggest the model has richer visual representations but may still "
        "lack temporal or contextual information for disambiguation."
    ),
    'hybrid': (
        "Model confusion structure approaches FSboard hybrid fingerprint "
        "(RGB + keypoints). Classic confusion pairs largely absent. "
        "Consistent with training on rich multimodal data with temporal context. "
        "Remaining errors are likely sequence-level and context-dependent."
    ),
}


# ── PATH A: CONFUSION MATRIX ───────────────────────────────────────────────────

def _extract_pairs_from_cm(
    cm: np.ndarray,
    labels: List[str]
) -> Dict[Tuple[str, str], float]:
    """
    Extract firing pair rates from a confusion matrix.
    Returns dict of {(true_letter, error_target): confusion_rate}
    where confusion_rate = cm[true_idx, error_idx] / row_sum
    """
    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)

    # Row-normalize
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    firing = {}
    for true_l, err_l, family, lm_pct, mechanism in DECODER_RING_PAIRS:
        if true_l not in label_idx or err_l not in label_idx:
            continue
        i = label_idx[true_l]
        j = label_idx[err_l]
        rate = cm_norm[i, j]
        if rate >= CM_FIRING_THRESHOLD:
            firing[(true_l, err_l)] = float(rate)

    return firing


def _extract_accuracy_from_cm(
    cm: np.ndarray,
    labels: List[str]
) -> Tuple[Dict[str, float], Dict[str, List[Tuple[str, int]]]]:
    """Extract per-letter accuracy and top errors from confusion matrix."""
    label_idx = {l: i for i, l in enumerate(labels)}
    per_letter_accuracy = {}
    top_errors = {}

    for l in labels:
        i = label_idx[l]
        row = cm[i]
        total = row.sum()
        if total == 0:
            per_letter_accuracy[l] = 0.0
            top_errors[l] = []
            continue
        correct = cm[i, i]
        per_letter_accuracy[l] = float(correct / total)
        # Top 3 errors (excluding correct)
        error_row = row.copy()
        error_row[i] = 0
        top_idx = np.argsort(error_row)[::-1][:3]
        top_errors[l] = [
            (labels[j], int(error_row[j]))
            for j in top_idx if error_row[j] > 0
        ]

    return per_letter_accuracy, top_errors


# ── PATH B: PER-LETTER SUMMARY ─────────────────────────────────────────────────

def _extract_pairs_from_summary(
    per_letter_accuracy: Dict[str, float],
    top_errors: Dict[str, List[Tuple[str, int]]]
) -> Dict[Tuple[str, str], float]:
    """
    Extract firing pairs from per-letter accuracy + top errors.

    Confidence mirrors the CM path: estimated confusion rate for the pair.
    For true_letter L with accuracy `acc` and top error (E, count):
        estimated_total = error_count_sum / max(1 - acc, epsilon)
        confusion_rate  = count / estimated_total
        confidence      = confusion_rate * rank_weight

    rank_weight: top error = 1.0, second = 0.7, third = 0.5
    This makes Path B confidence directly comparable to Path A's
    normalized confusion matrix rate.
    """
    firing = {}
    known_pairs = {(tl, el): (fam, lm, mech)
                   for tl, el, fam, lm, mech in DECODER_RING_PAIRS}

    RANK_WEIGHTS = [1.0, 0.7, 0.5]
    EPSILON = 1e-6

    for true_l, errors in top_errors.items():
        acc = per_letter_accuracy.get(true_l, 1.0)

        if acc > (1.0 - ACCURACY_COLLAPSE_THRESHOLD):
            continue

        total_errors = sum(count for _, count in errors)
        if total_errors == 0:
            continue

        # Estimate total predictions from error counts and accuracy
        # If acc=0.0, all predictions are errors → estimated_total = total_errors
        # If acc=0.2, errors represent 80% → estimated_total = total_errors / 0.8
        estimated_total = total_errors / max(1.0 - acc, EPSILON)

        for rank, (err_l, count) in enumerate(errors):
            pair = (true_l, err_l)
            if pair not in known_pairs:
                continue

            rank_weight = RANK_WEIGHTS[rank] if rank < 3 else 0.3
            confusion_rate = count / estimated_total
            confidence = min(confusion_rate * rank_weight, 1.0)

            if confidence >= CM_FIRING_THRESHOLD * 0.5:
                # Slightly lower threshold for summary path since we're estimating
                firing[pair] = float(confidence)

    return firing


# ── CORE DIAGNOSTIC LOGIC ──────────────────────────────────────────────────────

def _classify_firing_pairs(
    firing: Dict[Tuple[str, str], float]
) -> Dict[str, Dict]:
    """
    Group firing pairs by family and compute family-level signal strength.
    """
    family_pairs = {}
    pair_lookup = {(tl, el): (fam, lm, mech)
                   for tl, el, fam, lm, mech in DECODER_RING_PAIRS}

    for pair, confidence in firing.items():
        if pair not in pair_lookup:
            continue
        fam, lm_pct, mech = pair_lookup[pair]
        if fam not in family_pairs:
            family_pairs[fam] = {
                'pairs': [],
                'mean_confidence': 0.0,
                'max_confidence': 0.0,
                'mechanism': mech,
                'landmark_percentiles': [],
                'anchor_count': 0,
            }
        family_pairs[fam]['pairs'].append({
            'pair': pair,
            'confidence': confidence,
            'landmark_percentile': lm_pct,
            'is_anchor': pair in BOOTSTRAP_ANCHORS,
            'mechanism': mech,
        })
        if pair in BOOTSTRAP_ANCHORS:
            family_pairs[fam]['anchor_count'] += 1
        family_pairs[fam]['landmark_percentiles'].append(lm_pct)

    # Compute summary stats per family
    for fam in family_pairs:
        confs = [p['confidence'] for p in family_pairs[fam]['pairs']]
        family_pairs[fam]['mean_confidence'] = float(np.mean(confs))
        family_pairs[fam]['max_confidence'] = float(np.max(confs))
        family_pairs[fam]['pair_count'] = len(confs)

    return family_pairs


def _compute_spectrum_score(
    families: Dict[str, Dict],
    per_letter_accuracy: Optional[Dict[str, float]],
    refusal_rate: float
) -> float:
    """
    Compute spectrum score: 0.0 = landmark-only, 1.0 = hybrid.

    Score is driven DOWN by:
      - Strong V-family firing (geometric pairs)
      - Strong I-family firing (signer-context pairs)
      - MNT-cluster firing
      - Bootstrap anchors firing
      - Low accuracy on structurally complex letters
      - Low refusal rate with wrong answers (confident misidentification)

    Score is driven UP by:
      - Absence of structural pairs
      - Diffuse errors (no dominant target)
      - Geometric survivors performing well (C, O, V at high acc)
      - High refusal rate on ambiguous images
    """
    # Start at midpoint
    score = 0.5

    if not families:
        # No structural pairs firing — pull toward multimodal/hybrid
        score += 0.25
    else:
        # Each firing family pulls score down proportional to weight and confidence
        total_pull = 0.0
        for fam, data in families.items():
            weight = FAMILY_WEIGHTS.get(fam, 0.5)
            confidence = data['mean_confidence']
            anchor_bonus = 0.1 * data['anchor_count']
            pull = weight * confidence + anchor_bonus
            total_pull += pull

        # Normalize pull against the maximum possible pull if ALL known pairs fired
        # at full confidence. This keeps the score proportional rather than flooring
        # whenever a handful of strong pairs fire simultaneously.
        max_possible_pull = sum(
            FAMILY_WEIGHTS.get(fam, 0.5) + 0.1 * sum(
                1 for tl, el, f, _, _ in DECODER_RING_PAIRS
                if f == fam and (tl, el) in BOOTSTRAP_ANCHORS
            )
            for fam in FAMILY_WEIGHTS
        )
        # Pull can reduce score by at most 0.5 (from 0.5 to 0.0)
        normalized_pull = min((total_pull / max_possible_pull) * 0.5, 0.5)
        score -= normalized_pull

    # Geometric survivor check
    if per_letter_accuracy:
        survivor_accs = [
            per_letter_accuracy.get(l, 0.0)
            for l in GEOMETRIC_SURVIVORS
            if l in per_letter_accuracy
        ]
        if survivor_accs:
            mean_survivor = np.mean(survivor_accs)
            # Survivors performing well is expected even in landmark models
            # Only push score up if survivors are perfect AND no structural pairs
            if mean_survivor > 0.95 and not families:
                score += 0.10
            elif mean_survivor < 0.5:
                # Even geometric survivors failing → very unusual, pull toward hybrid
                score += 0.15

    # Refusal rate signal
    if refusal_rate > REFUSAL_HIGH:
        # High refusal = resolution failure, not the same as structural pairs
        # Don't penalize the score but note it
        pass
    elif refusal_rate < REFUSAL_LOW and families:
        # Low refusal + structural pairs = confident misidentification
        # This is the most dangerous combination — pull score down slightly more
        score -= 0.05

    return float(np.clip(score, 0.0, 1.0))


def _place_on_spectrum(score: float) -> str:
    for position, (low, high) in SPECTRUM_THRESHOLDS.items():
        if low <= score < high:
            return position
    return 'hybrid'


def _get_predicted_failures(
    firing: Dict[Tuple[str, str], float]
) -> List[Dict]:
    """
    Given firing anchor pairs, predict additional likely failures
    not yet confirmed in the data.
    """
    predicted = []
    seen = set(firing.keys())

    for anchor, predictions in PREDICTED_FROM_ANCHOR.items():
        if anchor in firing:
            for pred_pair in predictions:
                if pred_pair not in seen:
                    predicted.append({
                        'pair': pred_pair,
                        'triggered_by': anchor,
                        'rationale': (
                            f'{pred_pair[0]}→{pred_pair[1]} predicted because '
                            f'{anchor[0]}→{anchor[1]} is firing and these share '
                            f'the same confusion family'
                        )
                    })
                    seen.add(pred_pair)

    return predicted


def _classify_failure_mode(
    per_letter_accuracy: Dict[str, float],
    refusal_rate: float
) -> str:
    if refusal_rate > REFUSAL_HIGH:
        return 'refusal'
    zero_acc_count = sum(
        1 for acc in per_letter_accuracy.values()
        if acc <= ACCURACY_ZERO_THRESHOLD
    )
    total = len(per_letter_accuracy)
    if zero_acc_count > total * 0.3 and refusal_rate < REFUSAL_LOW:
        return 'silent'
    if refusal_rate > 0.10:
        return 'mixed'
    return 'silent'


def _build_report(
    dataset_name: str,
    spectrum_position: str,
    spectrum_score: float,
    firing_pairs: Dict,
    families: Dict,
    predicted: List,
    failure_mode: str,
    per_letter_accuracy: Dict,
    refusal_rate: float,
) -> str:
    lines = []
    lines.append('=' * 65)
    lines.append('DECODER RING — ASL MODEL TRAINING DATA AUDIT')
    lines.append('=' * 65)
    if dataset_name:
        lines.append(f'Dataset: {dataset_name}')

    lines.append(f'\nSPECTRUM POSITION: {spectrum_position.upper()}')
    lines.append(f'Spectrum score: {spectrum_score:.3f}  '
                 f'(0.0=landmark-only → 1.0=hybrid)')

    lines.append(f'\nFAILURE MODE: {failure_mode.upper()}')
    if failure_mode == 'silent':
        lines.append('  Confident misidentification — model sees inputs and is '
                     'wrong without refusing.')
    elif failure_mode == 'refusal':
        lines.append('  Resolution failure — model cannot process image quality '
                     'and refuses rather than guessing.')
    else:
        lines.append('  Mixed — some refusals, some confident errors.')

    if refusal_rate is not None:
        lines.append(f'  Refusal rate: {refusal_rate:.1%}')

    # Geometric survivors
    if per_letter_accuracy:
        survivors = {l: per_letter_accuracy[l]
                     for l in GEOMETRIC_SURVIVORS
                     if l in per_letter_accuracy}
        if survivors:
            lines.append(f'\nGEOMETRIC SURVIVORS (expected near-perfect):')
            for l, acc in survivors.items():
                status = '✓' if acc > 0.90 else '⚠'
                lines.append(f'  {status} {l}: {acc:.1%}')

    # Firing pairs
    lines.append(f'\nFIRING DECODER RING PAIRS ({len(firing_pairs)}):')
    if not firing_pairs:
        lines.append('  None detected — confusion structure does not match '
                     'known landmark model fingerprint.')
    else:
        pair_lookup = {(tl, el): (fam, lm, mech)
                       for tl, el, fam, lm, mech in DECODER_RING_PAIRS}
        for pair, conf in sorted(firing_pairs.items(),
                                 key=lambda x: -x[1]):
            fam, lm_pct, mech = pair_lookup.get(pair, ('unknown', 0, 'unknown'))
            anchor_flag = ' [ANCHOR]' if pair in BOOTSTRAP_ANCHORS else ''
            lines.append(
                f'  {pair[0]}→{pair[1]:<3} | '
                f'confidence={conf:.2f} | '
                f'{fam} | '
                f'LM-pct={lm_pct:.0f}th | '
                f'{mech}{anchor_flag}'
            )

    # Families
    if families:
        lines.append(f'\nACTIVE FAMILIES:')
        for fam, data in sorted(families.items(),
                                key=lambda x: -x[1]['mean_confidence']):
            lines.append(
                f'  {fam}: {data["pair_count"]} pairs, '
                f'mean confidence={data["mean_confidence"]:.2f}, '
                f'anchors={data["anchor_count"]}'
            )

    # Predicted failures
    if predicted:
        lines.append(f'\nPREDICTED ADDITIONAL FAILURES '
                     f'(not yet confirmed, inferred from firing pairs):')
        for p in predicted:
            lines.append(f'  {p["pair"][0]}→{p["pair"][1]}: {p["rationale"]}')

    # Training implication
    lines.append(f'\nTRAINING DATA IMPLICATION:')
    impl = TRAINING_IMPLICATION_TEXT.get(spectrum_position, '')
    for line in textwrap.wrap(impl, width=63):
        lines.append(f'  {line}')

    # Deployment risk
    lines.append(f'\nDEPLOYMENT RISK:')
    risk = DEPLOYMENT_RISK_TEXT.get(spectrum_position, '')
    for line in textwrap.wrap(risk, width=63):
        lines.append(f'  {line}')

    lines.append('\n' + '=' * 65)
    lines.append('Reference: FSboard (Georg et al. 2024, CC BY 4.0, '
                 'arXiv:2407.15806)')
    lines.append('Validation: Mantel r=0.945 p<0.001 (Roboflow vs ISL, '
                 'n=9999 permutations)')
    lines.append('=' * 65)

    return '\n'.join(lines)


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def decode(
    # Path A — confusion matrix
    confusion_matrix: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    # Path B — per-letter summary
    per_letter_accuracy: Optional[Dict[str, float]] = None,
    top_errors: Optional[Dict[str, List[Tuple[str, int]]]] = None,
    # Shared optional
    refusal_rate: float = 0.0,
    dataset_name: str = '',
) -> Dict:
    """
    Decode a model's confusion structure into a training data audit.

    Parameters
    ----------
    Path A (confusion matrix):
        confusion_matrix : np.ndarray, shape (n, n)
            Raw or normalized confusion matrix. Rows = true, cols = predicted.
        labels : list of str
            Letter labels aligned to matrix axes. Must exclude J, Z.

    Path B (per-letter summary):
        per_letter_accuracy : dict {letter: float}
            Per-letter accuracy, e.g. {'A': 0.25, 'K': 0.0, ...}
        top_errors : dict {letter: [(error_letter, count), ...]}
            Top error targets per letter, e.g. {'K': [('V', 41), ('U', 1)]}

    Shared:
        refusal_rate : float
            Fraction of inputs where model refused/gave invalid output.
            Default 0.0.
        dataset_name : str
            Label for the report. Default ''.

    Returns
    -------
    dict with keys:
        spectrum_position, spectrum_score, firing_pairs, families,
        predicted_additional_failures, failure_mode,
        training_data_implication, deployment_risk, report
    """
    # ── Input validation ───────────────────────────────────────────────────────
    has_cm = confusion_matrix is not None
    has_summary = per_letter_accuracy is not None and top_errors is not None

    if not has_cm and not has_summary:
        raise ValueError(
            'Provide either (confusion_matrix + labels) '
            'or (per_letter_accuracy + top_errors).'
        )

    # ── Extract firing pairs ───────────────────────────────────────────────────
    if has_cm:
        if labels is None:
            raise ValueError('labels required when providing confusion_matrix.')
        cm = np.array(confusion_matrix, dtype=float)
        firing = _extract_pairs_from_cm(cm, labels)
        per_letter_accuracy, top_errors = _extract_accuracy_from_cm(cm, labels)
    else:
        firing = _extract_pairs_from_summary(per_letter_accuracy, top_errors)

    # ── Classify families ──────────────────────────────────────────────────────
    families = _classify_firing_pairs(firing)

    # ── Compute spectrum score ─────────────────────────────────────────────────
    spectrum_score = _compute_spectrum_score(
        families, per_letter_accuracy, refusal_rate)
    spectrum_position = _place_on_spectrum(spectrum_score)

    # ── Predicted additional failures ──────────────────────────────────────────
    predicted = _get_predicted_failures(firing)

    # ── Failure mode ───────────────────────────────────────────────────────────
    failure_mode = _classify_failure_mode(per_letter_accuracy, refusal_rate)

    # ── Build report ───────────────────────────────────────────────────────────
    report = _build_report(
        dataset_name=dataset_name,
        spectrum_position=spectrum_position,
        spectrum_score=spectrum_score,
        firing_pairs=firing,
        families=families,
        predicted=predicted,
        failure_mode=failure_mode,
        per_letter_accuracy=per_letter_accuracy,
        refusal_rate=refusal_rate,
    )

    return {
        'spectrum_position':            spectrum_position,
        'spectrum_score':               spectrum_score,
        'firing_pairs':                 firing,
        'families':                     families,
        'predicted_additional_failures': predicted,
        'failure_mode':                 failure_mode,
        'training_data_implication':    TRAINING_IMPLICATION_TEXT[spectrum_position],
        'deployment_risk':              DEPLOYMENT_RISK_TEXT[spectrum_position],
        'report':                       report,
    }
