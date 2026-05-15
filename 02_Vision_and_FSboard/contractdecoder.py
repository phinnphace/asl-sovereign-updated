{
    'spectrum_position': 'landmark-only' | 'landmark+context' | 'multimodal' | 'hybrid',
    'spectrum_score': float,          # 0.0 = landmark-only, 1.0 = hybrid
    'firing_pairs': [...],            # which decoder ring pairs detected
    'families': {...},                # V-family, I-family, M/N/T etc and their strength
    'predicted_additional_failures':  # pairs not yet confirmed but predicted
    'failure_mode': 'silent' | 'refusal' | 'mixed',
    'training_data_implication': str, # plain language
    'deployment_risk': str,           # plain language
    'report': str                     # full printable report
}