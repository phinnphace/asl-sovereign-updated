# --- GEMMA 4 CONFUSION MATRIX: ROBOFLOW DATASET (row-normalized) ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

LABELS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
n = len(LABELS)

roboflow_matrix = np.array([
    [12,1,2,0,1,1,0,0,0,0,0,5,0,0,1,1,0,0,11,0,0,0,0,0,0],
    [6,5,0,0,7,1,1,0,0,0,0,8,0,0,0,0,0,0,0,1,0,0,7,0,0],
    [0,0,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [15,0,0,2,0,0,1,0,43,2,0,1,0,0,0,0,0,0,0,2,0,0,0,0,0],
    [0,3,0,0,7,1,0,0,0,0,0,9,11,0,1,0,0,1,0,2,0,2,30,0,0],
    [4,1,1,0,11,0,0,0,0,0,4,0,0,0,1,0,0,0,0,0,0,3,7,0,1],
    [0,0,0,18,0,0,0,0,14,5,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0],
    [2,17,1,2,16,1,0,1,3,0,0,6,0,0,0,1,0,0,0,0,0,7,5,0,0],
    [1,0,0,9,0,0,1,0,34,2,0,2,0,0,0,4,0,1,0,1,0,5,0,0,0],
    [3,4,3,7,9,2,1,0,1,2,0,4,0,0,0,1,0,1,0,0,1,2,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,40,0,0,0],
    [3,0,0,6,1,0,1,0,23,0,0,3,0,0,0,1,0,0,0,2,0,0,1,0,1],
    [11,2,1,0,2,1,1,0,0,0,0,8,0,0,1,0,0,2,9,1,0,1,2,0,0],
    [4,5,1,0,11,2,0,1,1,0,0,5,1,0,1,2,0,0,0,0,0,0,5,0,0],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,44,0,0,0,0,0,0,1,0,0,0],
    [1,0,0,28,0,1,2,1,5,5,1,18,0,0,0,2,0,0,0,1,0,1,2,0,0],
    [1,0,0,6,6,0,1,1,2,1,0,0,0,0,0,0,0,0,0,2,1,5,2,0,2],
    [5,1,0,0,7,1,1,0,1,0,0,5,3,0,0,0,0,1,0,0,0,10,6,0,3],
    [11,1,1,1,1,2,0,1,0,0,0,17,1,0,2,3,0,0,12,0,0,4,1,1,0],
    [0,0,0,17,0,0,2,0,11,1,0,3,0,0,0,0,0,0,0,3,0,0,0,0,0],
    [7,2,0,0,8,1,0,1,0,0,0,4,2,0,0,0,0,0,0,5,0,7,12,0,2],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0],
    [3,0,0,0,11,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,4,16,0,0],
    [0,0,2,5,0,0,1,0,25,6,0,0,0,0,0,3,0,4,0,1,0,1,0,0,0],
    [3,0,0,11,1,0,0,3,2,0,0,11,0,0,0,0,0,0,0,1,0,17,2,0,1],
])

# Row-normalize: each row sums to 1.0
row_sums = roboflow_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
norm_roboflow = roboflow_matrix / row_sums

overall_acc = roboflow_matrix.diagonal().sum() / roboflow_matrix.sum() * 100

fig = plt.figure(figsize=(16, 13))
gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.35)
ax_cm = fig.add_subplot(gs[0])

sns.heatmap(norm_roboflow, annot=False, cmap='YlOrRd',
            xticklabels=LABELS, yticklabels=LABELS,
            linewidths=0.3, linecolor='white', ax=ax_cm,
            cbar_kws={'shrink': 0.5, 'label': 'Proportion of true label'},
            vmin=0, vmax=1.0)

for i in range(n):
    ax_cm.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                    color='#185FA5', alpha=0.6, zorder=2))
    ax_cm.text(i+0.5, i+0.5, f'{norm_roboflow[i,i]:.2f}',
               ha='center', va='center', fontsize=7,
               color='white', fontweight='bold', zorder=3)

ax_cm.set_xlabel('Predicted letter', fontsize=11)
ax_cm.set_ylabel('Actual letter', fontsize=11)
ax_cm.set_title(
    f'Gemma 4 Confusion Matrix — Roboflow ASL (simple prompt, temp=0.1)\n'
    f'Overall accuracy: {overall_acc:.1f}% | 1,185 images | 25 letters | row-normalized',
    fontsize=12, pad=10)

ax_call = fig.add_subplot(gs[1])
ax_call.axis('off')
callout = (
    'WHAT WE SEE\n'
    '\u2022 C (1.00), O (0.96), V (1.00): near-perfect -- geometrically isolated letters survive\n'
    '\u2022 K: 0.00 -- 40/41 images predicted as V (2.3rd percentile landmark distance -- predicted failure)\n'
    '\u2022 D: 0.03 -- 43/66 predicted as I (52nd percentile -- NOT predicted by geometry)\n'
    '\u2022 F, G, M, N, Q, U, X, Y: at or near 0.00\n\n'
    'TWO FAILURE MECHANISMS (see landmark geometry section)\n'
    '\u2022 V-family collapses (K\u2192V, Y\u2192V, R\u2192V): predicted by landmark geometry -- geometrically similar\n'
    '\u2022 I/D-family collapses (D\u2192I, X\u2192I, T\u2192D): NOT predicted -- suggests signer-context training data\n\n'
    'PATTERN: Landmark-like (blocky off-diagonal clusters) -- '
    'full vision model producing hand-only skeleton model confusion structure'
)
ax_call.text(0.01, 0.95, callout, transform=ax_call.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#E6F1FB',
                       edgecolor='#185FA5', linewidth=1.2, alpha=0.9))
plt.tight_layout()
plt.show()
print(f'Roboflow overall: {overall_acc:.1f}%')


# --- GEMMA 4 CONFUSION MATRIX: ISL DATASET (row-normalized) ---

isl_matrix = np.array([
    [9,3,1,0,1,0,0,0,0,0,0,5,0,0,0,2,0,0,12,1,0,0,1,0,0],
    [3,5,1,0,9,0,2,0,0,0,0,8,1,0,0,0,0,0,0,0,0,0,7,0,0],
    [0,0,59,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [18,0,0,3,0,0,0,0,37,0,0,1,0,0,0,2,0,0,0,4,0,0,1,0,0],
    [0,4,0,0,6,0,0,0,0,0,0,7,12,0,1,0,0,0,0,0,0,2,34,0,0],
    [2,1,0,0,11,0,0,0,0,0,2,3,1,0,1,0,0,0,0,1,0,3,6,1,0],
    [0,0,0,19,0,0,2,0,12,4,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [4,8,0,2,15,2,0,0,0,2,0,7,2,0,0,0,0,0,0,0,0,6,14,1,1],
    [2,0,0,5,1,0,3,2,32,0,0,2,0,0,0,3,0,2,0,6,0,3,0,0,0],
    [1,7,1,5,6,4,0,1,0,3,1,6,0,1,0,0,0,0,1,0,0,2,5,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,0,0,0],
    [1,1,0,4,2,0,1,0,18,2,0,3,0,0,0,1,0,1,0,5,0,2,1,0,0],
    [11,4,3,0,2,0,1,0,0,0,0,6,0,0,0,0,0,0,8,3,0,1,1,0,1],
    [6,3,2,2,6,1,0,0,0,0,0,4,0,0,0,0,0,0,0,1,0,0,14,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,0,0,0,0,0,0,0,0,0,0],
    [2,1,0,23,1,3,5,1,9,5,0,7,0,0,0,1,0,0,0,3,0,7,0,0,0],
    [5,1,0,5,1,0,0,1,2,0,0,1,0,0,0,2,0,0,0,0,0,7,5,0,0],
    [5,1,0,0,9,0,2,0,2,0,0,4,3,0,0,0,0,0,0,1,0,5,4,0,4],
    [11,1,1,1,1,1,1,0,0,0,0,15,1,0,1,1,0,1,10,3,0,6,3,0,0],
    [0,0,0,15,0,0,1,0,10,1,0,4,1,0,0,2,0,0,0,3,0,0,0,0,0],
    [7,0,0,0,11,0,0,1,0,0,0,4,4,0,1,0,0,0,0,2,1,11,11,0,2],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0],
    [3,1,0,0,13,0,0,0,0,0,0,0,0,0,1,0,0,2,0,0,0,1,12,0,0],
    [0,0,3,2,0,0,0,0,29,8,0,1,0,0,0,2,0,2,0,1,0,0,0,0,0],
    [0,1,0,4,2,1,0,1,2,1,0,11,3,0,0,3,0,2,0,5,0,13,2,1,0],
])

# Row-normalize: each row sums to 1.0
row_sums = isl_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
norm_isl = isl_matrix / row_sums

isl_acc = isl_matrix.diagonal().sum() / isl_matrix.sum() * 100

fig = plt.figure(figsize=(16, 13))
gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.35)
ax_cm = fig.add_subplot(gs[0])

sns.heatmap(norm_isl, annot=False, cmap='YlOrRd',
            xticklabels=LABELS, yticklabels=LABELS,
            linewidths=0.3, linecolor='white', ax=ax_cm,
            cbar_kws={'shrink': 0.5, 'label': 'Proportion of true label'},
            vmin=0, vmax=1.0)

for i in range(n):
    ax_cm.add_patch(plt.Rectangle((i, i), 1, 1, fill=True,
                    color='#185FA5', alpha=0.6, zorder=2))
    ax_cm.text(i+0.5, i+0.5, f'{norm_isl[i,i]:.2f}',
               ha='center', va='center', fontsize=7,
               color='white', fontweight='bold', zorder=3)

ax_cm.set_xlabel('Predicted letter', fontsize=11)
ax_cm.set_ylabel('Actual letter', fontsize=11)
ax_cm.set_title(
    f'Gemma 4 Confusion Matrix — ISL Dataset (simple prompt, temp=0.1)\n'
    f'Overall accuracy: {isl_acc:.1f}% | 1,185 images | 25 letters | row-normalized',
    fontsize=12, pad=10)

ax_call = fig.add_subplot(gs[1])
ax_call.axis('off')
callout = (
    'WHAT WE SEE\n'
    '\u2022 C (0.98), O (1.00), V (1.00): same letters survive as Roboflow -- convergence confirmed\n'
    '\u2022 K: 0.00 -- all 41 images predicted as V. Complete collapse. Zero variance.\n'
    '\u2022 F, H, M, N, Q, R, X, Y: 0.00 -- 9 letters with complete failure\n'
    '\u2022 Zero refusals -- model sees ISL hands clearly and is confidently wrong\n\n'
    'WHAT CHANGED vs ROBOFLOW\n'
    '\u2022 Roboflow: 12 refusals (model uncertain on some images)\n'
    '\u2022 ISL: 0 refusals -- confident misidentification is categorically more dangerous than refusal\n'
    '\u2022 ISL hand orientations outside model training distribution -- '
    'pattern-matches to nearest ASL analog\n\n'
    'PATTERN: Landmark-like (blocky clusters) -- '
    'cross-linguistic transfer fails; model calibrated to ASL-adjacent training data specifically'
)
ax_call.text(0.01, 0.95, callout, transform=ax_call.transAxes,
             fontsize=9, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FAECE7',
                       edgecolor='#D85A30', linewidth=1.2, alpha=0.9))
plt.tight_layout()
plt.show()
print(f'ISL overall: {isl_acc:.1f}%')
print(f'Convergence: Roboflow {overall_acc:.1f}% vs ISL {isl_acc:.1f}% -- ~1% gap across different sign languages')
print(f'K: 0.00 on both datasets. 40/41 Roboflow and 41/41 ISL predicted as V.')
