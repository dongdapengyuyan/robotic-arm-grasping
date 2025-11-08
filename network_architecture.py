import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_network_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.axis('off')

    # Color scheme
    color_input = '#E8F4F8'
    color_frozen = '#FFE6E6'
    color_trainable = '#E6F3E6'
    color_output = '#FFF4E6'

    # Arrow style
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#333333')

    y_pos = 27

    # INPUT
    ax.text(5, y_pos, 'INPUT: RGB Image (Batch, 3, 224, 224)',
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color_input, edgecolor='black', lw=2))

    # Arrow
    y_pos -= 1
    ax.annotate('', xy=(5, y_pos - 0.3), xytext=(5, y_pos + 0.3), arrowprops=arrow_props)

    # ResNet-34 Encoder Box
    y_pos -= 1
    box_height = 11
    encoder_box = FancyBboxPatch((0.5, y_pos - box_height), 9, box_height,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor=color_frozen, lw=2.5)
    ax.add_patch(encoder_box)

    # Encoder Title
    ax.text(5, y_pos - 0.5, 'ResNet-34 Encoder (Frozen)',
            ha='center', va='center', fontsize=12, weight='bold')

    # Encoder layers
    layer_y = y_pos - 1.5
    layers = [
        'Conv1: 7×7 conv, 64 filters, stride 2',
        'BatchNorm + ReLU + MaxPool',
        '',
        'Layer1: [BasicBlock × 3]',
        '  64 filters, residual connections',
        '',
        'Layer2: [BasicBlock × 4]',
        '  128 filters, residual connections',
        '',
        'Layer3: [BasicBlock × 6]',
        '  256 filters, residual connections',
        '',
        'Layer4: [BasicBlock × 3]',
        '  512 filters, residual connections',
        '',
        'AdaptiveAvgPool2d: Output (1, 1)',
        '',
        'Feature Vector: (Batch, 512)',
        'Frozen Parameters: ~11.2M'
    ]

    for layer in layers:
        if layer:
            weight = 'bold' if 'Layer' in layer or 'Conv1' in layer or 'Feature' in layer or 'Frozen' in layer else 'normal'
            fontsize = 9.5 if weight == 'bold' else 9
            ax.text(5, layer_y, layer, ha='center', va='center',
                    fontsize=fontsize, weight=weight, family='monospace')
        layer_y -= 0.5

    # Arrow
    y_pos = y_pos - box_height - 0.5
    ax.annotate('', xy=(5, y_pos - 0.3), xytext=(5, y_pos + 0.3), arrowprops=arrow_props)

    # Regression Head Box
    y_pos -= 1
    box_height = 6.5
    head_box = FancyBboxPatch((0.5, y_pos - box_height), 9, box_height,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color_trainable, lw=2.5)
    ax.add_patch(head_box)

    # Head Title
    ax.text(5, y_pos - 0.5, 'Regression Head (Trainable)',
            ha='center', va='center', fontsize=12, weight='bold')

    # Head layers
    layer_y = y_pos - 1.3
    head_layers = [
        'FC1: Linear(512 → 256)',
        '  BatchNorm1d(256) + ReLU',
        '  Dropout(p=0.5)',
        '',
        'FC2: Linear(256 → 128)',
        '  BatchNorm1d(128) + ReLU',
        '  Dropout(p=0.5)',
        '',
        'FC3: Linear(128 → 3)',
        '  Output: [x, y, angle]',
        '',
        'Trainable Parameters: ~0.2M'
    ]

    for layer in head_layers:
        if layer:
            weight = 'bold' if 'FC' in layer or 'Output' in layer or 'Trainable' in layer else 'normal'
            fontsize = 9.5 if weight == 'bold' else 9
            ax.text(5, layer_y, layer, ha='center', va='center',
                    fontsize=fontsize, weight=weight, family='monospace')
        layer_y -= 0.45

    # Arrow
    y_pos = y_pos - box_height - 0.5
    ax.annotate('', xy=(5, y_pos - 0.3), xytext=(5, y_pos + 0.3), arrowprops=arrow_props)

    # OUTPUT
    y_pos -= 1
    ax.text(5, y_pos, 'OUTPUT: (Batch, 3)',
            ha='center', va='center', fontsize=11, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color_output, edgecolor='black', lw=2))

    y_pos -= 0.7
    ax.text(5, y_pos, 'x ∈ [0, 1],  y ∈ [0, 1],  angle ∈ [-1, 1]',
            ha='center', va='center', fontsize=10, style='italic', family='monospace')

    plt.tight_layout()
    plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Network architecture diagram saved as 'network_architecture.png'")
    plt.show()


if __name__ == '__main__':
    create_network_architecture()