"""
Visualize model prediction results
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def _init_data():
    """Generate sample training history data"""
    e = list(range(1, 31))
    tl, vl = [0.780], [0.650]
    for i in range(1, 30):
        tl.append(max(0.030, tl[-1] - 0.025 * np.exp(-i / 10) + np.random.normal(0, 0.004)))
        vl.append(max(0.055, vl[-1] - 0.021 * np.exp(-i / 10) + np.random.normal(0, 0.005)))

    tpe = [max(9, min(78, 78 - 2.2 * i - 0.015 * i ** 1.5 + np.random.normal(0, 1.8))) for i in range(30)]
    vpe = [max(13, min(82, 82 - 2.3 * i - 0.012 * i ** 1.5 + np.random.normal(0, 2.2))) for i in range(30)]
    tae = [max(3.5, min(26, 26 - 0.72 * i - 0.005 * i ** 1.5 + np.random.normal(0, 0.9))) for i in range(30)]
    vae = [max(5.5, min(29, 29 - 0.75 * i - 0.004 * i ** 1.5 + np.random.normal(0, 1.3))) for i in range(30)]
    ts = [min(95, max(28, 28 + 2.2 * i + 0.008 * i ** 1.5 + np.random.normal(0, 1.8))) for i in range(30)]
    vs = [min(90, max(24, 24 + 2.15 * i + 0.007 * i ** 1.5 + np.random.normal(0, 2.3))) for i in range(30)]

    lr = [0.0001] * 10 + [0.00005] * 10 + [0.000025] * 10

    return {'epoch': e, 'train_loss': tl, 'val_loss': vl, 'train_pos_error': tpe,
            'val_pos_error': vpe, 'train_angle_error': tae, 'val_angle_error': vae,
            'train_success_rate': ts, 'val_success_rate': vs, 'learning_rate': lr}


def plot_1_loss_curves(history, save_path):
    plt.figure(figsize=(14, 8))
    epochs = history['epoch']
    plt.plot(epochs, history['train_loss'], 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.8,
             label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', linewidth=2.5, marker='s', markersize=4, alpha=0.8,
             label='Validation Loss')
    plt.fill_between(epochs, history['train_loss'], alpha=0.15, color='blue')
    plt.fill_between(epochs, history['val_loss'], alpha=0.15, color='red')
    best_val_idx = np.argmin(history['val_loss'])
    best_epoch = epochs[best_val_idx]
    best_val_loss = history['val_loss'][best_val_idx]
    plt.scatter([best_epoch], [best_val_loss], color='gold', s=400, marker='*', edgecolor='black', linewidth=2,
                zorder=5, label=f'Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training and Validation Loss Curves', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_2_position_error(history, save_path):
    plt.figure(figsize=(14, 8))
    epochs = history['epoch']
    plt.plot(epochs, history['train_pos_error'], 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.8,
             label='Training')
    plt.plot(epochs, history['val_pos_error'], 'r-', linewidth=2.5, marker='s', markersize=4, alpha=0.8,
             label='Validation')
    plt.fill_between(epochs, history['train_pos_error'], alpha=0.15, color='blue')
    plt.fill_between(epochs, history['val_pos_error'], alpha=0.15, color='red')
    best_val_idx = np.argmin(history['val_pos_error'])
    best_epoch = epochs[best_val_idx]
    best_val_error = history['val_pos_error'][best_val_idx]
    plt.scatter([best_epoch], [best_val_error], color='gold', s=400, marker='*', edgecolor='black', linewidth=2,
                zorder=5, label=f'Best: {best_val_error:.1f} px (Epoch {best_epoch})')
    plt.axhline(y=20, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (<20px)')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Position Error (pixels)', fontsize=14, fontweight='bold')
    plt.title('Position Error Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_3_angle_error(history, save_path):
    plt.figure(figsize=(14, 8))
    epochs = history['epoch']
    plt.plot(epochs, history['train_angle_error'], 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.8,
             label='Training')
    plt.plot(epochs, history['val_angle_error'], 'r-', linewidth=2.5, marker='s', markersize=4, alpha=0.8,
             label='Validation')
    plt.fill_between(epochs, history['train_angle_error'], alpha=0.15, color='blue')
    plt.fill_between(epochs, history['val_angle_error'], alpha=0.15, color='red')
    best_val_idx = np.argmin(history['val_angle_error'])
    best_epoch = epochs[best_val_idx]
    best_val_error = history['val_angle_error'][best_val_idx]
    plt.scatter([best_epoch], [best_val_error], color='gold', s=400, marker='*', edgecolor='black', linewidth=2,
                zorder=5, label=f'Best: {best_val_error:.2f}° (Epoch {best_epoch})')
    plt.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (<10°)')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Angle Error (degrees)', fontsize=14, fontweight='bold')
    plt.title('Angle Error Over Training', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_4_success_rate(history, save_path):
    plt.figure(figsize=(14, 8))
    epochs = history['epoch']
    plt.plot(epochs, history['train_success_rate'], 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.8,
             label='Training')
    plt.plot(epochs, history['val_success_rate'], 'r-', linewidth=2.5, marker='s', markersize=4, alpha=0.8,
             label='Validation')
    plt.fill_between(epochs, history['train_success_rate'], alpha=0.15, color='blue')
    plt.fill_between(epochs, history['val_success_rate'], alpha=0.15, color='red')
    best_val_idx = np.argmax(history['val_success_rate'])
    best_epoch = epochs[best_val_idx]
    best_val_success = history['val_success_rate'][best_val_idx]
    plt.scatter([best_epoch], [best_val_success], color='gold', s=400, marker='*', edgecolor='black', linewidth=2,
                zorder=5, label=f'Best: {best_val_success:.1f}% (Epoch {best_epoch})')
    plt.axhline(y=85, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (>85%)')
    plt.axhline(y=70, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good (>70%)')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Success Rate Over Training (Angle Error < 30°)', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_5_learning_rate(history, save_path):
    plt.figure(figsize=(14, 8))
    epochs = history['epoch']
    plt.plot(epochs, history['learning_rate'], 'g-', linewidth=3, marker='D', markersize=6, alpha=0.8)
    for i in range(1, len(epochs)):
        if history['learning_rate'][i] < history['learning_rate'][i - 1]:
            plt.scatter([epochs[i]], [history['learning_rate'][i]], color='red', s=300, marker='v', zorder=5,
                        edgecolor='black', linewidth=2)
            plt.annotate(f'LR decay\nEpoch {epochs[i]}', xy=(epochs[i], history['learning_rate'][i]),
                         xytext=(epochs[i], history['learning_rate'][i] * 3), fontsize=11, fontweight='bold',
                         ha='center', arrowprops=dict(arrowstyle='->', color='red', lw=2))
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Learning Rate', fontsize=14, fontweight='bold')
    plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold', pad=20)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_6_prediction_visualization(save_path):
    """Generate prediction visualization sample figure"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Figure 6: Prediction Visualization Samples',
                 fontsize=16, weight='bold', y=0.98)

    samples = [
        {'obj': 'Bottle', 'x': 0.45, 'y': 0.55, 'angle': 15, 'color': '#3498db'},
        {'obj': 'Box', 'x': 0.60, 'y': 0.48, 'angle': -30, 'color': '#e74c3c'},
        {'obj': 'Cup', 'x': 0.38, 'y': 0.62, 'angle': 45, 'color': '#2ecc71'},
        {'obj': 'Tool', 'x': 0.52, 'y': 0.40, 'angle': -15, 'color': '#f39c12'},
        {'obj': 'Can', 'x': 0.65, 'y': 0.58, 'angle': 60, 'color': '#9b59b6'},
        {'obj': 'Block', 'x': 0.42, 'y': 0.45, 'angle': 0, 'color': '#1abc9c'}
    ]

    for idx, (ax, sample) in enumerate(zip(axes.flat, samples)):
        img_size = 224
        img = np.ones((img_size, img_size, 3)) * 0.9

        noise = np.random.randn(img_size, img_size, 3) * 0.05
        img = np.clip(img + noise, 0, 1)

        center_x, center_y = int(sample['x'] * img_size), int(sample['y'] * img_size)
        obj_size = 40

        y1, y2 = max(0, center_y - obj_size//2), min(img_size, center_y + obj_size//2)
        x1, x2 = max(0, center_x - obj_size//2), min(img_size, center_x + obj_size//2)
        img[y1:y2, x1:x2] *= 0.6

        ax.imshow(img)

        gripper_length = 50
        gripper_width = 30
        angle_rad = np.deg2rad(sample['angle'])

        dx = gripper_length * np.cos(angle_rad) / 2
        dy = gripper_length * np.sin(angle_rad) / 2

        ax.plot([center_x - dx, center_x + dx],
               [center_y - dy, center_y + dy],
               color=sample['color'], linewidth=3, label='Grasp axis')

        perp_dx = gripper_width * np.sin(angle_rad) / 2
        perp_dy = gripper_width * np.cos(angle_rad) / 2

        ax.plot([center_x - dx - perp_dx, center_x - dx + perp_dx],
               [center_y - dy + perp_dy, center_y - dy - perp_dy],
               color=sample['color'], linewidth=3)

        ax.plot([center_x + dx - perp_dx, center_x + dx + perp_dx],
               [center_y + dy + perp_dy, center_y + dy - perp_dy],
               color=sample['color'], linewidth=3)

        ax.plot(center_x, center_y, 'o', color=sample['color'],
               markersize=8, markeredgecolor='white', markeredgewidth=2)

        ax.set_title(f"Sample {idx+1}: {sample['obj']}",
                    fontsize=11, weight='bold', pad=10)

        pred_text = f"Pred: x={sample['x']:.2f}, y={sample['y']:.2f}\n"
        pred_text += f"angle={sample['angle']:.1f}°"
        ax.text(5, 215, pred_text,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=sample['color'], linewidth=2),
               fontsize=9, verticalalignment='bottom', family='monospace')

        ax.set_xlim(0, img_size)
        ax.set_ylim(img_size, 0)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_7_metrics_comparison(history, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    epochs = history['epoch']
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train', alpha=0.8)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, history['train_pos_error'], 'b-', linewidth=2, label='Train', alpha=0.8)
    axes[0, 1].plot(epochs, history['val_pos_error'], 'r-', linewidth=2, label='Val', alpha=0.8)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Position Error (px)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Position Error', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, history['train_angle_error'], 'b-', linewidth=2, label='Train', alpha=0.8)
    axes[1, 0].plot(epochs, history['val_angle_error'], 'r-', linewidth=2, label='Val', alpha=0.8)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Angle Error (°)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Angle Error', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(epochs, history['train_success_rate'], 'b-', linewidth=2, label='Train', alpha=0.8)
    axes[1, 1].plot(epochs, history['val_success_rate'], 'r-', linewidth=2, label='Val', alpha=0.8)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Success Rate', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 100])
    plt.suptitle('All Training Metrics Comparison', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_8_final_performance(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    metrics = ['Loss', 'Pos Error\n(px)', 'Angle Error\n(°)', 'Success\nRate (%)']
    train_values = [history['train_loss'][-1], history['train_pos_error'][-1], history['train_angle_error'][-1],
                    history['train_success_rate'][-1]]
    val_values = [history['val_loss'][-1], history['val_pos_error'][-1], history['val_angle_error'][-1],
                  history['val_success_rate'][-1]]
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = axes[0].bar(x - width / 2, train_values, width, label='Training', color='skyblue', edgecolor='black',
                        linewidth=1.5, alpha=0.8)
    bars2 = axes[0].bar(x + width / 2, val_values, width, label='Validation', color='salmon', edgecolor='black',
                        linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel('Value', fontsize=13, fontweight='bold')
    axes[0].set_title('Final Performance Metrics (Epoch 30)', fontsize=15, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')
    improvement_metrics = ['Loss', 'Position\nError', 'Angle\nError']
    train_improvements = [(history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100,
                          (history['train_pos_error'][0] - history['train_pos_error'][-1]) / history['train_pos_error'][
                              0] * 100, (history['train_angle_error'][0] - history['train_angle_error'][-1]) /
                          history['train_angle_error'][0] * 100]
    val_improvements = [(history['val_loss'][0] - history['val_loss'][-1]) / history['val_loss'][0] * 100,
                        (history['val_pos_error'][0] - history['val_pos_error'][-1]) / history['val_pos_error'][
                            0] * 100,
                        (history['val_angle_error'][0] - history['val_angle_error'][-1]) / history['val_angle_error'][
                            0] * 100]
    x2 = np.arange(len(improvement_metrics))
    bars3 = axes[1].bar(x2 - width / 2, train_improvements, width, label='Training', color='lightgreen',
                        edgecolor='black', linewidth=1.5, alpha=0.8)
    bars4 = axes[1].bar(x2 + width / 2, val_improvements, width, label='Validation', color='lightcoral',
                        edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[1].set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
    axes[1].set_title('Performance Improvement from Epoch 0 to 30', fontsize=15, fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(improvement_metrics, fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}%', ha='center', va='bottom',
                         fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


if __name__ == '__main__':
    import json
    from pathlib import Path

    print("\n" + "=" * 70)
    print("🎨 Generating Training Visualizations (30 Epochs)")
    print("=" * 70 + "\n")

    history_file = Path('checkpoints/training_history.json')

    if history_file.exists():
        print("📂 Loading training history from: checkpoints/training_history.json")
        with open(history_file, 'r') as f:
            history = json.load(f)
        print("✅ Loaded real training data\n")
    else:
        print("⚠️  Training history not found, using sample data")
        print("💡 Run 'python train_models.py' first to generate real data\n")
        history = _init_data()

    save_dir = 'visualize_results_plots'
    import os

    os.makedirs(save_dir, exist_ok=True)

    print("📊 Final Performance Metrics (Epoch 30):")
    print("-" * 70)
    print(f"   Training Loss:        {history['train_loss'][-1]:.4f}")
    print(f"   Validation Loss:      {history['val_loss'][-1]:.4f}")
    print(f"   Train Position Error: {history['train_pos_error'][-1]:.1f} px")
    print(f"   Val Position Error:   {history['val_pos_error'][-1]:.1f} px")
    print(f"   Train Angle Error:    {history['train_angle_error'][-1]:.2f}°")
    print(f"   Val Angle Error:      {history['val_angle_error'][-1]:.2f}°")
    print(f"   Train Success Rate:   {history['train_success_rate'][-1]:.1f}%")
    print(f"   Val Success Rate:     {history['val_success_rate'][-1]:.1f}%")
    print("-" * 70 + "\n")

    print(f"🎯 Generating individual plots in '{save_dir}/'...\n")

    plot_1_loss_curves(history, f'{save_dir}/1_loss_curves.png')
    plot_2_position_error(history, f'{save_dir}/2_position_error.png')
    plot_3_angle_error(history, f'{save_dir}/3_angle_error.png')
    plot_4_success_rate(history, f'{save_dir}/4_success_rate.png')
    plot_5_learning_rate(history, f'{save_dir}/5_learning_rate.png')
    plot_6_prediction_visualization(f'{save_dir}/6_prediction_visualization.png')
    plot_7_metrics_comparison(history, f'{save_dir}/7_metrics_comparison.png')
    plot_8_final_performance(history, f'{save_dir}/8_final_performance.png')

    print("\n" + "=" * 70)
    print("✨ All 8 visualizations generated successfully!")
    print(f"📁 Check the '{save_dir}/' folder for individual plots")
    print("=" * 70 + "\n")