import json
import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science', 'no-latex'])

def plot_loss_professional(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    train_loss = np.array(data.get('train_losses', []))
    val_loss = np.array(data.get('val_losses', []))

    steps_train = np.arange(len(train_loss))
    steps_val = np.linspace(0, len(train_loss) - 1, len(val_loss))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)     # 8, 5 for one plot over page (4, 3 for two three side by side?)

    # 1. Training Loss (Raw + Smoothed)
    ax.plot(steps_train, train_loss, color='#377eb8', alpha=0.2, linewidth=0.5, label='_nolegend_')
    
    # Exponential Moving Average smoothing (W&B Style)
    def ema_smooth(scalars, weight=0.99): 
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    ax.plot(steps_train, ema_smooth(train_loss), color='#377eb8', linewidth=1.5, label='Train (EMA)')

    # 2. Validation Loss
    ax.plot(steps_val, val_loss, color='#e27577', linewidth=1.5, marker='o', 
            markersize=4, markeredgewidth=0.5, label='Validation')

    # 3. Log Scale - This is the magic fix for "squashed" plots
    ax.set_yscale('log')
    
    # Formatting
    ax.set_xlabel('training steps', fontsize=12)
    ax.set_ylabel('loss', fontsize=12)
    #ax.set_title('Convergence Profile: Table Transformer', fontsize=14, pad=15)
    
    ax.grid(True, which="both", ls="-", alpha=0.2) # Show minor gridlines for log scale
    ax.legend(frameon=True, loc='upper right')
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('improved_loss_log.png')
    plt.show()

plot_loss_professional('plot_data/table_transformer_rgb.json')