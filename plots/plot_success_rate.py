# plot successplots as bar_plots, in sim and real
# in sim use distance to target < threshold as success across all tasks, in real as well
# 3 bars + 1 + 1? for real world makes 4 or 5 bars

# plot with Dino and diffferent Obs Encoder

import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Hook (Sim)', 'Push (Sim)', 'Push (Real)']
success_rates = [0.45, 0.98, 0.65]  # Dummy values

# Plotting
plt.style.use(['seaborn-v0_8-paper']) # Using a built-in style that looks academic
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

fig, ax = plt.subplots(figsize=(6, 4))
colors = ['#1f77b4', '#1f77b4', '#ff7f0e'] # Use same color for Sim, different for Real

bars = ax.bar(categories, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

# Add text labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{height*100:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Formatting
ax.set_ylabel('Success Rate')
ax.set_ylim(0, 1.1)
ax.set_title('Evaluation Success Rates across Domains', pad=15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent clipping
plt.tight_layout()

# Save plot
plt.savefig('success_rates_comparison.pdf', format='pdf')
print("Plot saved as success_rates_comparison.pdf")