import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # Import for formatting

# Ensure scienceplots is installed: pip install scienceplots
try:
    import scienceplots
    plt.style.use(['science', 'ieee'])
except ImportError:
    plt.style.use('seaborn-v0_8-paper') 

def scheduler(t, eta_max, T_warmup, T_total):
    if t < T_warmup:
        return eta_max * (t / T_warmup)
    else:
        return 0.5 * eta_max * (1 + np.cos(((t - T_warmup) / (T_total - T_warmup)) * np.pi))

# Configuration
eta_max = 0.0001
T_warmup = 1000
T_total = 50_000

# Generate data
t_values = np.linspace(0, T_total, 1000)
eta_values = [scheduler(t, eta_max, T_warmup, T_total) for t in t_values]

# Plotting
fig, ax = plt.subplots(figsize=(5, 3)) # Using fig, ax for cleaner formatting
ax.plot(t_values, eta_values, label=r'$\eta_t$ Schedule', color='tab:blue')

# --- Scientific Notation Logic ---
# This forces the y-axis to use scientific notation for small numbers
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1)) # Trigger scientific notation outside [0.1, 10]
ax.yaxis.set_major_formatter(formatter)

# Annotations
ax.axvline(x=T_warmup, color='gray', linestyle='--', alpha=0.5)
# Moving the text slightly higher/further so it doesn't crowd the line
ax.text(T_warmup + 1000, eta_max * 0.95, 'Warmup End', fontsize=8, verticalalignment='top')

ax.set_xlabel('training step ($t$)')
ax.set_ylabel('learning rate ($\eta_t$)')
#ax.set_title(r'Learning Rate Scheduler: Linear Warmup \& Cosine Decay')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig('lr_scheduler.pdf')
plt.show()