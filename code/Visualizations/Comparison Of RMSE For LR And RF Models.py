import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Environmental Model', 'Socioeconomic model', 'Combined Model']
lr_values = [[2.1, 0.85], [4.22, 0.41], [1.94, 0.86]]
rf_values = [[1.59, 0.901], [2.99, 0.7], [1.41, 0.927]]

# Separate data for RMSE and R2
rmse_lr = [row[0] for row in lr_values]
rmse_rf = [row[0] for row in rf_values]

r2_lr = [row[1] for row in lr_values]
r2_rf = [row[1] for row in rf_values]

# Bar positions
bar_width = 0.35
x = np.arange(len(models))

# Plot RMSE
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.bar(x - bar_width/2, rmse_lr, bar_width, label='LR RMSE', color='blue')
ax1.bar(x + bar_width/2, rmse_rf, bar_width, label='RF RMSE', color='orange')

ax1.set_xlabel('Models', fontsize=16)
ax1.set_ylabel('RMSE Values', fontsize=16)
ax1.set_title('Comparison of RMSE for LR and RF Models', fontsize=18)
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=14)
ax1.legend(fontsize=14)
plt.tight_layout()
plt.show()

# Plot R2
fig, ax2 = plt.subplots(figsize=(10, 6))
ax2.bar(x - bar_width/2, r2_lr, bar_width, label='LR R2', color='blue')
ax2.bar(x + bar_width/2, r2_rf, bar_width, label='RF R2', color='orange')

ax2.set_xlabel('Models', fontsize=16)
ax2.set_ylabel('R2 Values', fontsize=16)
ax2.set_title('Comparison of R2 for LR and RF Models', fontsize=18)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=14)
ax2.legend(fontsize=14)
plt.tight_layout()
plt.show()
