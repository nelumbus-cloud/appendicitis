import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import auc
from matplotlib.ticker import MaxNLocator

# Create directory if it doesn't exist
dir_name = "tabular_only"

os.makedirs(dir_name, exist_ok=True)

# Load the JSON data
with open("tabular_only_results.json", "r") as f:
    results = json.load(f)

# Extract hyperparameters for reference
hyperparams = results["hyperparameters"]
num_folds = len(results["folds"])
num_epochs = hyperparams["num_epochs"]

# 1. Plot ROC curves for all folds
plt.figure(figsize=(10, 8))
mean_tpr = np.zeros(100)
mean_fpr = np.linspace(0, 1, 100)
all_aurocs = []

for fold in results["folds"]:
    fold_num = fold["fold"]
    roc_data = fold["roc_data"]
    fpr = np.array(roc_data["fpr"])
    tpr = np.array(roc_data["tpr"])
    auroc = roc_data["auroc"]
    all_aurocs.append(auroc)
    
    # Interpolate TPR at standard FPR points for averaging
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    mean_tpr += interp_tpr
    
    plt.plot(fpr, tpr, alpha=0.3, label=f'Fold {fold_num+1} (AUC = {auroc:.3f})')

# Plot mean ROC curve
mean_tpr /= num_folds
mean_auroc = np.mean(all_aurocs)
std_auroc = np.std(all_aurocs)
print("Mean auroc: ", mean_auroc)
print("Std error: ", std_auroc/np.sqrt(10))

plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auroc:.2f}', lw=2)

# Plot chance line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Across All Folds (Tabular Only)')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig(f'{dir_name}/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Plot training and validation metrics over epochs (averaged across folds)
metrics = ['loss', 'acc', 'auroc']
titles = ['Loss', 'Accuracy', 'AUROC']
colors = ['blue', 'green', 'purple']

for i, metric in enumerate(metrics):
    plt.figure(figsize=(10, 6))
    
    # Arrays to store metrics for each fold and epoch
    train_values = np.zeros((num_folds, num_epochs))
    val_values = np.zeros((num_folds, num_epochs))
    
    for fold_idx, fold in enumerate(results["folds"]):
        epochs_data = fold["epochs"]
        
        for epoch_idx, epoch_data in enumerate(epochs_data):
            train_values[fold_idx, epoch_idx] = epoch_data[f"train_{metric}"]
            val_values[fold_idx, epoch_idx] = epoch_data[f"val_{metric}"]
    
    # Calculate mean and std across folds
    train_mean = np.mean(train_values, axis=0)
    train_std = np.std(train_values, axis=0)
    val_mean = np.mean(val_values, axis=0)
    val_std = np.std(val_values, axis=0)
    
    epochs = range(1, num_epochs + 1)
    
    plt.plot(epochs, train_mean, color=colors[i], label=f'Training {titles[i]}')
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color=colors[i], alpha=0.2)
    
    plt.plot(epochs, val_mean, color=colors[i], linestyle='--', label=f'Validation {titles[i]}')
    plt.fill_between(epochs, val_mean - val_std, val_mean + val_std, color=colors[i], alpha=0.1)
    
    plt.xlabel('Epoch')
    plt.ylabel(titles[i])
    plt.title(f'Training and Validation {titles[i]} Over Epochs')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'{dir_name}/{metric}_over_epochs.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Plot best AUROC by fold (bar chart)
plt.figure(figsize=(12, 6))
fold_nums = [fold["fold"] for fold in results["folds"]]
best_aurocs = [fold["best_auroc"] for fold in results["folds"]]

sns.barplot(x=fold_nums, y=best_aurocs)
plt.axhline(y=np.mean(best_aurocs), color='r', linestyle='--', label=f'Mean: {np.mean(best_aurocs):.3f}')
plt.xlabel('Fold')
plt.ylabel('Best AUROC')
plt.title('Best AUROC Score by Fold')
plt.ylim([min(best_aurocs) - 0.05, 1.0])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.savefig(f'{dir_name}/best_auroc_by_fold.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Plot learning curves (final epoch metrics)
plt.figure(figsize=(14, 7))

# Extract final epoch metrics for each fold
final_metrics = []
for fold in results["folds"]:
    final_epoch = fold["epochs"][-1]
    final_metrics.append({
        "fold": fold["fold"],
        "train_loss": final_epoch["train_loss"],
        "val_loss": final_epoch["val_loss"],
        "train_acc": final_epoch["train_acc"],
        "val_acc": final_epoch["val_acc"],
        "train_auroc": final_epoch["train_auroc"],
        "val_auroc": final_epoch["val_auroc"]
    })

# Create subplots for each metric
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = [("loss", "Loss"), ("acc", "Accuracy"), ("auroc", "AUROC")]

for i, (metric, title) in enumerate(metrics):
    ax = axes[i]
    train_values = [m[f"train_{metric}"] for m in final_metrics]
    val_values = [m[f"val_{metric}"] for m in final_metrics]
    
    x = np.arange(len(fold_nums))
    width = 0.35
    
    ax.bar(x - width/2, train_values, width, label=f'Training {title}')
    ax.bar(x + width/2, val_values, width, label=f'Validation {title}')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel(title)
    ax.set_title(f'Final Epoch {title} by Fold')
    ax.set_xticks(x)
    ax.set_xticklabels(fold_nums)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{dir_name}/final_metrics_by_fold.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Training progression heatmap
plt.figure(figsize=(12, 8))
auroc_progress = np.zeros((num_folds, num_epochs))

for fold_idx, fold in enumerate(results["folds"]):
    for epoch_idx, epoch_data in enumerate(fold["epochs"]):
        auroc_progress[fold_idx, epoch_idx] = epoch_data["val_auroc"]

sns.heatmap(auroc_progress, cmap="viridis", 
           xticklabels=10 if num_epochs > 30 else 5, 
           yticklabels=fold_nums,
           vmin=0.5, vmax=1.0)
plt.xlabel('Epoch')
plt.ylabel('Fold')
plt.title('Validation AUROC Progression Across Folds and Epochs')
plt.tight_layout()
plt.savefig(f'{dir_name}/auroc_progression_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Summary statistics table as a figure
plt.figure(figsize=(10, 6))
plt.axis('off')

# Calculate summary statistics
mean_best_auroc = np.mean(best_aurocs)
std_best_auroc = np.std(best_aurocs)
min_best_auroc = np.min(best_aurocs)
max_best_auroc = np.max(best_aurocs)

# Create a table with summary statistics and hyperparameters
table_data = [
    ["Metric", "Value"],
    ["Mean Best AUROC", f"{mean_best_auroc:.4f}"],
    ["Std Best AUROC", f"{std_best_auroc:.4f}"],
    ["Min Best AUROC", f"{min_best_auroc:.4f}"],
    ["Max Best AUROC", f"{max_best_auroc:.4f}"],
    ["Number of Folds", f"{num_folds}"],
    ["Number of Epochs", f"{num_epochs}"],
    ["Learning Rate", f"{hyperparams['learning_rate']}"],
    ["Weight Decay", f"{hyperparams['weight_decay']}"],
    ["Batch Size", f"{hyperparams['batch_size']}"],
]

plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.5])
plt.title('Model Performance Summary and Hyperparameters', pad=20)
plt.tight_layout()
plt.savefig(f'{dir_name}/summary_stats.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Boxplot of performance metrics across folds
plt.figure(figsize=(10, 6))

# Extract final metrics for each fold
final_train_auroc = [fold["epochs"][-1]["train_auroc"] for fold in results["folds"]]
final_val_auroc = [fold["epochs"][-1]["val_auroc"] for fold in results["folds"]]
best_val_auroc = [fold["best_auroc"] for fold in results["folds"]]

data = [final_train_auroc, final_val_auroc, best_val_auroc]
labels = ['Final Train AUROC', 'Final Val AUROC', 'Best Val AUROC']

plt.boxplot(data, labels=labels)
plt.ylabel('AUROC')
plt.title('Distribution of AUROC Metrics Across Folds')
plt.grid(axis='y', alpha=0.3)
plt.savefig(f'{dir_name}/auroc_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations have been created and saved to the '{dir_name}' directory.")
