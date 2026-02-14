"""
Plot training curves for LUNA16 two-stage pipeline.
Reads CSV logs from practice4/weights/ and saves plots to practice4/images/.

Usage:
    python practice4/plot_curves.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

WEIGHTS_DIR = "practice4/weights"
SAVE_DIR = "practice4/images"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# ============================================================
#  Stage 1: Segmentation Curves
# ============================================================

def plot_stage1():
    # --- Loss ---
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    for i in range(5):
        log = pd.read_csv(f"{WEIGHTS_DIR}/stage1/fold{i+1}/training_log.csv")
        label = f"Fold {i+1}"
        c = COLORS[i]
        ax1.plot(log['epoch'], log['train_loss'], '--', color=c, alpha=0.5)
        ax1.plot(log['epoch'], log['val_loss'], '-', color=c, label=label)
    
    ax1.set_title("Stage 1: Loss (Train -- / Val —)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f"{SAVE_DIR}/stage1_loss.png", bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {SAVE_DIR}/stage1_loss.png")

    # --- Dice ---
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for i in range(5):
        log = pd.read_csv(f"{WEIGHTS_DIR}/stage1/fold{i+1}/training_log.csv")
        c = COLORS[i]
        ax2.plot(log['epoch'], log['lung_Dice'], '-', color=c, label=f"Fold {i+1}")
    
    ax2.set_title("Stage 1: Validation Dice")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Dice Coefficient")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{SAVE_DIR}/stage1_dice.png", bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {SAVE_DIR}/stage1_dice.png")

    # --- IoU ---
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    for i in range(5):
        log = pd.read_csv(f"{WEIGHTS_DIR}/stage1/fold{i+1}/training_log.csv")
        c = COLORS[i]
        ax3.plot(log['epoch'], log['lung_IoU'], '-', color=c, label=f"Fold {i+1}")
    
    ax3.set_title("Stage 1: Validation IoU")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("IoU")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(f"{SAVE_DIR}/stage1_iou.png", bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {SAVE_DIR}/stage1_iou.png")


# ============================================================
#  Stage 2: Classification Curves
# ============================================================

def plot_stage2():
    # --- Loss ---
    fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
    for i in range(5):
        log = pd.read_csv(f"{WEIGHTS_DIR}/stage2/fold{i+1}/training_log.csv")
        c = COLORS[i]
        ax1.plot(log['epoch'], log['train_loss'], '--', color=c, alpha=0.5)
        ax1.plot(log['epoch'], log['val_loss'], '-', color=c, label=f"Fold {i+1}")
    ax1.set_title("Stage 2: Loss (Train -- / Val —)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f"{SAVE_DIR}/stage2_loss.png", bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {SAVE_DIR}/stage2_loss.png")

    # --- AUC-ROC ---
    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
    for i in range(5):
        log = pd.read_csv(f"{WEIGHTS_DIR}/stage2/fold{i+1}/training_log.csv")
        c = COLORS[i]
        ax2.plot(log['epoch'], log['auc_roc'], '-', color=c, label=f"Fold {i+1}")
    ax2.set_title("Stage 2: Validation AUC-ROC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{SAVE_DIR}/stage2_auc.png", bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {SAVE_DIR}/stage2_auc.png")

    # --- F1 ---
    fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=150)
    for i in range(5):
        log = pd.read_csv(f"{WEIGHTS_DIR}/stage2/fold{i+1}/training_log.csv")
        c = COLORS[i]
        ax3.plot(log['epoch'], log['f1'], '-', color=c, label=f"Fold {i+1}")
    ax3.set_title("Stage 2: Validation F1 Score")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(f"{SAVE_DIR}/stage2_f1.png", bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {SAVE_DIR}/stage2_f1.png")

    # --- Precision ---
    fig4, ax4 = plt.subplots(figsize=(6, 4), dpi=150)
    for i in range(5):
        log = pd.read_csv(f"{WEIGHTS_DIR}/stage2/fold{i+1}/training_log.csv")
        c = COLORS[i]
        ax4.plot(log['epoch'], log['precision'], '-', color=c, label=f"Fold {i+1}")
    ax4.set_title("Stage 2: Validation Precision")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Precision")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(f"{SAVE_DIR}/stage2_precision.png", bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {SAVE_DIR}/stage2_precision.png")

    # --- Recall ---
    fig5, ax5 = plt.subplots(figsize=(6, 4), dpi=150)
    for i in range(5):
        log = pd.read_csv(f"{WEIGHTS_DIR}/stage2/fold{i+1}/training_log.csv")
        c = COLORS[i]
        ax5.plot(log['epoch'], log['recall'], '-', color=c, label=f"Fold {i+1}")
    ax5.set_title("Stage 2: Validation Recall")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Recall")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(f"{SAVE_DIR}/stage2_recall.png", bbox_inches='tight')
    plt.close(fig5)
    print(f"Saved: {SAVE_DIR}/stage2_recall.png")



if __name__ == "__main__":
    plot_stage1()
    plot_stage2()
