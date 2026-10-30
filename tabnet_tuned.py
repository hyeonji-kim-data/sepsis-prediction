# =======================================
# TabNet (tuned with Optuna)
# =======================================

import os
import random
import numpy as np
import pandas as pd
import torch
import optuna
import matplotlib
matplotlib.use("Agg")   # Save figures to files without opening windows (must come before pyplot)
import matplotlib.pyplot as plt
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             confusion_matrix, accuracy_score, precision_score, f1_score)

torch.set_num_threads(1)   # Use a single thread for reproducible results
optuna.logging.set_verbosity(optuna.logging.WARNING)   # Hide Optuna's per-trial logs (own summary is printed instead)


# -------------------------------------------
# 0. Settings
# -------------------------------------------
MODEL_NAME = "TabNet_tuned"                        # Used for the output folder name
MODEL_LABEL = "TabNet (after Optuna tuning)"       # Used in figure titles

# True: run the Optuna tuning first
# False: skip the tuning and train with BEST_PARAMS below
RUN_OPTUNA = False
N_TRIALS = 50

# The train / validation split is fixed (random_state=42), so the seeds
# only change the initial weights and the batch order.
SEEDS = [0, 1, 2, 3, 4]

# Best parameters from an earlier Optuna run (5-fold CV AUC 0.8151).
# Other settings are fixed in fit_model() and are the same as in
# tabnet_default.py: cat_emb_dim=2, virtual_batch_size = batch_size / 4,
# class-balanced sampling (weights=1), drop_last=False,
# max 200 epochs, early stopping patience 20.
BEST_PARAMS = {
    'n_d': 16,                 # Width of the decision layer (n_a is set to the same value)
    'n_steps': 4,              # Number of decision steps
    'gamma': 1.4161830698351696,             # How much a feature can be reused across steps
    'lambda_sparse': 0.0003599339025369645,  # Penalty that makes the masks select fewer features
    'lr': 0.01726884470759044,
    'mask_type': 'entmax',
    'batch_size': 128,         # 4 updates per epoch with 509 training patients
}

OUT_DIR = os.path.join("figures", MODEL_NAME)
os.makedirs(OUT_DIR, exist_ok=True)


def set_seed(seed):
    """Fix the random seeds of Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -------------------------------------------
# 1. Load data and split train / validation
# -------------------------------------------
X_full_df = pd.read_csv("X_train.csv").astype(np.float32)
X_test_df = pd.read_csv("X_test.csv").astype(np.float32)
y_full = pd.read_csv("y_train.csv").squeeze().astype(np.int64).values   # 1 = death, 0 = survival
y_test = pd.read_csv("y_test.csv").squeeze().astype(np.int64).values

FEATURES = list(X_full_df.columns)

# Categorical features. All of them are assumed to be binary (0/1),
# because the model is built with 2 categories for each of them.
# Only the ones that exist in the data are kept.
# (Unlike FT-Transformer, TabNet takes all features in one array and
# only needs to know which columns are categorical.)
CATEGORICAL = [c for c in ['liver', 'Heart', 'Respiratory', 'multiorgan', 'Renal',
                           'Immunocompromised', 'arf', 'Sex', 'mv', 'vaso', 'AF_hos']
               if c in FEATURES]
CAT_IDX = [FEATURES.index(c) for c in CATEGORICAL]

X_full, X_test = X_full_df.values, X_test_df.values

# Hold out 20% of the training data as a validation set, stratified by outcome.
# The split is fixed (random_state=42) and shared by all seeds.
idx_tr, idx_va = train_test_split(np.arange(len(y_full)), test_size=0.2,
                                  stratify=y_full, random_state=42)
X_tr, y_tr = X_full[idx_tr], y_full[idx_tr]
X_va, y_va = X_full[idx_va], y_full[idx_va]

# Number of patients and deaths in each set
y_all = np.concatenate([y_full, y_test])
for name, y in [("Overall", y_all), ("Train (before split)", y_full),
                ("Train", y_tr), ("Validation", y_va), ("Test", y_test)]:
    print(f"{name}: n = {len(y)}, deaths = {y.sum()} ({y.mean():.1%})")


# -------------------------------------------
# 2. Train and predict
# -------------------------------------------
def fit_model(params, X_a, y_a, X_b, y_b, seed):
    """Train on (X_a, y_a) with early stopping on the AUC of (X_b, y_b).

    pytorch-tabnet restores the weights from the best epoch automatically
    at the end of training.
    """
    set_seed(seed)
    bs = params['batch_size']
    model = TabNetClassifier(
        n_d=params['n_d'], n_a=params['n_d'],     # n_a = n_d, as recommended in the TabNet paper
        n_steps=params['n_steps'],
        gamma=params['gamma'], lambda_sparse=params['lambda_sparse'],
        optimizer_fn=torch.optim.Adam, optimizer_params={'lr': params['lr']},
        mask_type=params['mask_type'],
        cat_idxs=CAT_IDX,
        cat_dims=[2] * len(CAT_IDX),              # 2 categories (0/1) for each feature
        cat_emb_dim=2,                            # Embedding size for each categorical feature
        seed=seed, verbose=0, device_name='cpu')
    model.fit(X_train=X_a, y_train=y_a, eval_set=[(X_b, y_b)], eval_name=['val'],
              eval_metric=['auc'], max_epochs=200, patience=20,
              batch_size=bs,
              virtual_batch_size=max(16, bs // 4),   # Size used for Ghost Batch Normalization
              # weights=1: sample deaths and survivors in equal proportions in each batch.
              # This handles the class imbalance in training, which FT-Transformer does not do.
              weights=1,
              # drop_last=False: keep the last incomplete batch, so no patient is left out.
              drop_last=False)
    return model


def predict(model, X):
    """Return the predicted probability of death for each row of X.

    No shuffling, so the output lines up with the labels (y).
    """
    return model.predict_proba(X)[:, 1]   # Column 1 = probability of class 1 (death)


# -------------------------------------------
# 3. Optuna tuning (only when RUN_OPTUNA = True)
# -------------------------------------------
def objective(trial):
    """Return the mean 5-fold CV AUC for one set of hyperparameters.

    Uses all 637 training patients (the validation set is included in
    the folds). The test set is never used here.
    Note: early stopping in each fold uses the same fold that is scored,
    so the CV AUC is slightly optimistic. It is only used to compare
    settings, not to report performance.
    """
    # Search space. Batch sizes are much smaller than the default (1024),
    # so there are several updates per epoch.
    params = {
        'n_d': trial.suggest_categorical('n_d', [8, 16, 32, 64]),
        'n_steps': trial.suggest_int('n_steps', 3, 7),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-2, log=True),   # Searched on a log scale
        'lr': trial.suggest_float('lr', 1e-3, 3e-2, log=True),
        'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # Same folds for every trial
    aucs = []
    for k, (a, b) in enumerate(skf.split(X_full, y_full)):
        model = fit_model(params, X_full[a], y_full[a], X_full[b], y_full[b], seed=k)
        aucs.append(roc_auc_score(y_full[b], predict(model, X_full[b])))

        # Report the running mean after each fold. The pruner stops the trial
        # early if it is clearly worse than earlier trials at the same fold.
        trial.report(float(np.mean(aucs)), k)
        if trial.should_prune():
            raise optuna.TrialPruned()

    print(f"trial {trial.number:2d} | CV AUC {np.mean(aucs):.4f}")
    return float(np.mean(aucs))


study = None
if RUN_OPTUNA:
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),   # seed=42 makes the search reproducible
        # Pruning starts after the first 8 trials, and from the second fold on
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    study.optimize(objective, n_trials=N_TRIALS)
    params = study.best_params
    print(f"\nBest CV AUC: {study.best_value:.4f}\nBest params: {params}")
    study.trials_dataframe().to_csv(os.path.join(OUT_DIR, "optuna_trials.csv"), index=False)
else:
    params = BEST_PARAMS


# -------------------------------------------
# 4. Metric functions
# -------------------------------------------
def threshold_f1(y, prob):
    """Return the threshold that gives the highest F1 score.

    Used on the validation set. The class imbalance is also handled here,
    by moving the threshold away from 0.5.
    F1 is used because it balances recall (missed deaths) and precision
    (false alarms), while accuracy can look high just by predicting
    survival for most patients.
    """
    p, r, t = precision_recall_curve(y, prob)
    f1 = 2 * p * r / (p + r + 1e-12)     # 1e-12 avoids division by zero
    # p and r have one more value than t (the last point has no threshold),
    # so the last F1 value is dropped.
    return float(t[np.argmax(f1[:-1])])


def evaluate(y, prob, th):
    """Return metrics when patients with prob >= th are classified as death."""
    pred = (prob >= th).astype(int)
    # labels=[0, 1] keeps the matrix 2x2 even if only one class is predicted
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'Threshold': th, 'AUC': roc_auc_score(y, prob), 'Accuracy': accuracy_score(y, pred),
            'Precision': precision_score(y, pred, zero_division=0),
            'Recall': tp / (tp + fn),          # Share of deaths detected (sensitivity)
            'Specificity': tn / (tn + fp),     # Share of survivors correctly identified
            'F1 Score': f1_score(y, pred, zero_division=0)}


def oof_probs(params, seed):
    """Return cross-validated predicted probabilities for all
        637 training patients. Used only to choose the decision threshold."""
    oof = np.zeros(len(y_full))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for a, b in skf.split(X_full, y_full):
        # Within the 4 training folds, hold out 20% for early stopping only
        a_tr, a_es = train_test_split(a, test_size=0.2, stratify=y_full[a], random_state=seed)
        m = fit_model(params, X_full[a_tr], y_full[a_tr], X_full[a_es], y_full[a_es], seed)
        oof[b] = predict(m, X_full[b])   # Predict only the fold not used for training
    return oof


def permutation_importance_auc(model, X, y, seed, n_repeats=5):
    """Return how much the AUC drops when each feature is shuffled.

    A larger drop means the model relies more on that feature.
    Each feature is shuffled n_repeats times and the drops are averaged.
    Note: if two features are strongly correlated, shuffling one of them
    may cause only a small drop, so both can look less important.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict(model, X))
    drops = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        d = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])   # Shuffle only feature j
            d.append(base - roc_auc_score(y, predict(model, Xp)))
        drops[j] = np.mean(d)
    return drops


# -------------------------------------------
# 5. Final training over 5 seeds and summary
# -------------------------------------------
oofs = []
rows, probs_va, probs_te, perm_imps, mask_imps = [], [], [], [], []
for seed in SEEDS:
    model = fit_model(params, X_tr, y_tr, X_va, y_va, seed)
    pv, pt = predict(model, X_va), predict(model, X_test)
    probs_va.append(pv); probs_te.append(pt)

    # Choose the threshold on the validation set, then apply it to the test set.
    # Test results are reported at both 0.5 and this threshold.
    # (AUC does not depend on the threshold, so it is the same for both.)
    oof = oof_probs(params, seed); oofs.append(oof)
    th = threshold_f1(y_full, oof)    # Threshold from the 637 cross-validated predictions
    r05, rth = evaluate(y_test, pt, 0.5), evaluate(y_test, pt, th)
    rows.append({'seed': seed, **{f"{k} (0.5)": v for k, v in r05.items() if k != 'Threshold'},
                 **{f"{k} (CV threshold)": v for k, v in rth.items()}})

    # Permutation importance on the test set (the validation set is small).
    # It is only used to describe the model, not to choose features or settings.
    # Calculated the same way as for FT-Transformer, so the two can be compared.
    perm_imps.append(permutation_importance_auc(model, X_test, y_test, seed))

    # TabNet's built-in importance: how much each feature is selected by the
    # attention masks, computed on the training data. Sums to 1.
    mask_imps.append(model.feature_importances_)

    print(f"seed {seed} | Test AUC {r05['AUC']:.4f} | Recall @0.5: {r05['Recall']:.3f} -> "
          f"@CV threshold ({th:.3f}): {rth['Recall']:.3f}")

# Mean and SD over seeds (ddof=0: SD of these 5 runs, not a sample estimate)
result = pd.DataFrame(rows)
result.to_csv(os.path.join(OUT_DIR, "metrics_by_seed.csv"), index=False)
m = result.drop(columns='seed')
summary = pd.DataFrame({'mean': m.mean(), 'std': m.std(ddof=0)}).round(4)
summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"))
print(f"\nSummary - {MODEL_LABEL} (test set, mean ± SD over {len(SEEDS)} seeds)")
print(summary.to_string())

# The figures use the average of the 5 seeds' predicted probabilities,
# with a threshold chosen again on the averaged validation probabilities.
# So the figures can differ slightly from the mean values in the summary table.
pv_ens, pt_ens = np.mean(probs_va, axis=0), np.mean(probs_te, axis=0)
th_ens = threshold_f1(y_full, np.mean(oofs, axis=0))
SUB = f"Averaged predictions of {len(SEEDS)} seeds · Test n = {len(y_test)} (deaths {y_test.sum()})"   # Figure caption


# -------------------------------------------
# 6. Plot style
# -------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": "#0b0b0b",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
})

# Colors
C_BLUE = "#2a78d6"     # Survival, ROC curve, specificity, feature importance, CV AUC per trial
C_ORANGE = "#eb6834"   # Death, recall, best CV AUC so far
C_AQUA = "#1baf7a"     # F1
C_TEXT = "#0b0b0b"     # Text and threshold lines
C_MUTED = "#8a8984"    # Captions, random-guess line, error bars


# -------------------------------------------
# 7. Figures
# -------------------------------------------
def plot_confusion():
    """Confusion matrices on the test set: threshold 0.5 vs. CV-based threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"wspace": 0.55})
    for ax, th, name in [(axes[0], 0.5, "Default threshold 0.5"),
                         (axes[1], th_ens, f"CV-based threshold {th_ens:.2f}")]:
        cm = confusion_matrix(y_test, (pt_ens >= th).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                pct = cm[i, j] / cm[i].sum() * 100     # Share within each actual class (row)
                ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}%)", ha="center", va="center",
                        fontsize=12, color="white" if cm[i, j] > cm.max() * 0.6 else C_TEXT)   # White text on dark cells
        ax.set_xticks([0, 1], ["Predicted survival", "Predicted death"])
        ax.set_yticks([0, 1], ["Actual survival", "Actual death"])
        recall = cm[1, 1] / cm[1].sum()
        ax.set_title(f"{name}\nRecall {recall:.0%} · Missed deaths {cm[1, 0]}", fontsize=11)
    fig.suptitle(f"{MODEL_LABEL} Confusion Matrix", fontweight="bold", y=1.04)
    fig.text(0.5, -0.02, SUB, ha="center", color=C_MUTED, fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, "01_confusion_matrix.png")); plt.close(fig)


def plot_roc():
    """ROC curves on the test set: each seed (thin) and seed-averaged prediction (thick).

    The title shows the mean ± SD of the per-seed AUCs, and the legend shows
    the AUC of the seed-averaged prediction, so the two values can differ.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for i, pt in enumerate(probs_te):
        fpr, tpr, _ = roc_curve(y_test, pt)
        ax.plot(fpr, tpr, color=C_BLUE, alpha=0.25, lw=1,
                label="Each seed" if i == 0 else None)   # Label only once in the legend
    fpr, tpr, _ = roc_curve(y_test, pt_ens)
    ax.plot(fpr, tpr, color=C_BLUE, lw=2.2,
            label=f"Seed-averaged prediction (AUC {roc_auc_score(y_test, pt_ens):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=C_MUTED, lw=1, label="Random (AUC 0.5)")
    aucs = result["AUC (0.5)"]
    ax.set_title(f"{MODEL_LABEL} ROC Curve\nAUC per seed {aucs.mean():.3f} ± {aucs.std(ddof=0):.3f}", fontsize=11)
    ax.set_xlabel("1 - Specificity (false positive rate)")
    ax.set_ylabel("Recall (true positive rate)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, "02_roc_curve.png")); plt.close(fig)


def plot_prob_dist():
    """Histogram of predicted probabilities on the test set, by actual outcome."""
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 26)     # 25 bins of width 0.04
    ax.hist(pt_ens[y_test == 0], bins=bins, color=C_BLUE, alpha=0.55, edgecolor="white",
            label=f"Actual survival (n = {(y_test == 0).sum()})")
    ax.hist(pt_ens[y_test == 1], bins=bins, color=C_ORANGE, alpha=0.75, edgecolor="white",
            label=f"Actual death (n = {(y_test == 1).sum()})")
    ax.axvline(0.5, color=C_TEXT, ls="--", lw=1.2)
    ax.axvline(th_ens, color=C_TEXT, ls=":", lw=1.5)
    ymax = ax.get_ylim()[1]
    left_first = th_ens < 0.5   # Place the two labels on opposite sides so they do not overlap
    ax.text(0.5, ymax * 0.97, " Default 0.5 ", va="top", fontsize=9, color=C_TEXT,
            ha="left" if left_first else "right")
    ax.text(th_ens, ymax * 0.97, f" CV-based {th_ens:.2f} ", va="top", fontsize=9, color=C_TEXT,
            ha="right" if left_first else "left")
    ax.set_xlabel("Predicted probability of death"); ax.set_ylabel("Number of patients")
    ax.set_title(f"{MODEL_LABEL} Predicted Probability Distribution", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1, 0.7))
    fig.text(0.5, -0.07, SUB, ha="center", color=C_MUTED, fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, "03_probability_distribution.png")); plt.close(fig)


def plot_threshold_curve():
    """Recall, specificity and F1 on the test set for thresholds from 0.01 to 0.99.

    For illustration only: the threshold used in this study was chosen
    on the validation set, not from this curve.
    """
    ths = np.linspace(0.01, 0.99, 99)
    rec, spec, f1s = [], [], []
    for t in ths:
        pred = (pt_ens >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        rec.append(tp / (tp + fn)); spec.append(tn / (tn + fp))
        f1s.append(f1_score(y_test, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ths, rec, color=C_ORANGE, lw=2, label="Recall (deaths detected)")
    ax.plot(ths, spec, color=C_BLUE, lw=2, label="Specificity (survivors correctly identified)")
    ax.plot(ths, f1s, color=C_AQUA, lw=2, label="F1")
    left_first = th_ens < 0.5   # Place the two labels on opposite sides so they do not overlap
    for t, ls, name, ha in [(0.5, "--", " Default 0.5 ", "left" if left_first else "right"),
                            (th_ens, ":", f" CV-based {th_ens:.2f} ", "right" if left_first else "left")]:
        ax.axvline(t, color=C_TEXT, ls=ls, lw=1.2)
        ax.text(t, 1.02, name, ha=ha, fontsize=9, color=C_TEXT)
    ax.set_xlabel("Threshold (classified as death at or above this probability)")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"{MODEL_LABEL} Recall, Specificity and F1 by Threshold", fontsize=11, pad=16)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3),
              ncol=3, handlelength=1.5, columnspacing=1.5)
    fig.text(0.5, -0.07, SUB, ha="center", color=C_MUTED, fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, "04_threshold_curve.png")); plt.close(fig)


def plot_importance(values, fname, title, xlabel):
    """Bar chart of the top 10 features (mean ± SD over seeds).

    Also saves the values for all features as a CSV file.
    """
    values = np.array(values)                     # Shape: (seeds, features)
    mean, std = values.mean(axis=0), values.std(axis=0)
    # Sort from largest to smallest, keep the top 10, then reverse again
    # because barh draws from the bottom up (so the largest ends up on top)
    order = np.argsort(mean)[::-1][:10][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([FEATURES[i] for i in order], mean[order], xerr=std[order], color=C_BLUE,
            height=0.6, error_kw={"ecolor": C_MUTED, "lw": 1, "capsize": 2})
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    fig.text(0.5, -0.07, f"Mean ± SD over {len(SEEDS)} seeds · Top 10", ha="center",
             color=C_MUTED, fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, fname)); plt.close(fig)
    pd.DataFrame({"feature": FEATURES, "mean": mean, "std": std}) \
        .sort_values("mean", ascending=False) \
        .to_csv(os.path.join(OUT_DIR, fname.replace(".png", ".csv")), index=False)


def plot_optuna_history():
    """CV AUC of each completed Optuna trial and the best value so far.

    Pruned trials are left out because they stopped before all 5 folds.
    """
    df = study.trials_dataframe()
    done = df[df["state"] == "COMPLETE"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(done["number"], done["value"], s=22, color=C_BLUE, alpha=0.7, label="CV AUC per trial")
    ax.plot(done["number"], done["value"].cummax(), color=C_ORANGE, lw=2, label="Best so far")
    ax.set_xlabel("Optuna trial"); ax.set_ylabel("5-fold CV AUC")
    ax.set_title(f"{MODEL_LABEL} Optuna Tuning History (pruned trials excluded)", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, "06_optuna_history.png")); plt.close(fig)


plot_confusion()
plot_roc()
plot_prob_dist()
plot_threshold_curve()
plot_importance(perm_imps, "05_permutation_importance.png",
                f"{MODEL_LABEL} Permutation Importance", "AUC drop when the feature is shuffled")
plot_importance(mask_imps, "05b_mask_importance.png",          # TabNet only
                f"{MODEL_LABEL} Mask-based Feature Importance", "TabNet mask importance (sums to 1)")
if study is not None:          # Only when the tuning was run in this session
    plot_optuna_history()
else:
    print("\nRUN_OPTUNA = False, so plot 06 (Optuna tuning history) is skipped")

print(f"\nFigures saved to: {OUT_DIR}/")