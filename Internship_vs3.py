 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cimcb_lite as cb

print('All packages successfully loaded')


#%% use venv environment 
filename = "Internship_GastricCancer_NMR.xlsx"
dataTable, peakTable = cb.utils.load_dataXL(filename, DataSheet="Data", PeakSheet="Peak")
print("Data shape:", getattr(dataTable, "shape", None))
print("Peak shape:", getattr(peakTable, "shape", None))

import pandas as pd, shutil
pd.set_option("display.max_columns", None)
pd.set_option("display.width", shutil.get_terminal_size(fallback=(120,20)).columns)

print("\n=== dataTable (first 20) ===")
print(dataTable.head(20).to_string(index=False))
print("\n=== peakTable (first 20) ===")
print(peakTable.head(20).to_string(index=False))


#%% Create a clean peak table 
rsd = peakTable['QC_RSD']  
percMiss = peakTable['Perc_missing']  
peakTableClean = peakTable[(rsd < 20) & (percMiss < 10)]   

print("Number of peaks remaining: {}".format(len(peakTableClean))) #number of peak remaining 52


# %% View: peakTableClean
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
peakTableClean.head(20)                  


#%% packages for plotting
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (must be imported before)
from sklearn.impute import IterativeImputer


#%% Impute with Random Forest + PCA 

# ---- 1) Feature list from peak table ----
if "peakTableClean" in globals():
    names = pd.Index(peakTableClean["Name"])
else:
    names = pd.Index(peakTable["Name"])

# keep only names that exist as columns in dataTable, preserve order & drop dups
seen = set()
peaklist = [c for c in names if c in dataTable.columns and not (c in seen or seen.add(c))]

if not peaklist:
    raise ValueError("No overlap between peak names and dataTable columns.")

# ---- 2) Build matrix, log10, mark invalids as NaN ----
X = dataTable[peaklist].to_numpy(dtype=float)

# avoid -inf from log10 on non-positive values
X = np.where(X <= 0, np.nan, X)
Xlog = np.log10(X)

# drop all-NaN columns (IterativeImputer can't handle them)
valid_cols = ~np.all(np.isnan(Xlog), axis=0)
if not np.all(valid_cols):
    dropped = [peaklist[i] for i, ok in enumerate(valid_cols) if not ok]
    print(f"Dropped {len(dropped)} all-NaN features:", dropped[:10], "..." if len(dropped) > 10 else "")
    peaklist = [p for p, ok in zip(peaklist, valid_cols) if ok]
    Xlog = Xlog[:, valid_cols]

# ---- 3) Random-Forest imputation on log data ----
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=0,
    n_jobs=-1
)
imputer = IterativeImputer(
    estimator=rf,
    max_iter=10,
    initial_strategy="median",
    skip_complete=True,
    random_state=0
)
X_imp = imputer.fit_transform(Xlog)

# ---- 4) Scale (for PCA) ----
Xscale = cb.utils.scale(X_imp, method="auto")   # auto/pareto/vast/level

print("Xscale (RF-imputed):", *Xscale.shape, " [rows, cols]")

# ---- 5) PCA (scores + simple loadings) ----
group_col = "SampleType" if "SampleType" in dataTable.columns else ("Class" if "Class" in dataTable.columns else None)
groups = dataTable[group_col].astype(str).values if group_col else None

pca = PCA(n_components=2, random_state=0)
PCs = pca.fit_transform(Xscale)
exp = pca.explained_variance_ratio_ * 100

# PC1 vs PC2 plot
plt.figure(figsize=(6.8, 5.6))
if groups is not None:
    for g in sorted(pd.unique(groups)):
        idx = (groups == g)
        plt.scatter(PCs[idx, 0], PCs[idx, 1], label=str(g), alpha=0.9)
    plt.legend(title=group_col)
else:
    plt.scatter(PCs[:, 0], PCs[:, 1], alpha=0.9)
plt.xlabel(f"PC1 ({exp[0]:.1f}%)")
plt.ylabel(f"PC2 ({exp[1]:.1f}%)")
plt.title("PCA Scores: PC1 vs PC2 (RF-imputed, auto-scaled)")
plt.tight_layout()
plt.show()

# barplot: top15 metabolites from PC1
loadings = pca.components_.T                 # shape: (n_features, 2)
feat_names = peaklist
topk = min(15, len(feat_names))
idx_pc1 = np.argsort(np.abs(loadings[:, 0]))[-topk:][::-1]

plt.figure(figsize=(8, 6))
plt.barh([feat_names[i] for i in idx_pc1], loadings[idx_pc1, 0])
plt.gca().invert_yaxis()
plt.xlabel("PC1 loading")
plt.title(f"Top {topk} |PC1| loadings")
plt.tight_layout()
plt.show()


#univariate comparison of HC vs HE patient data
# %% Select subset of Data for statistical comparison
dataTable2 = dataTable[(dataTable.Class == "GC") | (dataTable.Class == "HE")]  # Reduce data table only to GC and HE class members
pos_outcome = "GC" 

# Calculate basic statistics and create a statistics table.
statsTable = cb.utils.univariate_2class(dataTable2,
                                        peakTableClean,
                                        group='Class',                # Column used to determine the groups
                                        posclass=pos_outcome,         # Value of posclass in the group column
                                        parametric=True)              # Set parametric = True or False

# %% View: statsTable
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
statsTable.head(20) 


# %% Save StatsTable to Excel
statsTable.to_excel("stats.xlsx", sheet_name='StatsTable', index=False)
print("done!")


                                             
# %% Create a Binary Y vector for stratifiying the samples
outcomes = dataTable2['Class']                                  
Y = [1 if outcome == 'GC' else 0 for outcome in outcomes]       # Change Y into binary (GC = 1, HE = 0)  
Y = np.array(Y)                                                 # convert boolean list into to a numpy array

# Split DataTable2 and Y into train and test (with stratification)
dataTrain, dataTest, Ytrain, Ytest = train_test_split(dataTable2, Y, test_size=0.25, stratify=Y, random_state=10)

print("DataTrain = {} samples with {} postive cases.".format(len(Ytrain),sum(Ytrain)))
print("DataTest = {} samples with {} postive cases.".format(len(Ytest),sum(Ytest)))


# %% Extract and scale the metabolite data from the dataTable
peaklist = peakTableClean['Name']                           
XT = dataTrain[peaklist]                                    
XTlog = np.log(XT)                                         
XTscale = cb.utils.scale(XTlog, method='auto')              
XTknn = cb.utils.knnimpute(XTscale, k=3)                    


# %% initalise cross_val kfold (stratified) 
cv = cb.cross_val.kfold(model=cb.model.PLS_SIMPLS,                   
                        X=XTknn,                                 
                        Y=Ytrain,                               
                        param_dict={'n_components': [1,2,3,4,5,6]},                 
                        folds=5,                                     
                        bootnum=100)                                

# run the cross validation
cv.run()  


#%% Plot CV performance (PLS-DA) without Bokeh — uses XTknn and Ytrain
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# Sanity checks
X = XTknn
y = np.asarray(Ytrain).astype(int)
assert X.shape[0] == y.shape[0], f"Row mismatch: X has {X.shape[0]}, y has {y.shape[0]}"

# Use the same grid you passed to cb.cross_val.kfold
n_comp_list = [1, 2, 3, 4, 5, 6]
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

auc_means, auc_cis, acc_means, acc_cis = [], [], [], []

for ncomp in n_comp_list:
    auc_scores, acc_scores = [], []
    for tr, te in skf.split(X, y):
        pls = PLSRegression(n_components=ncomp)
        pls.fit(X[tr], y[tr])

        # Continuous scores → AUC
        proba = np.clip(pls.predict(X[te]).ravel(), 0, 1)
        auc_scores.append(roc_auc_score(y[te], proba))

        # Threshold at 0.5 → Accuracy
        pred_cls = (proba > 0.5).astype(int)
        acc_scores.append(accuracy_score(y[te], pred_cls))

    # mean ± 95% CI across folds
    auc_scores = np.array(auc_scores)
    acc_scores = np.array(acc_scores)
    auc_means.append(auc_scores.mean())
    acc_means.append(acc_scores.mean())
    auc_cis.append(1.96 * auc_scores.std(ddof=1) / np.sqrt(k))
    acc_cis.append(1.96 * acc_scores.std(ddof=1) / np.sqrt(k))

auc_means = np.array(auc_means); auc_cis = np.array(auc_cis)
acc_means = np.array(acc_means); acc_cis = np.array(acc_cis)

best_auc_idx = int(np.argmax(auc_means))
best_acc_idx = int(np.argmax(acc_means))

# ---- Plot AUC ----
plt.figure(figsize=(6.8, 5.0))
plt.errorbar(n_comp_list, auc_means, yerr=auc_cis, fmt='-o')
plt.axvline(n_comp_list[best_auc_idx], linestyle='--', linewidth=1)
plt.xlabel("Number of latent variables (PLS components)")
plt.ylabel("CV AUC")
plt.title("PLS-DA k-fold CV (AUC)")
plt.tight_layout()
plt.show()

# ---- Plot Accuracy (optional) ----
plt.figure(figsize=(6.8, 5.0))
plt.errorbar(n_comp_list, acc_means, yerr=acc_cis, fmt='-o')
plt.axvline(n_comp_list[best_acc_idx], linestyle='--', linewidth=1)
plt.xlabel("Number of latent variables (PLS components)")
plt.ylabel("CV Accuracy")
plt.title("PLS-DA k-fold CV (Accuracy)")
plt.tight_layout()
plt.show()

print(f"Best AUC at n_components={n_comp_list[best_auc_idx]}: {auc_means[best_auc_idx]:.3f} ± {auc_cis[best_auc_idx]:.3f}")
print(f"Best ACC at n_components={n_comp_list[best_acc_idx]}: {acc_means[best_acc_idx]:.3f} ± {acc_cis[best_acc_idx]:.3f}")


# %% PLS-DA k-fold CV with matplotlib (no Bokeh)
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

# --- 1) Choose X (your RF-imputed, scaled matrix) and restrict to GC vs HE rows ---
# If you want to use a different matrix (e.g., XTknn), just set X_cv = XTknn instead.
X_full = Xscale

if "Class" not in dataTable.columns:
    raise ValueError("dataTable must have a 'Class' column for CV labels.")

mask = dataTable["Class"].astype(str).isin(["GC", "HE"])
X_cv = X_full[mask.values, :]
y_labels = dataTable.loc[mask, "Class"].astype(str).values

# map to 0/1 for AUC (GC=1 positive class)
y_cv = (y_labels == "GC").astype(int)

# sanity checks
if X_cv.shape[0] != y_cv.shape[0]:
    raise ValueError("Row count mismatch between X and y after filtering.")
if len(np.unique(y_cv)) != 2:
    raise ValueError("Need a binary outcome (GC vs HE) for AUC.")

# --- 2) CV grid over n_components ---
n_comp_list = [1, 2, 3, 4, 5, 6]
k = 5
cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

means, cis, all_scores = [], [], []

for ncomp in n_comp_list:
    fold_scores = []
    for tr, te in cv.split(X_cv, y_cv):
        pls = PLSRegression(n_components=ncomp)
        pls.fit(X_cv[tr], y_cv[tr])
        # Predict continuous scores and clip to [0,1] for AUC
        y_pred = np.clip(pls.predict(X_cv[te]).ravel(), 0, 1)
        fold_scores.append(roc_auc_score(y_cv[te], y_pred))
    fold_scores = np.array(fold_scores, float)
    mean = fold_scores.mean()
    # 95% CI via t-approx across folds
    sd = fold_scores.std(ddof=1)
    ci = 1.96 * sd / np.sqrt(k)
    means.append(mean); cis.append(ci); all_scores.append(fold_scores)

means = np.array(means); cis = np.array(cis)
best_idx = int(np.argmax(means))

# --- 3) Plot (matplotlib) ---
plt.figure(figsize=(6.8, 5.2))
plt.errorbar(n_comp_list, means, yerr=cis, fmt='-o')
plt.axvline(n_comp_list[best_idx], linestyle='--', linewidth=1)
plt.xlabel("Number of latent variables (PLS components)")
plt.ylabel("CV AUC (GC vs HE)")
plt.title("PLS-DA k-fold cross-validation")
plt.tight_layout()
plt.show()

print(f"Best n_components = {n_comp_list[best_idx]}  |  CV AUC = {means[best_idx]:.3f} ± {cis[best_idx]:.3f}")


# %% PLS-DA: R² & Q² vs number of components (matplotlib)
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold

X = XTknn if 'XTknn' in globals() else Xscale
y = np.asarray(Ytrain if 'Ytrain' in globals() else Y).astype(float).ravel()

# sanity check
assert X.shape[0] == y.shape[0], f"Row mismatch: X={X.shape[0]} vs y={y.shape[0]}"

# Treat binary labels as 0/1 for PLS-DA
if y.dtype.kind not in "iu":  # not integer
    y_int = (y > 0.5).astype(int)  # safe if y already 0/1 floats
else:
    y_int = y.astype(int)

# ---------------- grid and CV setup ----------------
n_comp_list = [1, 2, 3, 4, 5, 6]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

r2_vals, q2_vals = [], []

for ncomp in n_comp_list:
    # --- R² (fit on all X,y you provided here) ---
    pls = PLSRegression(n_components=ncomp)
    pls.fit(X, y)
    r2_vals.append(float(pls.score(X, y)))  # coefficient of determination on the fit

    # --- Q² (out-of-fold CV predictions aggregated across folds) ---
    yhat_oof = np.empty_like(y, dtype=float)
    for tr, te in kfold.split(X, y_int):
        pls_cv = PLSRegression(n_components=ncomp)
        pls_cv.fit(X[tr], y[tr])
        yhat_oof[te] = pls_cv.predict(X[te]).ravel()

    # clip to [0,1] since we're modeling a binary outcome
    yhat_oof = np.clip(yhat_oof, 0.0, 1.0)
    press = np.sum((y - yhat_oof) ** 2)                 # prediction error sum of squares
    sst   = np.sum((y - y.mean()) ** 2)                 # total sum of squares
    q2    = 1.0 - (press / sst)
    q2_vals.append(float(q2))

# ---------------- plot ----------------
plt.figure(figsize=(7.2, 5.2))
plt.plot(n_comp_list, r2_vals, marker='o', label='R² (fit)')
plt.plot(n_comp_list, q2_vals, marker='o', label='Q² (5-fold CV)')
plt.xlabel('Number of PLS components')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('PLS-DA — R² & Q² vs components')
plt.legend()
plt.tight_layout()
plt.show()

best_idx = int(np.argmax(q2_vals))
print(f"Best Q² at {n_comp_list[best_idx]} components: {q2_vals[best_idx]:.3f} (R²={r2_vals[best_idx]:.3f})")

 
#%% Build Xtest with the SAME preprocessing as training (log -> scale by train μ/σ -> KNN impute)
# 1) Recreate the training log matrix (to get μ/σ from TRAIN only)
XT = dataTrain[peaklist].to_numpy(dtype=float)
XTlog = np.log(XT)

# training mean/std (ignore NaNs)
mu = np.nanmean(XTlog, axis=0)
sigma = np.nanstd(XTlog, axis=0, ddof=1)
sigma[sigma == 0] = 1.0

# 2) Test set log + scale using TRAIN μ/σ
XTe = dataTest[peaklist].to_numpy(dtype=float)
XTelog = np.log(XTe)
XTe_scale = (XTelog - mu) / sigma

# 3) KNN impute test (same k=3 as train)
XTest_knn = cb.utils.knnimpute(XTe_scale, k=3)

print("XTest_knn shape:", XTest_knn.shape)


#%% Fit FINAL PLS-DA with optimal n_components and evaluate (train & test)
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

# Fixed number of components
ncomp = 2
print(f"Using n_components = {ncomp}")

# Train PLS on TRAIN matrix
pls_final = PLSRegression(n_components=ncomp)
pls_final.fit(XTknn, np.asarray(Ytrain).astype(float))

# --- TRAIN evaluation
scores_tr = np.clip(pls_final.predict(XTknn).ravel(), 0, 1)
pred_tr   = (scores_tr >= 0.5).astype(int)
auc_tr    = roc_auc_score(Ytrain, scores_tr)
acc_tr    = accuracy_score(Ytrain, pred_tr)

# --- TEST evaluation
scores_te = np.clip(pls_final.predict(XTest_knn).ravel(), 0, 1)
pred_te   = (scores_te >= 0.5).astype(int)
auc_te    = roc_auc_score(Ytest, scores_te)
acc_te    = accuracy_score(Ytest, pred_te)

print(f"[TRAIN] AUC={auc_tr:.3f}  ACC={acc_tr:.3f}  (n={len(Ytrain)})")
print(f"[TEST ] AUC={auc_te:.3f}  ACC={acc_te:.3f}  (n={len(Ytest)})")
print("Confusion matrix [TEST]:\n", confusion_matrix(Ytest, pred_te))

# ROC (TEST)
fpr, tpr, _ = roc_curve(Ytest, scores_te)
plt.figure(figsize=(6.2, 5.0))
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("PLS-DA ROC (TEST) — n_comp=2")
plt.tight_layout(); plt.show()


#%% VIP scores (shape-safe) + top features

def vip_scores_sklearn(pls):
    """
    VIP for sklearn.cross_decomposition.PLSRegression (single y).
    Formula: VIP_j = sqrt( p * sum_a( SSY_a * w_{ja}^2 / sum_j w_{ja}^2 ) / sum_a SSY_a )
    where SSY_a = sum_i t_{ia}^2 * q_a^2
    """
    T = pls.x_scores_      # (n_samples, A)
    W = pls.x_weights_     # (p, A)
    Q = pls.y_loadings_    # (1, A) for single target

    # Defensive shapes
    if Q.ndim == 2:
        q = Q.reshape(-1)              # -> (A,)
        if q.size != W.shape[1]:
            # If somehow more values, take the first A
            q = q[:W.shape[1]]
    else:
        q = Q                          # already (A,)
    p, A = W.shape

    # SSY per component (amount of y explained by each component)
    ssy = (T**2).sum(axis=0) * (q**2)  # (A,)

    # Avoid division by zero if a component has zero weight norm
    wnorm2 = (W**2).sum(axis=0)        # (A,)
    wnorm2[wnorm2 == 0] = np.finfo(float).eps

    # VIP per feature
    vip = np.sqrt(p * ((W**2 / wnorm2) @ ssy) / ssy.sum())
    return vip  # (p,)

# Compute VIP on your fitted model
vip = vip_scores_sklearn(pls_final)          # pls_final from your previous cell
vip_df = pd.DataFrame({"Feature": list(peaklist), "VIP": vip})
vip_df = vip_df.sort_values("VIP", ascending=False).reset_index(drop=True)

print("\nTop 15 VIP features:")
print(vip_df.head(15).to_string(index=False))

# Plot top 15
topk = min(15, len(vip_df))
plt.figure(figsize=(8, 6))
plt.barh(vip_df.loc[:topk-1, "Feature"], vip_df.loc[:topk-1, "VIP"])
plt.gca().invert_yaxis()
plt.axvline(1.0, linestyle="--", linewidth=1)   # common VIP>1 heuristic
plt.xlabel("VIP")
plt.title(f"Top {topk} VIP features (n_components={pls_final.n_components})")
plt.tight_layout()
plt.show()

# Save full VIP table
vip_df.to_csv("vip_PLSDa_train.csv", index=False)
print("Saved VIP table to vip_PLSDa_train.csv")


#%% Train PLS-DA with cimcb_lite (n_comp=2), evaluate & plot (TRAIN + optional TEST) — no Bokeh
from matplotlib import gridspec
from sklearn.metrics import (
    r2_score, roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, confusion_matrix, f1_score
)
from scipy.stats import mannwhitneyu, gaussian_kde
from sklearn.cross_decomposition import PLSRegression

def pretty_pls_eval(scores, y, cutoff=0.5, title_tag="Train", n_boot=800):
    """CIMCB-style violin + PDF + ROC (with bootstrap CI) + metrics table; robust formatting."""
    scores = np.asarray(scores).ravel()
    y = np.asarray(y).astype(int)

    # safety
    scores = np.nan_to_num(scores, nan=0.5)
    scores = np.clip(scores, 0.0, 1.0)

    # ----- point metrics -----
    pred = (scores >= cutoff).astype(int)
    r2   = r2_score(y, scores)
    if len(np.unique(y)) == 2:
        mw_p = mannwhitneyu(scores[y==0], scores[y==1], alternative="two-sided").pvalue
        auc  = roc_auc_score(y, scores)
    else:
        mw_p = np.nan
        auc  = np.nan
    acc  = accuracy_score(y, pred)
    ppv  = precision_score(y, pred, zero_division=0)
    sens = recall_score(y, pred)  # sensitivity
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    f1   = f1_score(y, pred)

    # ----- bootstrap CIs -----
    rng = np.random.default_rng(0)
    auc_vals, acc_vals, ppv_vals, sens_vals, spec_vals, f1_vals = [], [], [], [], [], []
    f_grid = np.linspace(0, 1, 101)
    tprs_boot = []

    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb, sb = y[idx], scores[idx]
        pb = (sb >= 0.5).astype(int)

        if len(np.unique(yb)) > 1:
            try:
                auc_vals.append(roc_auc_score(yb, sb))
                fpr_b, tpr_b, _ = roc_curve(yb, sb)
                tprs_boot.append(np.interp(f_grid, fpr_b, tpr_b, left=0, right=1))
            except Exception:
                pass
        acc_vals.append(accuracy_score(yb, pb))
        ppv_vals.append(precision_score(yb, pb, zero_division=0))
        sens_vals.append(recall_score(yb, pb))
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yb, pb, labels=[0,1]).ravel()
        spec_vals.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) else np.nan)
        f1_vals.append(f1_score(yb, pb))

    def ci(arr):
        arr = np.asarray(arr, float)
        return (np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)) if arr.size else (np.nan, np.nan)

    auc_ci  = ci(np.array(auc_vals, float))
    acc_ci  = ci(np.array(acc_vals, float))
    ppv_ci  = ci(np.array(ppv_vals, float))
    sens_ci = ci(np.array(sens_vals, float))
    spec_ci = ci(np.array(spec_vals, float))
    f1_ci   = ci(np.array(f1_vals, float))

    # ----- style/layout -----
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15, "font.size": 11
    })
    fig = plt.figure(figsize=(12, 6.8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3.3, 0.9], hspace=0.35, wspace=0.3)

    # (1) Violin + box
    ax1 = fig.add_subplot(gs[0, 0])
    d0, d1 = scores[y==0], scores[y==1]
    parts = ax1.violinplot([d0, d1], positions=[0,1], showmeans=False, widths=0.8)
    for pc in parts['bodies']:
        pc.set_facecolor("#a8d8e8"); pc.set_alpha(0.5)
    for k in ('cbars','cmins','cmaxes'):
        parts[k].set_visible(False)
    box = ax1.boxplot([d0, d1], positions=[0,1], widths=0.35, patch_artist=True)
    for b in box['boxes']: b.set(facecolor="#f2cfd5", alpha=0.85)
    for med in box['medians']: med.set(color="#333", linewidth=1.4)
    ax1.axhline(0.5, ls="--", lw=1, color="#333")
    ax1.set_xticks([0,1]); ax1.set_xticklabels(["0 (HE)", "1 (GC)"])
    ax1.set_ylabel("Predicted Score"); ax1.set_title("Cut-off: 0.5")

    # (2) PDF per class
    ax2 = fig.add_subplot(gs[0, 1])
    xs = np.linspace(0, 1, 400)
    def kde_or_hist(ax, data, label):
        try:
            if len(data) > 1 and np.std(data) > 0:
                kde = gaussian_kde(data)
                ax.fill_between(xs, kde(xs), alpha=0.35, label=label); return
        except Exception:
            pass
        h, bins = np.histogram(data, bins=20, range=(0,1), density=True)
        ctr = 0.5*(bins[1:]+bins[:-1])
        ax.plot(ctr, h, label=label)
    kde_or_hist(ax2, d0, "HE (0)")
    kde_or_hist(ax2, d1, "GC (1)")
    ax2.axvline(0.5, ls="--", lw=1, color="#333")
    ax2.set_xlabel("Predicted Score"); ax2.set_ylabel("p.d.f.")
    ax2.set_title("Class score densities"); ax2.legend()

    # (3) ROC with bootstrap band
    ax3 = fig.add_subplot(gs[0, 2])
    if len(np.unique(y)) > 1:
        fpr_curve, tpr_curve, _ = roc_curve(y, scores)
        if len(tprs_boot):
            band = np.nanpercentile(np.vstack(tprs_boot), [2.5, 97.5], axis=0)
            ax3.fill_between(f_grid, band[0], band[1], alpha=0.2, label="95% CI")
        ax3.plot(fpr_curve, tpr_curve, lw=2,
                 label=f"ROC (AUC={auc:.2f}; 95% CI {auc_ci[0]:.2f}-{auc_ci[1]:.2f})")
    ax3.plot([0,1], [0,1], ls="--", c="#666")
    ax3.set_xlabel("1 - Specificity"); ax3.set_ylabel("Sensitivity")
    ax3.set_title(f"ROC ({title_tag})"); ax3.legend(loc="lower right")

    # ----- robust formatting helpers -----
    def _to_float(x): return float(np.asarray(x).squeeze()[()])
    def fmt_ci(point, ci_pair):
        lo, hi = ci_pair
        return f"{_to_float(point):.2f} ({_to_float(lo):.2f}, {_to_float(hi):.2f})"

    # (4) Metrics table
    axT = fig.add_subplot(gs[1, :]); axT.axis("off")
    cell_text = [[
        str(title_tag),
        f"{_to_float(mw_p):.2e}" if not np.isnan(mw_p) else "nan",
        f"{_to_float(r2):.2f}",
        fmt_ci(auc,  auc_ci)  if not np.isnan(auc)  else "nan",
        fmt_ci(acc,  acc_ci),
        fmt_ci(ppv,  ppv_ci),
        fmt_ci(sens, sens_ci),
        fmt_ci(spec, spec_ci),
        fmt_ci(f1,   f1_ci),
    ]]
    cols = ["Evaluate","MW-U Pvalue","R2","AUC","Accuracy","Precision","Sensitivity","Specificity","F1score"]
    tbl = axT.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.2)

    fig.suptitle(f"Score cut-off fixed to: {cutoff}", fontsize=14, y=0.98)
    plt.show()

# ----- TRAIN with cimcb_lite (n_comp fixed to 2) -----
modelPLS = cb.model.PLS_SIMPLS(n_components=2)
Ypred = modelPLS.train(XTknn, Ytrain)   # scores on TRAIN returned by cimcb_lite


# %% Plotting for the abvove training of the PLS_DA model
def pretty_pls_eval(scores, y, cutoff=0.5, title_tag="Train", n_boot=800,
                    figsize=(12, 9.5), table_fontsize=12, table_scale=(1.4, 2.0)):
    import numpy as np, matplotlib.pyplot as plt
    from matplotlib import gridspec
    from sklearn.metrics import (r2_score, roc_auc_score, roc_curve, accuracy_score,
                                 precision_score, recall_score, confusion_matrix, f1_score)
    from scipy.stats import mannwhitneyu, gaussian_kde

    scores = np.asarray(scores).ravel()
    y = np.asarray(y).astype(int)
    scores = np.nan_to_num(scores, nan=0.5)
    scores = np.clip(scores, 0.0, 1.0)

    # ---- point metrics ----
    pred = (scores >= cutoff).astype(int)
    r2   = r2_score(y, scores)
    if len(np.unique(y)) == 2:
        mw_p = mannwhitneyu(scores[y==0], scores[y==1], alternative="two-sided").pvalue
        auc  = roc_auc_score(y, scores)
    else:
        mw_p = np.nan; auc = np.nan
    acc  = accuracy_score(y, pred)
    ppv  = precision_score(y, pred, zero_division=0)
    sens = recall_score(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    f1   = f1_score(y, pred)

    # ---- bootstrap CIs (same as before) ----
    rng = np.random.default_rng(0)
    auc_vals, acc_vals, ppv_vals, sens_vals, spec_vals, f1_vals = [], [], [], [], [], []
    f_grid = np.linspace(0, 1, 101); tprs_boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        yb, sb = y[idx], scores[idx]; pb = (sb >= cutoff).astype(int)
        if len(np.unique(yb)) > 1:
            try:
                auc_vals.append(roc_auc_score(yb, sb))
                fpr_b, tpr_b, _ = roc_curve(yb, sb)
                tprs_boot.append(np.interp(f_grid, fpr_b, tpr_b, left=0, right=1))
            except Exception:
                pass
        acc_vals.append(accuracy_score(yb, pb))
        ppv_vals.append(precision_score(yb, pb, zero_division=0))
        sens_vals.append(recall_score(yb, pb))
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yb, pb, labels=[0,1]).ravel()
        spec_vals.append(tn_b / (tn_b + fp_b) if (tn_b + fp_b) else np.nan)
        f1_vals.append(f1_score(yb, pb))

    def ci(arr):
        arr = np.asarray(arr, float)
        return (np.nanpercentile(arr, 2.5), np.nanpercentile(arr, 97.5)) if arr.size else (np.nan, np.nan)
    auc_ci  = ci(np.array(auc_vals, float))
    acc_ci  = ci(np.array(acc_vals, float))
    ppv_ci  = ci(np.array(ppv_vals, float))
    sens_ci = ci(np.array(sens_vals, float))
    spec_ci = ci(np.array(spec_vals, float))
    f1_ci   = ci(np.array(f1_vals, float))

    # ---- layout / style (bigger fig + taller table row) ----
    plt.rcParams.update({
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.15, "font.size": 11
    })
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, height_ratios=[3.0, 2.2], hspace=0.35, wspace=0.3)

    # (1) Violin + box
    ax1 = fig.add_subplot(gs[0, 0])
    d0, d1 = scores[y==0], scores[y==1]
    parts = ax1.violinplot([d0, d1], positions=[0,1], showmeans=False, widths=0.8)
    for pc in parts['bodies']: pc.set_alpha(0.5)
    for k in ('cbars','cmins','cmaxes'): parts[k].set_visible(False)
    box = ax1.boxplot([d0, d1], positions=[0,1], widths=0.35, patch_artist=True)
    for b in box['boxes']: b.set_alpha(0.85)
    ax1.axhline(cutoff, ls="--", lw=1)
    ax1.set_xticks([0,1]); ax1.set_xticklabels(["0 (HE)", "1 (GC)"])
    ax1.set_ylabel("Predicted Score"); ax1.set_title(f"Cut-off: {cutoff}")

    # (2) PDF
    ax2 = fig.add_subplot(gs[0, 1])
    xs = np.linspace(0, 1, 400)
    def kde_or_hist(ax, data, label):
        try:
            if len(data) > 1 and np.std(data) > 0:
                kde = gaussian_kde(data); ax.fill_between(xs, kde(xs), alpha=0.35, label=label); return
        except Exception:
            pass
        h, bins = np.histogram(data, bins=20, range=(0,1), density=True)
        ctr = 0.5*(bins[1:]+bins[:-1]); ax.plot(ctr, h, label=label)
    kde_or_hist(ax2, d0, "HE (0)"); kde_or_hist(ax2, d1, "GC (1)")
    ax2.axvline(cutoff, ls="--", lw=1); ax2.set_xlabel("Predicted Score"); ax2.set_ylabel("p.d.f.")
    ax2.set_title("Class score densities"); ax2.legend()

    # (3) ROC
    ax3 = fig.add_subplot(gs[0, 2])
    if len(np.unique(y)) > 1:
        fpr_curve, tpr_curve, _ = roc_curve(y, scores)
        if len(tprs_boot):
            band = np.nanpercentile(np.vstack(tprs_boot), [2.5, 97.5], axis=0)
            ax3.fill_between(f_grid, band[0], band[1], alpha=0.2, label="95% CI")
        ax3.plot(fpr_curve, tpr_curve, lw=2,
                 label=f"ROC (AUC={auc:.2f}; 95% CI {auc_ci[0]:.2f}-{auc_ci[1]:.2f})")
    ax3.plot([0,1], [0,1], ls="--", c="#666")
    ax3.set_xlabel("1 - Specificity"); ax3.set_ylabel("Sensitivity"); ax3.set_title(f"ROC ({title_tag})")
    ax3.legend(loc="lower right")

    # ---- robust formatting helpers for table ----
    def _to_float(x):
        arr = np.asarray(x)
        return float(arr) if arr.shape == () else float(arr.ravel()[0])
    def fmt_ci(point, ci_pair):
        lo, hi = ci_pair
        return f"{_to_float(point):.2f} ({_to_float(lo):.2f}, {_to_float(hi):.2f})"

    # (4) Bigger metrics table
    axT = fig.add_subplot(gs[1, :]); axT.axis("off")
    cell_text = [[
        str(title_tag),
        f"{_to_float(mw_p):.2e}" if not np.isnan(mw_p) else "nan",
        f"{_to_float(r2):.2f}",
        fmt_ci(auc,  auc_ci)  if not np.isnan(auc)  else "nan",
        fmt_ci(acc,  acc_ci),
        fmt_ci(ppv,  ppv_ci),
        fmt_ci(sens, sens_ci),
        fmt_ci(spec, spec_ci),
        fmt_ci(f1,   f1_ci),
    ]]
    cols = ["Evaluate","MW-U Pvalue","R2","AUC","Accuracy","Precision","Sensitivity","Specificity","F1score"]
    tbl = axT.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(table_fontsize)              # bigger font
    tbl.scale(table_scale[0], table_scale[1])     # widen & TALLER cells

    fig.suptitle(f"Score cut-off fixed to: {cutoff}", fontsize=14, y=0.97)
    plt.show()

pretty_pls_eval(Ypred, Ytrain, cutoff=0.5, title_tag="Train")

