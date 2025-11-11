# churn_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------
# 1) CONFIG
# -------------------------
RANDOM_STATE = 42
N_SPLITS = 10
EPOCHS = 36
BATCH_SIZE = 64
VERBOSE = 1
CSV_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Fixed best parameters
rf_best_params_global = {
    'clf__max_depth': 20,
    'clf__max_features': 'log2',
    'clf__min_samples_leaf': 9,
    'clf__min_samples_split': 8,
    'clf__n_estimators': 400
}

svm_best_params_global = {
    'clf__C': np.float64(21.368329072358772),
    'clf__gamma': np.float64(0.0007068974950624604)
}

# -------------------------
# 2) LOAD AND CLEAN DATA
# -------------------------
df = pd.read_csv(CSV_PATH)
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

y = df['Churn'].astype(str).str.strip().str.lower().map({'no': 0, 'yes': 1}).astype(int)
X = df.drop(columns=['Churn']).copy()

print("Shapes:", X.shape, y.shape)
print("Positive rate (churn=1):", y.mean())

# -------------------------
# 3) METRICS
# -------------------------
def compute_confusion(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp, tn, fp, fn

def safe_div(a, b):
    return a / b if b != 0 else 0.0

def metrics_from_counts(tp, tn, fp, fn):
    P = tp + fn
    N = tn + fp
    tpr = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    acc = safe_div(tp + tn, P + N)
    bacc = (tpr + tnr) / 2.0
    prec = safe_div(tp, tp + fp)
    f1 = safe_div(2 * prec * tpr, (prec + tpr))
    tss = tpr - fpr
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = safe_div(2 * (tp * tn - fp * fn), denom)
    return {
        'Accuracy': acc, 'BalancedAcc': bacc,
        'Precision': prec, 'Recall': tpr, 'F1': f1,
        'TSS': tss, 'HSS': hss
    }

def brier_scores(y_true, y_prob):
    bs = brier_score_loss(y_true, y_prob)
    ybar = np.mean(y_true)
    bs_ref = np.mean((y_true - ybar) ** 2)
    bss = 1.0 - (bs / bs_ref)
    return bs, bss

# -------------------------
# 4) CNN MODEL
# -------------------------
def build_conv1d_model(n_features):
    inputs = keras.Input(shape=(n_features, 1))
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy')
    return model

# -------------------------
# 5) CROSS VALIDATION
# -------------------------
def make_preprocess(cat_cols, num_cols):
    return ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

results_rf, results_svm, results_cnn = [], [], []
roc_rf, roc_svm, roc_cnn = [], [], []
auc_rf, auc_svm, auc_cnn = [], [], []

for fold, (tr, te) in enumerate(skf.split(X, y), 1):
    print(f"\n=== Fold {fold}/{N_SPLITS} ===")
    Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
    ytr, yte = y.iloc[tr].values, y.iloc[te].values

    cat_cols = Xtr.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = [c for c in Xtr.columns if c not in cat_cols]
    preprocess = make_preprocess(cat_cols, num_cols)

    # Random Forest
    rf_fixed = Pipeline([
        ('prep', preprocess),
        ('clf', RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=-1,
            **{k.replace('clf__', ''): v for k, v in rf_best_params_global.items()}
        ))
    ])
    rf_fixed.fit(Xtr, ytr)
    rf_prob = rf_fixed.predict_proba(Xte)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    tp, tn, fp, fn = compute_confusion(yte, rf_pred)
    m = metrics_from_counts(tp, tn, fp, fn)
    bs, bss = brier_scores(yte, rf_prob)
    auc = roc_auc_score(yte, rf_prob)
    results_rf.append({'Fold': fold, **m, 'BS': bs, 'BSS': bss, 'AUC': auc})
    roc_rf.append(roc_curve(yte, rf_prob))
    auc_rf.append(auc)

    # SVM
    svm_fixed = Pipeline([
        ('prep', preprocess),
        ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced',
                    random_state=RANDOM_STATE,
                    **{k.replace('clf__', ''): v for k, v in svm_best_params_global.items()}))
    ])
    svm_fixed.fit(Xtr, ytr)
    svm_prob = svm_fixed.predict_proba(Xte)[:, 1]
    svm_pred = (svm_prob >= 0.5).astype(int)
    tp, tn, fp, fn = compute_confusion(yte, svm_pred)
    m = metrics_from_counts(tp, tn, fp, fn)
    bs, bss = brier_scores(yte, svm_prob)
    auc = roc_auc_score(yte, svm_prob)
    results_svm.append({'Fold': fold, **m, 'BS': bs, 'BSS': bss, 'AUC': auc})
    roc_svm.append(roc_curve(yte, svm_prob))
    auc_svm.append(auc)

    # Conv1D
    Xt_tr = preprocess.fit_transform(Xtr)
    Xt_te = preprocess.transform(Xte)
    n_features = Xt_tr.shape[1]
    cnn = build_conv1d_model(n_features)
    Xtr_cnn = Xt_tr.reshape((-1, n_features, 1))
    Xte_cnn = Xt_te.reshape((-1, n_features, 1))
    pos = ytr.mean()
    class_weight = {0: 0.5 / (1 - pos + 1e-12), 1: 0.5 / (pos + 1e-12)}
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    rlrop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
    cnn.fit(Xtr_cnn, ytr, epochs=EPOCHS, batch_size=BATCH_SIZE,
            verbose=VERBOSE, class_weight=class_weight,
            validation_split=0.1, callbacks=[es, rlrop])
    cnn_prob = cnn.predict(Xte_cnn, batch_size=BATCH_SIZE, verbose=0).ravel()
    cnn_pred = (cnn_prob >= 0.5).astype(int)
    tp, tn, fp, fn = compute_confusion(yte, cnn_pred)
    m = metrics_from_counts(tp, tn, fp, fn)
    bs, bss = brier_scores(yte, cnn_prob)
    auc = roc_auc_score(yte, cnn_prob)
    results_cnn.append({'Fold': fold, **m, 'BS': bs, 'BSS': bss, 'AUC': auc})
    roc_cnn.append(roc_curve(yte, cnn_prob))
    auc_cnn.append(auc)

# -------------------------
# 6) DISPLAY TABLES
# -------------------------
def build_table(results):
    df_ = pd.DataFrame(results)
    avg = df_.mean(numeric_only=True)
    avg['Fold'] = 'Average'
    return pd.concat([df_, pd.DataFrame([avg])], ignore_index=True)

tbl_rf = build_table(results_rf)
tbl_svm = build_table(results_svm)
tbl_cnn = build_table(results_cnn)

print("\n=== Random Forest (10-fold) ===")
print(tbl_rf.round(4))
print("\n=== SVM (10-fold) ===")
print(tbl_svm.round(4))
print("\n=== Conv1D (10-fold) ===")
print(tbl_cnn.round(4))

# -------------------------
# 7) MEAN ROC PLOTS
# -------------------------
def plot_mean_roc(roc_list, auc_list, title):
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    for fpr, tpr, _ in roc_list:
        tpr_i = np.interp(mean_fpr, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)

    plt.figure(figsize=(6, 5))
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"Mean ROC (AUC={auc_mean:.3f}±{auc_std:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

plot_mean_roc(roc_rf, auc_rf, "Random Forest — Mean ROC")
plot_mean_roc(roc_svm, auc_svm, "SVM (RBF) — Mean ROC")
plot_mean_roc(roc_cnn, auc_cnn, "Conv1D — Mean ROC")

# -------------------------
# 8) SUMMARY
# -------------------------
def summary_row(df, name):
    row = df[df['Fold'] == 'Average'].copy()
    row.insert(0, 'Model', name)
    return row

summary = pd.concat([
    summary_row(tbl_rf, 'RandomForest'),
    summary_row(tbl_svm, 'SVM_RBF'),
    summary_row(tbl_cnn, 'Conv1D')
], ignore_index=True)

cols = ['Model', 'Accuracy', 'BalancedAcc', 'Precision', 'Recall', 'F1', 'TSS', 'HSS', 'BS', 'BSS', 'AUC']
print("\n=== Summary (Averages Across Folds) ===")
print(summary[cols].round(4))
