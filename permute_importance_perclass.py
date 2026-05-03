#!/usr/bin/env python3
"""Compute per-class permutation importance across folds.

Trains a LightGBM per-fold with StratifiedGroupKFold and in-fold RandomOverSampler,
then computes permutation importance per-class (f1 for each label treated as positive).

Saves JSON + PNGs to baseline_results/presentation_selected/.
"""
import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from cv_baseline import DFPreprocessor, DEFAULT_FEATURES, HEAVY_TAIL_COLS
from imblearn.over_sampling import RandomOverSampler

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


OUTDIR = 'baseline_results/presentation_selected'
os.makedirs(OUTDIR, exist_ok=True)


def build_features(df):
    candidate = list(DEFAULT_FEATURES)
    extras = [
        'first4_up_mean','first4_up_std','first4_down_mean','first4_down_std',
        'pkt_bin_p0','pkt_bin_p1','pkt_bin_p2','pkt_bin_p3','pkt_size_entropy',
        'pkt_bin_up_p0','pkt_bin_up_p1','pkt_bin_up_p2','pkt_bin_up_p3',
        'pkt_bin_down_p0','pkt_bin_down_p1','pkt_bin_down_p2','pkt_bin_down_p3',
        'pkt_size_entropy_up','pkt_size_entropy_down',
        'iat_up_median','iat_up_q10','iat_up_q90','iat_up_cv','iat_up_skew',
        'iat_down_median','iat_down_q10','iat_down_q90','iat_down_cv','iat_down_skew',
        'burst_iai_cv'
    ]
    for e in extras:
        if e in df.columns and e not in candidate:
            candidate.append(e)
    # add any detected burst-like columns
    for c in df.columns:
        if ('burst' in c or '_t20' in c or '_t50' in c or '_t200' in c) and c not in candidate:
            candidate.append(c)
    feats = [c for c in candidate if c in df.columns]
    return feats


def main(csv='traffic_dataset.csv', n_splits=5, seed=42):
    if LGBMClassifier is None:
        raise RuntimeError('lightgbm not installed')
    df = pd.read_csv(csv)
    feats = build_features(df)
    print('Using %d features' % len(feats))

    df = df.dropna(subset=['label','source_file'])
    Xdf = df[feats]
    y = df['label'].astype(int).values
    groups = df['source_file'].values

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # accumulator: class_label -> list of importances arrays (per-fold means)
    acc = defaultdict(list)

    pre = DFPreprocessor(feature_names=feats, heavy_tail=HEAVY_TAIL_COLS)

    for fold, (train_idx, test_idx) in enumerate(skf.split(Xdf, y, groups=groups)):
        print('Fold', fold)
        Xtr_df = Xdf.iloc[train_idx]
        Xte_df = Xdf.iloc[test_idx]
        ytr = y[train_idx]
        yte = y[test_idx]

        Xtr_arr = pre.transform(Xtr_df)
        Xte_arr = pre.transform(Xte_df)

        # oversample in transformed space
        ros = RandomOverSampler(random_state=seed)
        X_res, y_res = ros.fit_resample(Xtr_arr, ytr)

        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=seed)
        clf.fit(X_res, y_res)

        # Fit a scaler on the resampled transformed training data so transform at predict time is valid
        scaler = StandardScaler()
        scaler.fit(X_res)

        # create a small wrapper so permutation_importance can call predict(X_df)
        class WrappedEstimator(BaseEstimator):
            def __init__(self, pre, scaler, clf):
                self.pre = pre
                self.scaler = scaler
                self.clf = clf

            def fit(self, X=None, y=None):
                # no-op fit to satisfy sklearn API
                return self

            def predict(self, X):
                # X is expected to be a DataFrame here
                Xt = self.pre.transform(X)
                Xt = self.scaler.transform(Xt)
                return self.clf.predict(Xt)

        wrapped = WrappedEstimator(pre, scaler, clf)

        # for each class compute permutation_importance with a scorer that returns f1 for that class
        classes = sorted(list(set(y)))
        for cls in classes:
            # custom scorer to compute binary f1 for pos_label=cls
            def class_f1(est, X_val, y_true, cls=cls):
                y_pred = est.predict(X_val)
                y_bin_true = (y_true == cls).astype(int)
                y_bin_pred = (y_pred == cls).astype(int)
                # avoid zero division
                from sklearn.metrics import f1_score as _f1
                return float(_f1(y_bin_true, y_bin_pred, zero_division=0))

            scorer = class_f1
            print(' computing perm importances for class', cls)
            res = permutation_importance(wrapped, Xte_df, yte, scoring=scorer, n_repeats=10, random_state=seed, n_jobs=4)
            acc[cls].append(res.importances_mean)

    # aggregate per-class across folds
    out = {'n_features': len(feats), 'features': feats, 'per_class': {}}
    for cls, arrs in acc.items():
        # arrs is list of arrays (n_folds x n_features)
        stacked = np.vstack(arrs)
        means = np.mean(stacked, axis=0)
        stds = np.std(stacked, axis=0)
        feat_list = []
        for i, name in enumerate(feats):
            feat_list.append({'feature': name, 'mean_importance': float(means[i]), 'std': float(stds[i])})
        feat_list = sorted(feat_list, key=lambda x: -x['mean_importance'])
        out['per_class'][str(cls)] = feat_list

    out_path = os.path.join(OUTDIR, 'perm_importance_perclass.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    # plot top 15 for each class
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for cls, feats_info in out['per_class'].items():
            top = feats_info[:15]
            names = [t['feature'] for t in top][::-1]
            means = [t['mean_importance'] for t in top][::-1]
            errs = [t['std'] for t in top][::-1]
            plt.figure(figsize=(8,4))
            y_pos = np.arange(len(names))
            plt.barh(y_pos, means, xerr=errs, color=['tab:blue' if v>=0 else 'tab:red' for v in means])
            plt.yticks(y_pos, names)
            plt.xlabel('Mean drop in class f1 when permuted')
            plt.title(f'Per-class permutation importance (class {cls})')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTDIR, f'perm_importance_class_{cls}_top15.png'))
            plt.close()
    except Exception as e:
        print('Could not plot per-class importances:', e)

    print('Saved per-class permutation importance to', out_path)


if __name__ == '__main__':
    main()
