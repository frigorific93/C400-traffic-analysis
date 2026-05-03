#!/usr/bin/env python3
"""Train a RandomForest baseline on the flow-level dataset.

Splits by `source_file` to avoid leakage. Applies log1p to heavy-tailed features.
Saves model, metrics, and feature importances to output directory.
"""
import argparse
import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


HEAVY_TAIL_COLS = [
    'byte_count','upload_bytes','download_bytes','peak_rate_1s','peak_download_1s','peak_upload_1s',
    'avg_burst_bytes','max_burst_bytes','pkt_size_mean'
]

DEFAULT_FEATURES = [
    'packet_count','byte_count','upload_bytes','download_bytes','upload_packets','download_packets',
    'byte_ratio_up_down','packet_ratio_up_down','direction_imbalance',
    'flow_duration','iat_mean','iat_std','burst_count','avg_burst_bytes','peak_rate_1s',
    'peak_download_1s','peak_upload_1s','pkt_size_mean','pkt_size_std','pkt_q50'
]


def train(df, features, outdir, n_estimators=200, random_state=42, clf_kwargs=None, balance_by_rows=False, max_rows_per_file=0):
    os.makedirs(outdir, exist_ok=True)

    # drop rows with missing features
    df = df.dropna(subset=features + ['label','source_file'])

    # Optionally balance splits by flow counts (rows) per source_file within each label
    np.random.seed(random_state)

    train_files = []
    val_files = []
    test_files = []

    # helper: get files for a label and counts
    for lab, g in df.groupby('label'):
        lab = int(lab)
        files = list(g['source_file'].unique())
        # compute rows per file (counts in the full df)
        file_counts = {f: int((df['source_file'] == f).sum()) for f in files}

        if balance_by_rows:
            # target rows per split
            total_rows = sum(file_counts.values())
            target_train = 0.6 * total_rows
            target_val = 0.2 * total_rows
            target_test = total_rows - target_train - target_val

            # greedy assignment: sort files largest->smallest and assign to the split
            # that is currently farthest from its target (by remaining need)
            sorted_files = sorted(file_counts.items(), key=lambda x: -x[1])
            cur = {'train': 0, 'val': 0, 'test': 0}
            for f, cnt in sorted_files:
                need = {
                    'train': target_train - cur['train'],
                    'val': target_val - cur['val'],
                    'test': target_test - cur['test']
                }
                # pick split with max positive need; if all negative, pick smallest current
                best = max(need.items(), key=lambda x: (x[1], -cur[x[0]]))[0]
                if best == 'train':
                    train_files.append(f)
                    cur['train'] += cnt
                elif best == 'val':
                    val_files.append(f)
                    cur['val'] += cnt
                else:
                    test_files.append(f)
                    cur['test'] += cnt
    else:
            # fallback: simple per-label file slicing to ensure presence in each split
            np.random.shuffle(files)
            n = len(files)
            if n >= 3:
                n_train = max(1, int(0.6 * n))
                n_val = max(1, int(0.2 * n))
            elif n == 2:
                n_train = 1
                n_val = 1
            else:
                n_train = 1
                n_val = 0
            train_files.extend(files[:n_train])
            val_files.extend(files[n_train:n_train+n_val])
            test_files.extend(files[n_train+n_val:])

    # deduplicate file lists
    train_files = list(dict.fromkeys(train_files))
    val_files = list(dict.fromkeys(val_files))
    test_files = list(dict.fromkeys(test_files))

    def build_split_df(files_list):
        subset = df[df['source_file'].isin(files_list)].copy()
        if max_rows_per_file and max_rows_per_file > 0:
            parts = []
            for sf, g in subset.groupby('source_file'):
                if len(g) > max_rows_per_file:
                    parts.append(g.sample(n=max_rows_per_file, random_state=random_state))
                else:
                    parts.append(g)
            subset = pd.concat(parts, axis=0)
        return subset

    train_df = build_split_df(train_files)
    val_df = build_split_df(val_files)
    test_df = build_split_df(test_files)

    # deduplicate file lists
    train_files = list(dict.fromkeys(train_files))
    val_files = list(dict.fromkeys(val_files))
    test_files = list(dict.fromkeys(test_files))

    print(f'train files: {len(train_files)}, val files: {len(val_files)}, test files: {len(test_files)}')
    print('train/val/test sizes:', len(train_df), len(val_df), len(test_df))

    # Features: log1p for heavy-tailed
    X_train = train_df[features].copy()
    X_val = val_df[features].copy()
    X_test = test_df[features].copy()

    for c in HEAVY_TAIL_COLS:
        if c in X_train.columns:
            X_train[c] = np.log1p(X_train[c])
            X_val[c] = np.log1p(X_val[c])
            X_test[c] = np.log1p(X_test[c])

    # scale
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=features, index=X_val.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

    y_train = train_df['label'].astype(int)
    y_val = val_df['label'].astype(int)
    y_test = test_df['label'].astype(int)

    # Create classifier using provided kwargs so CLI flags can control behaviour.
    if clf_kwargs is None:
        clf_kwargs = {'n_estimators': n_estimators, 'random_state': random_state, 'n_jobs': -1}
    else:
        # ensure some defaults
        clf_kwargs = dict(clf_kwargs)
        clf_kwargs.setdefault('n_estimators', n_estimators)
        clf_kwargs.setdefault('random_state', random_state)
        clf_kwargs.setdefault('n_jobs', -1)

    clf = RandomForestClassifier(**clf_kwargs)
    clf.fit(X_train, y_train)

    def eval_and_report(X, y, name):
        preds = clf.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average='macro')
        print(f'[{name}] acc={acc:.4f} macro-F1={f1:.4f}')
        print(classification_report(y, preds))
        cm = confusion_matrix(y, preds)
        return {'acc': acc, 'f1_macro': f1, 'cm': cm.tolist()}

    results = {}
    results['train'] = eval_and_report(X_train, y_train, 'train')
    results['val'] = eval_and_report(X_val, y_val, 'val')
    results['test'] = eval_and_report(X_test, y_test, 'test')

    # feature importances
    importances = dict(zip(features, clf.feature_importances_.tolist()))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print('\nTop features:')
    for k,v in sorted_importances[:20]:
        print(f'  {k}: {v:.4f}')

    # save model, scaler, results
    joblib.dump(clf, os.path.join(outdir, 'rf_baseline.joblib'))
    joblib.dump(scaler, os.path.join(outdir, 'scaler.joblib'))
    with open(os.path.join(outdir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(outdir, 'feature_importances.json'), 'w') as f:
        json.dump(sorted_importances, f, indent=2)

    # confusion matrix plot for test
    cm = np.array(results['test']['cm'])
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix (test)')
    plt.colorbar()
    ticks = np.arange(len(np.unique(df['label'])))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel('pred')
    plt.ylabel('true')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'confusion_test.png'))
    print('Saved model and results to', outdir)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='traffic_dataset.csv')
    p.add_argument('--out', default='baseline_results')
    p.add_argument('--n-est', type=int, default=200)
    p.add_argument('--max-depth', type=int, default=None, help='max depth for RandomForest (to reduce overfitting)')
    p.add_argument('--min-samples-leaf', type=int, default=1, help='min samples per leaf for RandomForest')
    p.add_argument('--class-weight', type=str, default=None, help="class_weight for RandomForest: 'balanced' or None")
    p.add_argument('--features', type=str, default=None,
                   help='comma-separated list of features to use; if omitted, a default set is used')
    p.add_argument('--balance-by-rows', action='store_true', help='balance train/val/test by flow counts (rows) per label when choosing source_file assignments')
    p.add_argument('--max-rows-per-file', type=int, default=0, help='if >0, sample up to this many rows per source_file when building train/val/test splits')
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print('CSV not found:', args.csv); exit(1)

    df = pd.read_csv(args.csv)

    if args.features:
        feats = args.features.split(',')
    else:
        # prefer columns that exist
        feats = [c for c in DEFAULT_FEATURES if c in df.columns]

    print('Using features:', feats)
    # parse class weight
    cw = None
    if args.class_weight is not None:
        if args.class_weight.lower() == 'balanced':
            cw = 'balanced'

    # create classifier with chosen hyperparameters
    clf_kwargs = {
        'n_estimators': args.n_est,
        'random_state': 42,
        'n_jobs': -1,
    }
    if args.max_depth is not None:
        clf_kwargs['max_depth'] = args.max_depth
    if args.min_samples_leaf is not None:
        clf_kwargs['min_samples_leaf'] = args.min_samples_leaf
    if cw is not None:
        clf_kwargs['class_weight'] = cw

    # pass classifier params via partial-like pattern: create classifier here
    from functools import partial
    def train_with_clf(df_, features_, outdir_, n_estimators=200, random_state=42):
        clf = RandomForestClassifier(**clf_kwargs)
        # reuse the train() internals by monkeypatching a tiny wrapper - call same logic
        # We'll copy/paste the body by calling train() and then replacing classifier creation.
        # For simplicity, call train() which will create its own RF; so instead set global
        # behavior by temporarily monkeypatching RandomForestClassifier. Simpler: call
        # train() and then overwrite the saved model. To keep code straightforward, call
        # the internal train() but with replaced instantiation above; here we will just
        # call the original train() and ignore double instantiation cost.
        train(df_, features_, outdir_, n_estimators=n_estimators, random_state=random_state)

    # Optionally pass balancing options to train
    extra = {}
    if args.balance_by_rows:
        extra['balance_by_rows'] = True
    if args.max_rows_per_file and args.max_rows_per_file > 0:
        extra['max_rows_per_file'] = int(args.max_rows_per_file)

    train(df, feats, args.out, n_estimators=args.n_est, clf_kwargs=clf_kwargs,
        balance_by_rows=extra.get('balance_by_rows', False), max_rows_per_file=extra.get('max_rows_per_file', 0))
