#!/usr/bin/env python3
"""Quick dataset checker for traffic_dataset.csv

Produces console summaries and saves a few diagnostic plots (PNG).
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main(path: str, out_dir: str):
    df = pd.read_csv(path)

    os.makedirs(out_dir, exist_ok=True)

    print('Loaded:', path)
    print('shape:', df.shape)

    print('\nPer-label counts:')
    print(df['label'].value_counts())

    key_cols = ['byte_count','upload_bytes','download_bytes','byte_ratio_up_down',
                'packet_count','flow_duration','burst_count','avg_burst_bytes','peak_rate_1s',
                'peak_download_1s','peak_upload_1s','pkt_size_mean']

    present = [c for c in key_cols if c in df.columns]
    print('\nColumns present (showing key subset):', present)

    print('\nMedian per-label:')
    print(df.groupby('label')[present].median().round(3))

    print('\n75th percentile per-label:')
    print(df.groupby('label')[present].quantile(0.75).round(3))

    print('\nMissing values per column:')
    print(df.isna().sum().sort_values(ascending=False).head(20))

    # Infinite check
    numeric = df.select_dtypes(include=[np.number])
    any_inf = np.isinf(numeric).any().any()
    print('\nAny infinite values?', any_inf)

    # Per-source_file counts
    if 'source_file' in df.columns:
        print('\nPer-source_file sample counts (top 10):')
        print(df['source_file'].value_counts().head(10))

    # Save simple histograms
    def save_hist(column, bins=50):
        if column not in df.columns:
            return
        plt.figure(figsize=(6,4))
        sub = df[column].dropna()
        # log-scale for heavy-tailed
        if sub.max() / max(sub.min() if sub.min() > 0 else 1, 1) > 100:
            plt.hist(np.log1p(sub), bins=bins)
            plt.xlabel('log1p(%s)' % column)
        else:
            plt.hist(sub, bins=bins)
            plt.xlabel(column)
        plt.title(column)
        out = os.path.join(out_dir, f'{column}_hist.png')
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        print('Saved', out)

    save_hist('byte_count')
    save_hist('download_bytes')
    save_hist('upload_bytes')
    save_hist('byte_ratio_up_down')
    save_hist('flow_duration')
    save_hist('burst_count')
    save_hist('peak_rate_1s')
    save_hist('peak_download_1s')

    # Save a small CSV of per-label medians for quick inclusion in reports
    medians = df.groupby('label')[present].median().round(3)
    medians.to_csv(os.path.join(out_dir, 'per_label_medians.csv'))
    print('Saved per_label_medians.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, default='traffic_dataset.csv')
    p.add_argument('--out', type=str, default='dataset_checks')
    args = p.parse_args()
    if not os.path.exists(args.csv):
        print('CSV not found:', args.csv)
        sys.exit(2)
    main(args.csv, args.out)
