import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
    return df.loc[filter]
def main(args):
    data_types = {args.ID: 'string', args.docking_score: 'float16'}
    truth = pd.read_csv(args.truth_csv, usecols=[args.ID, args.docking_score], dtype=data_types)
    prediction = pd.read_csv(args.prediction_csv, usecols=[args.ID, args.docking_score], dtype=data_types)

    truth = remove_outliers(truth, args.docking_score)
    prediction = remove_outliers(prediction, args.docking_score)

    merged = pd.merge(truth, prediction, on=args.ID, suffixes=('_truth', '_prediction'))

    top_percent = 0.01
    top_count = int(len(merged) * top_percent)
    merged_sorted = merged.sort_values(by=args.docking_score + '_prediction', ascending=True)
    top_merged = merged_sorted.head(top_count)

    x_top = top_merged[args.docking_score + '_truth']
    y_top = top_merged[args.docking_score + '_prediction']

    xmin, xmax = x_top.min(), x_top.max()
    ymin, ymax = y_top.min(), y_top.max()

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # R2
    r2 = stats.linregress(x_top, y_top).rvalue ** 2
    axs[0, 0].scatter(x_top, y_top, label=f'R2: {r2:.2f}')
    axs[0, 0].set_xlim(xmin, xmax)
    axs[0, 0].set_ylim(ymin, ymax)
    axs[0, 0].set_title('R2')
    axs[0, 0].set_xlabel('Ground Truth')
    axs[0, 0].set_ylabel('Prediction')
    axs[0, 0].legend()

    # Pearson
    pearson, _ = stats.pearsonr(x_top, y_top)
    axs[0, 1].scatter(x_top, y_top, label=f'Pearson: {pearson:.2f}')
    axs[0, 1].set_xlim(xmin, xmax)
    axs[0, 1].set_ylim(ymin, ymax)
    axs[0, 1].set_title('Pearson')
    axs[0, 1].set_xlabel('Ground Truth')
    axs[0, 1].set_ylabel('Prediction')
    axs[0, 1].legend()

    # Spearman
    spearman, _ = stats.spearmanr(x_top, y_top)
    axs[1, 0].scatter(x_top, y_top, label=f'Spearman: {spearman:.2f}')
    axs[1, 0].set_xlim(xmin, xmax)
    axs[1, 0].set_ylim(ymin, ymax)
    axs[1, 0].set_title('Spearman')
    axs[1, 0].set_xlabel('Ground Truth')
    axs[1, 0].set_ylabel('Prediction')
    axs[1, 0].legend()

    # Top 1% enrichment
    total_top_count = len(merged[merged[args.docking_score + '_truth'] <= merged[args.docking_score + '_truth'].quantile(top_percent)])
    selected_top_count = len(top_merged[top_merged[args.docking_score + '_truth'] <= top_merged[args.docking_score + '_truth'].quantile(top_percent)])

    enrichment_factor = (selected_top_count / top_count) / (total_top_count / len(merged))

    axs[1, 1].scatter(x_top, y_top, label=f'Top 1% Enrichment\nEF: {enrichment_factor:.2f}')
    axs[1, 1].set_xlim(xmin, xmax)
    axs[1, 1].set_ylim(ymin, ymax)
    axs[1, 1].set_title('Top 1% Enrichment')
    axs[1, 1].set_xlabel('Ground Truth')
    axs[1, 1].set_ylabel('Prediction')
    axs[1, 1].legend()
    
    plt.savefig(f'{args.output_prefix}_metrics.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("truth_csv", help="Path to the ground truth CSV file")
    parser.add_argument("prediction_csv", help="Path to the prediction CSV file")
    parser.add_argument("--ID", default="ID", help="Column name for the ID")
    parser.add_argument("--docking_score", default="docking_score", help="Column name for the docking score")
    parser.add_argument("--output_prefix", default="output", help="Prefix for the output figure")
    args = parser.parse_args()
    main(args)

