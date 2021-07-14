"""
Script to explore MOOCCube data before modeling

@author: Abinash Sinha
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_num_uniq_items(item):
    item_set = set()
    for _, row in mooccube_df.iterrows():
        items = row[f'{item}_ids'].split(',')
        item_set = item_set | set(items)
    num_uniq_items = len(item_set)

    return num_uniq_items


def calculate_and_plot_stats(item, fig_id):
    series_num_items = mooccube_df['num_{}_ids'.format(item)]
    sum_num_items = series_num_items.sum()
    avg_num_items = series_num_items.mean()
    median_num_items = series_num_items.median()
    first_quartile = series_num_items.quantile(q=0.25)
    third_quartile = series_num_items.quantile(q=0.75)
    max_num_videos = series_num_items.max()
    min_num_videos = series_num_items.min()
    print('Number of {}s watched: {}'.format(item, sum_num_items))
    print('Average number of {}s watched: {}'.format(item, round(avg_num_items, 2)))
    print('Median number of {}s watched: {}'.format(item, median_num_items))
    print('1st quartile number of {}s watched: {}'.format(item, first_quartile))
    print('3rd quartile of {}s watched: {}'.format(item, third_quartile))
    print('Maximum number of {}s watched: {}'.format(item, max_num_videos))
    print('Minimum number of {}s watched: {}'.format(item, min_num_videos))
    num_uniq_items = get_num_uniq_items(f'{item}')
    print('Number of unique {}s watched: {}'.format(item, num_uniq_items))
    print('\n')
    # plot histogram for number of videos watched amongst all students
    plt.figure(fig_id)
    plt.hist(series_num_items, bins=128)
    plt.savefig(os.path.join(args.plot_dir, f'num_{item}s_hist.png'))
    # plot box plot for number of videos watched amongst all students
    fig_id += 1
    plt.figure(fig_id)
    plt.boxplot(series_num_items)
    plt.savefig(os.path.join(args.plot_dir, f'num_{item}s_boxplot.png'))
    # plot ecdf for number of videos watched amongst all students
    x = np.sort(series_num_items)
    y = np.arange(1, len(x) + 1) / len(x)
    fig_id += 1
    plt.figure(fig_id)
    plt.plot(x, y, marker='.', linestyle='none')
    plt.xlabel(f'Number of {item}s watched')
    plt.ylabel('ECDF')
    plt.margins(0.02)
    plt.savefig(os.path.join(args.plot_dir, f'num_{item}s_ecdf.png'))

    fig_id += 1

    return fig_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--plot_dir', default='plot', type=str)
    parser.add_argument('--data_name', default='MOOCCube', type=str)

    args = parser.parse_args()
    data_file = os.path.join(args.data_dir, args.data_name + '.csv')

    mooccube_df = pd.read_csv(data_file)

    num_students = mooccube_df.shape[0]
    print('Total number of students: {}'.format(num_students))
    print('\n')

    fig_idx = 0
    fig_idx = calculate_and_plot_stats('course', fig_idx)
    calculate_and_plot_stats('video', fig_idx)


