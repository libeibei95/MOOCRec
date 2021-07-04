"""
Script to explore MOOCCube data before modeling

@author: Abinash Sinha
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='data', type=str)
parser.add_argument('--plot_dir', default='plot', type=str)
parser.add_argument('--data_name', default='MOOCCube', type=str)

args = parser.parse_args()
data_file = os.path.join(args.data_dir, args.data_name + '.csv')

mooccube_df = pd.read_csv(data_file)

series_num_videos = mooccube_df['num_video_ids']
sum_num_videos = series_num_videos.sum()
avg_num_videos = series_num_videos.mean()
median_num_videos = series_num_videos.median()
first_quartile = series_num_videos.quantile(q=0.25)
third_quartile = series_num_videos.quantile(q=0.75)
max_num_videos = series_num_videos.max()
min_num_videos = series_num_videos.min()
num_students = mooccube_df.shape[0]

item_set = set()
for _, row in mooccube_df.iterrows():
    items = row['video_ids'].split(',')
    item_set = item_set | set(items)

num_uniq_videos = len(item_set)

print('Total number of students: {}'.format(num_students))
print('Number of unique videos watched: {}'.format(num_uniq_videos))
print('Number of videos watched: {}'.format(sum_num_videos))
print('Average number of videos watched: {}'.format(round(avg_num_videos, 2)))
print('Median number of videos watched: {}'.format(median_num_videos))
print('1st quartile number of videos watched: {}'.format(first_quartile))
print('3rd quartile of videos watched: {}'.format(third_quartile))
print('Maximum number of videos watched: {}'.format(max_num_videos))
print('Minimum number of videos watched: {}'.format(min_num_videos))

# plot histogram for number of videos watched amongst all students
plt.figure(0)
plt.hist(series_num_videos, bins=128)
plt.savefig(os.path.join(args.plot_dir, 'num_videos_hist.png'))

# plot box plot for number of videos watched amongst all students
plt.figure(1)
plt.boxplot(series_num_videos)
plt.savefig(os.path.join(args.plot_dir, 'num_videos_boxplot.png'))

# plot ecdf for number of videos watched amongst all students
x = np.sort(series_num_videos)
y = np.arange(1, len(x) + 1) / len(x)
plt.figure(2)
plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('Number of Videos watched')
plt.ylabel('ECDF')
plt.margins(0.02)
plt.savefig(os.path.join(args.plot_dir, 'num_videos_ecdf.png'))

plt.show()
