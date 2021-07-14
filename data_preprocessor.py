"""
Script to pre-process raw MOOCCube data to a format understandable by
scripts for exploration and modeling

@author: Abinash Sinha
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import defaultdict
from itertools import zip_longest


def get_num_items(x):
    video_ids_list = x.split(',')

    return len(video_ids_list)


def label_encode_items(mooccube_df, item):
    # label encoding to integers
    item_labeling_file = f'{item}_id_label_encodings.csv'
    if os.path.exists(item_labeling_file):
        df = pd.read_csv(item_labeling_file)
        label_encoding_dict = dict(zip(df.orig_video_id, df.new_video_id))

        def label_transform(x):
            orig_ids = x.split(',')
            new_ids = [str(label_encoding_dict[orig_id]) for orig_id in orig_ids]
            new_ids = ','.join(new_ids)
            return new_ids

    else:
        item_set = set()
        for _, row in mooccube_df.iterrows():
            items = row[f'{item}_ids'].split(',')
            item_set = item_set | set(items)

        print('Number of unique {}s watched: {}'.format(item, len(item_set)))

        item_list = list(item_set)
        le = preprocessing.LabelEncoder()
        label_encoded_item_list = le.fit_transform(item_list)

        label_encoded_item_list_df = pd.DataFrame()
        label_encoded_item_list_df[f'orig_{item}_id'] = item_list
        label_encoded_item_list_df[f'new_{item}_id'] = label_encoded_item_list
        label_encoded_item_list_df.to_csv(item_labeling_file, index=False)

        def label_transform(x):
            new_item_ids = le.transform(x.split(','))
            new_item_ids = [str(new_id) for new_id in new_item_ids]
            new_item_ids = ','.join(new_item_ids)
            return new_item_ids

    return label_transform


def prepare_user_items_sequences(d_dir, d_name):
    orig_data_path = os.path.join(d_dir, f'{d_name}_orig.csv')
    if not os.path.exists(orig_data_path):
        data_path = os.path.join(d_dir, f'{d_name}.json')
        chunks = pd.read_json(data_path, lines=True, chunksize=2000)
        header = True

        def get_csv_video_ids(x):
            student_video_ids_list = [activity['video_id'] for activity in x]
            student_video_ids_list = ','.join(student_video_ids_list)

            return student_video_ids_list

        def get_csv_course_ids(x):
            student_course_ids_list = [activity['course_id'] for activity in x]
            student_course_ids_list = ','.join(student_course_ids_list)

            return student_course_ids_list

        for c in chunks:
            activities_list = c['activity']
            students_video_ids_list = activities_list.apply(get_csv_video_ids)
            students_course_ids_list = activities_list.apply(get_csv_course_ids)
            num_video_ids = students_video_ids_list.apply(get_num_items)
            num_course_ids = students_course_ids_list.apply(get_num_items)
            mooccube_df = pd.DataFrame()
            mooccube_df['id'] = c['id']
            mooccube_df['video_ids'] = students_video_ids_list
            mooccube_df['num_video_ids'] = num_video_ids
            mooccube_df['course_ids'] = students_course_ids_list
            mooccube_df['num_course_ids'] = num_course_ids
            mooccube_df.to_csv(orig_data_path, header=header, index=False, mode='a')
            header = False

    mooccube_df = pd.read_csv(orig_data_path)

    label_transform_video_ids = label_encode_items(mooccube_df, 'video')
    label_transform_course_ids = label_encode_items(mooccube_df, 'course')

    mooccube_df['video_ids'] = mooccube_df['video_ids'].apply(label_transform_video_ids)
    mooccube_df['course_ids'] = mooccube_df['course_ids'].apply(label_transform_course_ids)
    data_path = os.path.join(d_dir, f'{d_name}_repeated.csv')
    mooccube_df.to_csv(data_path, index=False)


def soft_sample_test_data(d_dir, d_name, test_num=99, sample_type='random'):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.
    """
    np.random.seed(12345)

    data_file = os.path.join(d_dir, f'{d_name}.csv')
    test_file = os.path.join(d_dir, f'{d_name}_sample.csv')

    item_count = defaultdict(int)
    user_items = defaultdict()

    data_df = pd.read_csv(data_file)
    for _, row in data_df.iterrows():
        items = row['video_ids'].split(',')
        user_items[row['id']] = items
        for item in items:
            item_count[item] += 1

    all_item = list(item_count.keys())
    count = list(item_count.values())
    sum_value = np.sum([x for x in count])
    probability = [value / sum_value for value in count]

    user_neg_items = defaultdict()

    for user, user_seq in user_items.items():
        test_samples = []
        while len(test_samples) < test_num:
            if sample_type == 'random':
                sample_ids = np.random.choice(all_item, test_num, replace=False)
            else:  # sample_type == 'pop':
                sample_ids = np.random.choice(all_item, test_num, replace=False, p=probability)
            sample_ids = [str(item) for item in sample_ids if item not in user_seq and item not in test_samples]
            test_samples.extend(sample_ids)
        test_samples = test_samples[:test_num]
        test_samples = ','.join(test_samples)
        user_neg_items[user] = test_samples

    neg_items_df = pd.DataFrame(user_neg_items.items(), columns=['id', 'video_ids'])
    neg_items_df.to_csv(test_file, index=False)


def remove_consecutive_repititions(d_dir, d_name):
    data_path = os.path.join(d_dir, f'{d_name}_repeated.csv')
    mooccube_df = pd.read_csv(data_path)

    def remove_repititions(x):
        x = x.split(',')
        x = [i for i, j in zip_longest(x, x[1:]) if i != j]
        x = ','.join(x)

        return x

    mooccube_df['video_ids'] = mooccube_df['video_ids'].apply(remove_repititions)
    mooccube_df['num_video_ids'] = mooccube_df['video_ids'].apply(get_num_items)
    mooccube_df['course_ids'] = mooccube_df['course_ids'].apply(remove_repititions)
    mooccube_df['num_course_ids'] = mooccube_df['course_ids'].apply(get_num_items)

    data_path = os.path.join(d_dir, f'{d_name}.csv')
    mooccube_df.to_csv(data_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='data', type=str,
                        help='directory where data is(or will be) stored')
    parser.add_argument('--data_name', default='MOOCCube', type=str,
                        help='name of file storing data')

    args = parser.parse_args()
    prepare_user_items_sequences(args.data_dir, args.data_name)
    remove_consecutive_repititions(args.data_dir, args.data_name)
    soft_sample_test_data(args.data_dir, args.data_name)
    # hard_sample_test_data(args.data_dir, args.data_name)
