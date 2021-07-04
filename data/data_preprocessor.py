"""
Script to pre-process raw MOOCCube data to a format understandable by
scripts for exploration and modeling

@author: Abinash Sinha
"""

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import defaultdict


def prepare_user_items_sequences():
    data_name = 'MOOCCube'
    if not os.path.exists('MOOCCube_orig.csv'):
        chunks = pd.read_json('MOOCCube.json', lines=True, chunksize=2000)
        header = True

        def get_csv_video_ids(x):
            student_video_ids_list = [activity['video_id'] for activity in x]
            student_video_ids_list = ','.join(student_video_ids_list)

            return student_video_ids_list

        def get_num_videos(x):
            video_ids_list = x.split(',')

            return len(video_ids_list)

        for c in chunks:
            activities_list = c['activity']
            students_video_ids_list = activities_list.apply(get_csv_video_ids)
            num_video_ids = students_video_ids_list.apply(get_num_videos)
            mooccube_df = pd.DataFrame()
            mooccube_df['id'] = c['id']
            mooccube_df['video_ids'] = students_video_ids_list
            mooccube_df['num_video_ids'] = num_video_ids
            mooccube_df.to_csv('MOOCCube_orig.csv', header=header, index=False, mode='a')
            header = False

    mooccube_df = pd.read_csv('MOOCCube_orig.csv')

    # label encoding to integers
    item_labeling_file = 'video_id_label_encodings.csv'
    if os.path.exists(item_labeling_file):
        df = pd.read_csv(item_labeling_file)
        label_encoding_dict = dict(zip(df.orig_video_id, df.new_video_id))

        def label_transform(x):
            orig_video_ids = x.split(',')
            new_video_ids = [str(label_encoding_dict[orig_id]) for orig_id in orig_video_ids]
            new_video_ids = ','.join(new_video_ids)
            return new_video_ids

    else:
        item_set = set()
        for _, row in mooccube_df.iterrows():
            items = row['video_ids'].split(',')
            item_set = item_set | set(items)

        print('Number of unique videos watched: {}'.format(len(item_set)))

        item_list = list(item_set)
        le = preprocessing.LabelEncoder()
        label_encoded_item_list = le.fit_transform(item_list)

        label_encoded_item_list_df = pd.DataFrame()
        label_encoded_item_list_df['orig_video_id'] = item_list
        label_encoded_item_list_df['new_video_id'] = label_encoded_item_list
        label_encoded_item_list_df.to_csv(item_labeling_file, index=False)

        def label_transform(x):
            new_video_ids = le.transform(x.split(','))
            new_video_ids = [str(new_id) for new_id in new_video_ids]
            new_video_ids = ','.join(new_video_ids)
            return new_video_ids

    mooccube_df['video_ids'] = mooccube_df['video_ids'].apply(label_transform)
    mooccube_df.to_csv(data_name + '.csv', index=False)

    return data_name


def sample_test_data(data_name, test_num=99, sample_type='random'):
    """
    sample_type:
        random:  sample `test_num` negative items randomly.
        pop: sample `test_num` negative items according to item popularity.
    """
    np.random.seed(12345)

    data_file = f'{data_name}.csv'
    test_file = f'{data_name}_sample.csv'

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
        user_neg_items[user] = test_samples

    neg_items_df = pd.DataFrame(user_neg_items.items(), columns=['id', 'video_ids'])
    neg_items_df.to_csv(test_file, index=False)


if __name__ == '__main__':
    data_name = prepare_user_items_sequences()
    sample_test_data(data_name)
