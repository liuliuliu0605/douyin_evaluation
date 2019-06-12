import pandas as pd
import os
import numpy as np
import time


def process_video(path):
    data_type={'aweme_id': np.object, 'create_time': np.int32, 'duration': np.int32,
               'rate': np.int8, 'group_id': np.object, 'is_ads': np.bool_,
               'comment_count': np.float, 'digg_count': np.float,
               'share_count': np.float, 'play_count': np.float,
               'forward_count': np.float, 'download_count': np.float,
               'ratio': np.object, 'is_h265': np.bool_,
               'width': np.int16, 'height': np.int16,
               'bit_rate1': np.float, 'bit_rate2': np.float, 'bit_rate3': np.float,
               'uid': np.object, 'mid': np.object, 'cid': np.object, 'pid': np.object,
               'is_private': np.bool_, 'with_goods': np.bool_, 'is_delete': np.bool_,
               'allow_share': np.bool_, 'allow_comment': np.bool_}
    video = pd.read_csv(path, delimiter='\t', low_memory=True, nrows=None, na_values='None')
    total_num = video.shape[0]
    video.columns = ['aweme_id','create_time','duration','rate','`desc`','is_ads','group_id','comment_count','digg_count','share_count','play_count','forward_count','download_count','cover','ratio','is_h265','share_url','uri','play_addr1','play_addr2','width','height','bit_rate1','bit_rate2','bit_rate3','uid','mid','cid','pid','is_private','with_goods','is_delete','allow_share','allow_comment']
    video.drop(video[video.duration.isnull() | video.aweme_id.isnull() | video.uid.isnull()].index,
               inplace=True)
    filter_num = video.shape[0]
    basic_columns = ['aweme_id', 'create_time', 'duration', 'rate', 'group_id', 'is_ads',
                     'comment_count', 'digg_count', 'share_count', 'play_count', 'forward_count',
                     'download_count','ratio', 'is_h265','width', 'height', 'bit_rate1',
                     'bit_rate2', 'bit_rate3','uid', 'mid', 'cid', 'pid','is_private', 'with_goods',
                     'is_delete', 'allow_share', 'allow_comment']
    extend_columns = ['aweme_id', '`desc`', 'cover', 'share_url', 'uri', 'play_addr1', 'play_addr2']
    basic_columns.extend(['duration', 'width', 'height', 'is_h265'])
    video_basic = video[basic_columns]
    video_extend = video[extend_columns]
    video_extend = video_extend.rename(columns={'`desc`': 'description'})
    print("[Video] total:%d, filter:%d" % (total_num, filter_num))
    return video_basic, video_extend

def join_individual(video_basic_path, user_detail_path):
    '''
    join video_basic with user_detail, get return videos with column weight added.
    :param video_basic_path:
    :param user_detail_path:
    :return
    dataframe video_detail with weight column added
    '''
    data_type = {'uid': np.object}
    video = pd.read_csv(video_basic_path, delimiter='\t', dtype=data_type)
    video.set_index('uid', inplace=True)
    user_detail = pd.read_csv(user_detail_path, delimiter='\t', dtype=data_type)
    user = user_detail[['uid', 'weight']]
    user.set_index('uid', inplace=True)
    video_processed = user.join(video, how='inner')
    print('[Valid Video] %d' % video_processed.shape[0])
    return video_processed

def concate_all(path_list):
    print("Merging All...")
    all_video = pd.DataFrame()
    data_type = {'aweme_id': np.object}
    for path in path_list:
        part_video = pd.read_csv(path, delimiter='\t', dtype=data_type)
        all_video = pd.concat([all_video, part_video], axis=0)
    groups = all_video.groupby(['aweme_id'], as_index=True)
    weight = groups['weight'].mean()
    #weight = weight/weight.sum()
    avg_coloumns = groups[['comment_count',	'digg_count', 'share_count', 'play_count',
                           'forward_count',	'download_count']].mean().astype('int32')
    first_columns = groups[['uid', 'create_time', 'rate', 'is_ads', 'group_id',
                            'bit_rate1', 'bit_rate2', 'bit_rate3', 'mid', 'cid', 'pid', 'is_private',
                            'with_goods', 'is_delete', 'allow_share', 'allow_comment', 'duration',
                            'width', 'height', 'is_h265']].first()
    all_video = pd.concat([weight, first_columns, avg_coloumns], axis=1)
    return all_video

if __name__=='__main__':
    start = time.time()
    current_path = os.getcwd()
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width',100)
    video_dir = os.path.join(current_path, 'dataset', 'douyin', 'video2')
    user_dir = os.path.join(current_path, 'dataset', 'douyin', 'user')
    video_basic_path_list = []
    user_detail_path_list = []
    video_detail_path_list = []

    for sub_dir in os.listdir(video_dir):
        print("Routine %s" % sub_dir)
        video_path = os.path.join(video_dir, sub_dir, 'video_post.txt')
        video_basic, video_extend = process_video(video_path)
        video_basic_path = os.path.join(video_dir, sub_dir, 'video_basic.txt')
        video_extend_path = os.path.join(video_dir, sub_dir, 'video_extend.txt')
        video_basic.to_csv(video_basic_path, sep='\t', index=False)
        video_extend.to_csv(video_extend_path, sep='\t', index=False)
        video_basic_path_list.append(video_basic_path)
        user_detail_path = os.path.join(user_dir, sub_dir, 'output_user_detail.txt')
        user_detail_path_list.append(user_detail_path)

    for video_path, user_path in zip(video_basic_path_list, user_detail_path_list):
        video_detail = join_individual(video_path, user_path)
        video_detail_path = os.path.join(video_dir, sub_dir, 'video_detail.txt')
        video_detail.to_csv(video_detail_path, sep='\t')
        video_detail_path_list.append(video_detail_path)

    video_detail_all = concate_all(video_detail_path_list)
    print("[All Video] rows: %d, cols: %d" % video_detail_all.shape)
    print('Total weights: %.f' % video_detail_all.weight.sum())
    video_all_path = os.path.join(video_dir, 'video_detail_all.txt')
    video_detail_all.to_csv(video_all_path, sep='\t')

    end = time.time()
    print("Time: %d seconds" % (end-start) )



