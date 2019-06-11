import pandas as pd
import os
import numpy as np


def process_relation(path, user_set):
    data_type={'user_id': np.object, 'fan_user_id': np.object}
    user_relation = pd.read_csv(path, delimiter='\t', dtype = data_type)
    total_num = user_relation.shape[0]
    user_relation = user_relation[(user_relation['user_id'].isin(user_set)) & (user_relation['fan_user_id'].isin(user_set))]
    filter_num = user_relation.shape[0]
    print("[User Relation] total:%d, filter:%d"%(total_num, filter_num))
    return user_relation

def process_sample(path):
    '''
    group user_sample by user_id and sum their weights, for latter merging with user 
    :param 
    path: user_sample file path 
    :return
    user_sample_processed: series with weight as column and user_id as index?
    '''
    data_type = {'user_id': np.object}
    user_sample = pd.read_csv(path, delimiter='\t', dtype=data_type)
    total_num = user_sample.shape[0]
    user_sample['weight'] = 1/user_sample['degree']
    user_sample_processed = user_sample[['user_id', 'weight']]\
        .groupby('user_id', as_index=True).mean()
    user_sample_processed.index.rename('uid', inplace=True)
    unique_num = user_sample_processed.shape[0]
    print('[User] total: %d, unique: %d'%(total_num, unique_num))
    print('Total weights: %.f'%user_sample_processed.weight.sum())
    return user_sample_processed

def join_individual(user_sample_path, user_path):
    '''
    join user_sample with user, get return sampled user with column weight added.
    :param user_path: 
    :param user_sample_path: 
    :return
    dataframe user with weight column added
    '''
    data_type = {'uid': np.object, 'short_id': np.object}
    user = pd.read_csv(user_path, delimiter='\t', dtype=data_type)
    user.set_index('uid', inplace=True)
    user_sample_processed = process_sample(user_sample_path)
    user_processed = user_sample_processed.join(user)
    user_processed = user_processed[~user_processed.following_count.isnull() & ~user_processed.follower_count.isnull()]
    print('[Valid User] %d' % user_processed.shape[0])
    return user_processed

def concate_all(path_list):
    all_user = pd.DataFrame()
    data_type = {'uid': np.object, 'short_id': np.object}
    for path in path_list:
        part_user = pd.read_csv(path, delimiter='\t', dtype=data_type)
        all_user = pd.concat([all_user, part_user], axis=0)
    groups = all_user.groupby('uid', as_index=True)
    weight = groups['weight'].mean()
    #weight = weight/weight.sum()
    avg_coloumns = groups[['dongtai_count', 'aweme_count', 'follower_count', \
                          'following_count', 'favoriting_count', \
                          'total_favorited', 'mplatform_followers_count']].mean().astype('int32')
    first_columns = groups[['short_id', 'unique_id', 'nickname', 'gender',\
                            'birthday', 'country', 'province', 'city', 'region', \
                            'location',	 'avatar_uri', 'cover_uri', 'signature', \
                            'has_activity_medal', 'is_gov_media_vip',\
                            'enterprise_verify_reason']].first()
    all_user = pd.concat([weight, first_columns, avg_coloumns], axis=1)
    return all_user

if __name__=='__main__':
    current_path = os.getcwd()
    #print(pd.read_csv(os.path.join(current_path, 'dataset\\douyin\\test\\user_sample.txt'),delimiter='\t'))
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width',100)
    user_dir = os.path.join(current_path, 'dataset', 'douyin', 'user')
    user_path_list = []
    user_relation_path_list = []
    for sub_dir in os.listdir(user_dir):
        print("Routine %s" % sub_dir)
        user_path  = os.path.join(user_dir, sub_dir, 'output_user.txt')
        user_sample_path = os.path.join(user_dir, sub_dir, 'output_user_sample.txt')
        user_detail = join_individual(user_sample_path, user_path)
        user_detail_path = os.path.join(user_dir, sub_dir, 'output_user_detail.txt')
        user_path_list.append(user_detail_path)
        user_detail.to_csv(user_detail_path, sep='\t')
        user_relation_path = os.path.join(user_dir, sub_dir, 'output_user_relation.txt')
        user_relation_path_list.append(user_relation_path)
        user_relation = process_relation(user_relation_path, set(user_detail.index.tolist()))
        user_relation.to_csv(user_relation_path, sep='\t')
    print("Merging all...")
    user_all = concate_all(user_path_list)
    user_relation_all = pd.DataFrame()
    for path in user_relation_path_list:
        part = pd.read_csv(path, delimiter='\t')
        user_relation_all = pd.concat([user_relation_all, part], axis=0)
    user_relation_all = user_relation_all.drop_duplicates()
    print("[All User] rows: %d, cols: %d" % user_all.shape)
    print("[All User Relation] %d" % user_relation_all.shape[0])
    print('Total weights: %.f' % user_all.weight.sum())
    user_all_path = os.path.join(user_dir, 'user_detail_all.txt')
    user_all.to_csv(user_all_path, sep='\t')
    user_relation_all_path = os.path.join(user_dir, 'user_relation_all.txt')
    user_relation_all.to_csv(user_relation_all_path, sep='\t')