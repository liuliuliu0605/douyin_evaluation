from evaluation import Evaluation
from matplotlib import pyplot as plt

import pandas as pd
import os
import math
import time
import powerlaw
import json
import numpy as np

# data sets to be used if True
user_sample_file_name = 'user_sample.txt'
user_detail_file_name = 'user.txt'

ARCHIVE = False  # use processed data last time if True
TYPE = '.png'
styles = ['b-', 'y--', 'g:', 'r-.']
markers = ['o', '+', '^', '*']

data_dir = os.path.join(os.getcwd(), 'dataset')
draw_dir = os.path.join(os.getcwd(), 'png')

if user_sample_file_name:
    user_sample_file_name = user_sample_file_name + ".processed" if ARCHIVE else user_sample_file_name
    user_sample_file_path = os.path.join(data_dir, user_sample_file_name)
if user_detail_file_name:
    user_detail_file_name = user_detail_file_name + ".processed" if ARCHIVE else user_detail_file_name
    user_detail_file_path = os.path.join(data_dir, user_detail_file_name)
log_file = open(os.path.join(data_dir, 'log'), 'w') if not ARCHIVE else open(os.path.join(data_dir, 'log'), 'r')

# read data
if user_sample_file_name:
    data_type = {'user_id': np.object}
    user_sample = pd.read_csv(user_sample_file_path, delimiter='\t', dtype=data_type)
if user_detail_file_name:
    data_type = {'uid': np.object, 'short_id': np.object}
    user_detail = pd.read_csv(user_detail_file_path, delimiter='\t', dtype=data_type)

# Show basic statistic
print("[Statistics]")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)

if user_detail_file_name and user_sample_file_name:
    if not ARCHIVE:
        total_num = user_sample.shape[0]
        user_sample = user_sample[user_sample.is_reset == 0]
        valid_num = user_sample.shape[0]
        unique_num = user_sample.user_id.nunique()
        user_sample.to_csv(user_sample_file_path + '.processed', index=False, sep='\t')
        #user_detail = user_detail[user_detail.uid.isin(user_sample.user_id.unique())]
        user_detail = pd.merge(user_sample[['user_id', 'weight']],#.drop_duplicates('user_id'),
                               user_detail, left_on='user_id', right_on='uid', how='left')
        #print(sum(user_detail.weight))
        gov_num = user_detail[user_detail.is_gov_media_vip == 1].user_id.unique().shape[0]
        user_detail.to_csv(user_detail_file_path + '.processed', index=False, sep='\t')
        log_file.write(json.dumps((total_num, valid_num, unique_num, gov_num)))
    else:
        total_num, valid_num, unique_num, gov_num = json.loads(log_file.readline())

    print("*User Sample:")
    print("Total: %d, Valid: %d, Unique: %d, Government: %d" %
          (total_num, valid_num, unique_num, gov_num))
    print(user_detail.describe())
    
# draw
evaluation = Evaluation()
if user_detail_file_name:
    # Plot follower count pdf
    followers = user_detail[user_detail.follower_count > 0][['follower_count', 'weight']]
    followers_gov = user_detail[(user_detail.is_gov_media_vip == 1) & (user_detail.follower_count > 0)][['follower_count', 'weight']]
    fig, ax = evaluation.draw_pdf(followers.follower_count, weights=followers.weight,
                                  fit_function='power_law', y_label="p(X)", x_label='# of followers',
                                  style=styles[1], marker=markers[1],legend_label='all', xmin=1)
    fig, ax = evaluation.draw_pdf(followers_gov.follower_count, weights=followers_gov.weight,
                                  fit_function='power_law', y_label="p(X)", x_label='# of followers',
                                  style=styles[2], marker=markers[2], legend_label='government', xmin=1,
                                  fig_ax=(fig, ax))
    fig.savefig(os.path.join(draw_dir, 'pdf', 'followers_pdf'+TYPE))


    # Plot follower count cdf
    followers = user_detail[['follower_count', 'weight']]
    followers_gov = user_detail[user_detail.is_gov_media_vip == 1][['follower_count', 'weight']]
    fig, ax = evaluation.draw_cdf(followers.follower_count, weights=followers.weight,
                                  y_label="CDF", x_label='# of followers', y_scale='linear',
                                  style=styles[1], marker=markers[1], legend_label='all')
    fig, ax = evaluation.draw_cdf(followers_gov.follower_count, weights=followers_gov.weight,
                                  y_label="CDF", x_label='# of followers', y_scale='linear',
                                  style=styles[2], marker=markers[2], legend_label='government',
                                  fig_ax= (fig, ax))
    fig.savefig(os.path.join(draw_dir, 'cdf', 'followers_cdf1' + TYPE))

    # Plot following count pdf
    followings = user_detail[user_detail.following_count > 0][['following_count', 'weight']]
    followings_gov = user_detail[(user_detail.is_gov_media_vip == 1) & (user_detail.following_count > 0)][['following_count', 'weight']]
    fig, ax = evaluation.draw_pdf(followings.following_count, weights=followings.weight,
                                  fit_function=None, y_label="p(X)", x_label='# of followings',
                                  style=styles[1], marker=markers[1], legend_label='all', xmin=1)
    fig, ax = evaluation.draw_pdf(followings_gov.following_count, weights=followings_gov.weight,
                                  fit_function=None, y_label="p(X)", x_label='# of followings',
                                  style=styles[2], marker=markers[2], legend_label='government', xmin=1,
                                  fig_ax=(fig, ax))
    fig.savefig(os.path.join(draw_dir, 'pdf', 'followings_pdf0' + TYPE))

    # Plot following count cdf
    followings = user_detail[['following_count', 'weight']]
    followings_gov = user_detail[user_detail.is_gov_media_vip == 1][['following_count', 'weight']]
    fig, ax = evaluation.draw_cdf(followings.following_count, weights=followings.weight,
                                  y_label="CDF", x_label='# of followings', y_scale='linear',
                                  style=styles[1], marker=markers[1], legend_label='all')
    fig, ax = evaluation.draw_cdf(followings_gov.following_count, weights=followings_gov.weight,
                                  y_label="CDF", x_label='# of followings', y_scale='linear',
                                  style=styles[2], marker=markers[2], legend_label='government',
                                  fig_ax=(fig, ax))
    fig.savefig(os.path.join(draw_dir, 'cdf', 'followings_cdf1' + TYPE))

    # Plot published video count pdf
    videos_published = user_detail[user_detail.aweme_count > 0][['aweme_count', 'weight']]
    videos_published_gov = user_detail[(user_detail.is_gov_media_vip == 1) & (user_detail.aweme_count > 0)][
        ['aweme_count', 'weight']]
    fig, ax = evaluation.draw_pdf(videos_published.aweme_count, weights=videos_published.weight,
                                  fit_function=None, y_label="p(X)", x_label='# of videos published',
                                  style=styles[1], marker=markers[1], legend_label='all', xmin=1)
    fig, ax = evaluation.draw_pdf(videos_published_gov.aweme_count, weights=videos_published_gov.weight,
                                  fit_function=None, y_label="p(X)", x_label='# of videos published',
                                  style=styles[2], marker=markers[2], legend_label='government', xmin=1,
                                  fig_ax=(fig, ax))
    fig.savefig(os.path.join(draw_dir, 'pdf', 'videos_published_pdf0' + TYPE))

    # Plot published video count cdf
    videos_published = user_detail[['aweme_count', 'weight']]
    videos_published_gov = user_detail[user_detail.is_gov_media_vip == 1][['aweme_count', 'weight']]
    fig, ax = evaluation.draw_cdf(videos_published.aweme_count, weights=videos_published.weight,
                                  y_label="CDF", x_label='# of videos published', y_scale='linear',
                                  style=styles[1], marker=markers[1], legend_label='all')
    fig, ax = evaluation.draw_cdf(videos_published_gov.aweme_count, weights=videos_published_gov.weight,
                                  y_label="CDF", x_label='# of videos published', y_scale='linear',
                                  style=styles[2], marker=markers[2], legend_label='government',
                                  fig_ax=(fig, ax))
    fig.savefig(os.path.join(draw_dir, 'cdf', 'videos_published_cdf1' + TYPE))

    # Plot liked video count pdf
    videos_liked = user_detail[user_detail.favoriting_count > 0][['favoriting_count', 'weight']]
    videos_liked_gov = user_detail[(user_detail.is_gov_media_vip == 1) & (user_detail.favoriting_count > 0)][
        ['favoriting_count', 'weight']]
    fig, ax = evaluation.draw_pdf(videos_liked.favoriting_count, weights=videos_liked.weight,
                                  fit_function=None, y_label="p(X)", x_label='# of videos liked',
                                  style=styles[1], marker=markers[1], legend_label='all', xmin=1)
    fig, ax = evaluation.draw_pdf(videos_liked_gov.favoriting_count, weights=videos_liked_gov.weight,
                                  fit_function=None, y_label="p(X)", x_label='# of videos liked',
                                  style=styles[2], marker=markers[2], legend_label='government', xmin=1,
                                  fig_ax=(fig, ax))
    fig.savefig(os.path.join(draw_dir, 'pdf', 'videos_liked_pdf0' + TYPE))

    # Plot liked video count cdf
    videos_liked = user_detail[['favoriting_count', 'weight']]
    videos_liked_gov = user_detail[user_detail.is_gov_media_vip == 1][['favoriting_count', 'weight']]
    fig, ax = evaluation.draw_cdf(videos_liked.favoriting_count, weights=videos_liked.weight,
                                  y_label="CDF", x_label='# of videos liked', y_scale='linear',
                                  style=styles[1], marker=markers[1], legend_label='all')
    fig, ax = evaluation.draw_cdf(videos_liked_gov.favoriting_count, weights=videos_liked_gov.weight,
                                  y_label="CDF", x_label='# of videos liked', y_scale='linear',
                                  style=styles[2], marker=markers[2], legend_label='government',
                                  fig_ax=(fig, ax))
    fig.savefig(os.path.join(draw_dir, 'cdf', 'videos_liked_cdf1' + TYPE))
    import powerlaw
    # Plot realation between followers and followings
    followers = user_detail.follower_count
    followings = user_detail.following_count
    fig, ax = evaluation.draw_realation(followers, followings,
                                        x_label="# of followers", y_label="# of followings",
                                        style=styles[1], marker=markers[1])
    fig.savefig(os.path.join(draw_dir, 'relation', 'followers_followings' + TYPE))













log_file.close()
