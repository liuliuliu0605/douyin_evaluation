#--coding=utf-8--

from evaluation import Evaluation, figsize
from matplotlib import pyplot as plt

import pandas as pd
import os
import math
import time
import json
import numpy as np


# data sets to be used if True
user_sample_file_name = 'user_sample.txt'
user_detail_file_name = 'user_detail_all.txt'

TYPE = '.png'
styles = ['b-', 'y--', 'g:', 'r-.']
markers = ['o', '+', '^', '*']
color_list = ['peru', 'dodgerblue', 'brown', 'dimgray', 'springgreen',
              'fuchsia', 'greenyellow', 'hotpink', 'cyan', 'red',
              'grey', 'slateblue']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)

user_dir = os.path.join(os.getcwd(), 'dataset', 'douyin', 'user')
video_dir = os.path.join(os.getcwd(), 'dataset', 'douyin', 'video')
draw_dir = os.path.join(os.getcwd(), TYPE[1:])
evaluation = Evaluation()

# convergence analysis

fig_ax = evaluation.draw_zscore(base_line=True)
for sub_dir, color in zip(["data_%d" % x for x in range(1,12)],color_list[:11]):
    data_type = {'user_id': np.object}
    user_sample_path = os.path.join(user_dir, sub_dir, 'output_user_sample.txt')
    user_sample = pd.read_csv(user_sample_path, delimiter='\t', dtype=data_type)
    evaluation.draw_zscore(user_sample.degree, user_sample.weight, fig_ax=fig_ax, color=color)
    #evaluation.draw_zscore(user_sample.degree, fig_ax=fig_ax, color=color)
fig_ax[0].savefig(os.path.join(draw_dir, 'geweke'+TYPE))


# user analysis

data_type = {'uid': np.object, 'short_id': np.object}
user_detail_path = os.path.join(user_dir, 'user_detail_all.txt')
user_detail = pd.read_csv(user_detail_path, delimiter='\t', dtype=data_type)
user_detail = user_detail[~user_detail.following_count.isnull() & ~user_detail.follower_count.isnull()]
#user_detail['weight'] = 1
print(user_detail[user_detail.following_count.isnull()].uid)
print(user_detail.describe())

# Plot pdf

def plot_pdf(col_name='follower_count',label=None, x_label='# of followers', y_label='p(X)',
             fig_ax=None, condition=None, style='y--', marker='+'):
    if condition:
        sequence = user_detail[user_detail[condition[0]] == condition[1]][user_detail[col_name] > 0][[col_name, 'weight']]
    else:
        sequence = user_detail[user_detail[col_name] > 0][[col_name, 'weight']]
    fig, ax = evaluation.draw_pdf(sequence[col_name], weights=sequence['weight'], fig_ax=fig_ax,
                                  fit_function='power_law', y_label=y_label, x_label=x_label,
                                  style=style, marker=marker, legend_label=label, xmin=1)
    return fig, ax

## follower pdf
print("*follower pdf")
fig, ax = plot_pdf('follower_count', label=None, x_label='# of followers')
#fig, ax = plot_pdf('follower_count', label='Government', x_label='# of followers',
#                   fig_ax=(fig,ax), condition=('is_gov_media_vip',1))
fig.savefig(os.path.join(draw_dir, 'pdf_follower'+TYPE))

## following pdf
print("*following pdf")
fig, ax = plot_pdf('following_count', label=None, x_label='# of followings')
fig.savefig(os.path.join(draw_dir, 'pdf_following'+TYPE))

# Plot cdf

def plot_cdf(col_name='follower_count',label=None, x_label='# of followers', y_label='CDF',
             fig_ax=None, condition=None, style='y--', marker='+'):
    if condition:
        sequence = user_detail[user_detail[condition[0]] == condition[1]][[col_name, 'weight']]
    else:
        sequence = user_detail[[col_name, 'weight']]
    fig, ax = evaluation.draw_cdf(sequence[col_name], weights=sequence['weight'],
                                  fig_ax=fig_ax, y_label=y_label, x_label=x_label, y_scale='linear',
                                  style=style, marker=marker, legend_label=label)
    return fig, ax

## follower cdf
print("*follower cdf")
fig, ax = plot_cdf('follower_count', label=None, x_label='# of followers')
fig.savefig(os.path.join(draw_dir, 'cdf_follower'+TYPE))

## following cdf
print("*following cdf")
fig, ax = plot_cdf('following_count', label=None, x_label='# of followings')
fig.savefig(os.path.join(draw_dir, 'cdf_following'+TYPE))

## aweme cdf
print("*aweme cdf")
fig, ax = plot_cdf('aweme_count', label=None, x_label='# of posted videos')
fig.savefig(os.path.join(draw_dir, 'cdf_aweme'+TYPE))

## favouriting cdf
print("*favoriting cdf")
fig, ax = plot_cdf('favoriting_count', label=None, x_label='# of liked videos')
fig.savefig(os.path.join(draw_dir, 'cdf_favoriting'+TYPE))


# relation between followers and followings
print("*followers followings relation")
followers = user_detail.follower_count
followings = user_detail.following_count
weight = user_detail.weight
fig, ax = evaluation.draw_relation(followers, followings, weight,
                                    x_label="# of followers", y_label="# of followings",
                                    marker='o')
fig.savefig(os.path.join(draw_dir, 'followers_followings' + TYPE))


print("user province pdf")
#location_freq = user_detail.groupby('province', as_index=False)['weight'].sum().sort_values(by='weight', ascending=False)
#location_freq['weight'] = location_freq.weight/location_freq.weight.sum()*100
#fig, ax = evaluation.draw_bar(location_freq['province'][:10], location_freq['weight'][:10]) 
#fig.savefig(os.path.join(draw_dir, 'bar_user_province'+TYPE))


# video analysis
data_type = {'uid': np.object}
video_detail_path = os.path.join(video_dir, 'video_detail_all.txt')
video_detail = pd.read_csv(video_detail_path, delimiter='\t', dtype=data_type)
total_num = video_detail.shape[0]
video_detail = video_detail[(video_detail.digg_count>=0) & (video_detail.share_count>=0) & (video_detail.comment_count>=0) & (video_detail.duration<301000) & (~video_detail.bit_rate1.isnull())]
video_detail['duration'] = video_detail.duration/1000
video_detail['bit_rate1'] = video_detail.bit_rate1/1000
video_detail['bit_rate2'] = video_detail.bit_rate2/1000
video_detail['bit_rate3'] = video_detail.bit_rate3/1000
valid_num = video_detail.shape[0]
video_detail['create_time'] =  pd.to_datetime(video_detail.create_time.values, unit='s',utc=True).tz_convert("Asia/Shanghai")
video_detail['date'] = video_detail.create_time.dt.date
base_date = pd.datetime(2019, 6, 9).date()
video_detail = video_detail[video_detail.date < base_date]
choose_num = video_detail.shape[0]
print("[Video] total: %d, valid: %d, choose: %d(before %s)"%(total_num, valid_num, choose_num, str(base_date)))
#video_detail['weight'] = 1
#print(user_detail[user_detail.following_count.isnull()].uid)
print(video_detail.columns)
print(video_detail.describe())

# Plot pdf
def plot_pdf_video(col_name='digg_count',label=None, x_label='# of likes', y_label='p(X)',
             fig_ax=None, condition=None, style='y--', marker='+', fit_function='power_law',
             x_scale='log', y_scale='log', linear_bins=False):
    if condition:
        sequence = video_detail[video_detail[condition[0]] == condition[1]][video_detail[col_name] > 0][[col_name, 'weight']]
    else:
        sequence = video_detail[video_detail[col_name] > 0][[col_name, 'weight']]
    fig, ax = evaluation.draw_pdf(sequence[col_name], weights=sequence['weight'], fig_ax=fig_ax,
                                  fit_function=fit_function, y_label=y_label, x_label=x_label,
                                  style=style, marker=marker, legend_label=label, xmin=1,
                                  x_scale=x_scale, y_scale=y_scale, linear_bins=linear_bins)
    return fig, ax

## digg_count pdf
print("*digg_count|comment_count|share_count pdf")
fig, ax = plot_pdf_video('digg_count', label='like',style=styles[0], marker=markers[0])
fig, ax = plot_pdf_video('comment_count', label='comment', style=styles[1], marker=markers[1], fig_ax = (fig, ax))
fig, ax = plot_pdf_video('share_count', x_label='Video popularity', label='share', style=styles[2], marker=markers[2], fig_ax = (fig, ax))

fig.savefig(os.path.join(draw_dir, 'pdf_video_pop'+TYPE))

# Plot cdf
def plot_cdf_video(col_name='digg_count',label=None, x_label='# of likes', y_label='CDF',
             fig_ax=None, condition=None, style='y--', marker='+', x_scale='log', y_scale='linear'):
    if condition:
        sequence = video_detail[video_detail[condition[0]] == condition[1]][[col_name, 'weight']]
    else:
        sequence = video_detail[[col_name, 'weight']]
        #print(sequence.iloc[:10,:])
    sequence = sequence[~sequence[col_name].isnull()]
    valid_size = sequence.shape[0]
    print("%s not null: %d"%(col_name, valid_size))
    fig, ax = evaluation.draw_cdf(sequence[col_name], weights=sequence['weight'],
                                  fig_ax=fig_ax, y_label=y_label, x_label=x_label, x_scale=x_scale,
                                  y_scale=y_scale, style=style, marker=marker, legend_label=label)
    return fig, ax

print("*digg_count|comment_count|share_count cdf")
fig, ax = plot_cdf_video('digg_count', label='like',style=styles[0], marker=markers[0])
fig, ax = plot_cdf_video('comment_count', label='comment', style=styles[1], marker=markers[1], fig_ax = (fig, ax))
fig, ax = plot_cdf_video('share_count', x_label='Video popularity', label='share', style=styles[2], marker=markers[2], fig_ax = (fig, ax))
fig.savefig(os.path.join(draw_dir, 'cdf_video_pop'+TYPE))

print("*video duration pdf")
fig, ax = plot_pdf_video('duration', x_label='Video duration(seconds)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
fig.savefig(os.path.join(draw_dir, 'pdf_video_duration'+TYPE))

print("*video duration cdf")
fig, ax = plot_cdf_video('duration', x_label='Video duration(seconds)', style=styles[0], marker=markers[0])
fig.savefig(os.path.join(draw_dir, 'cdf_video_duration'+TYPE))

print("*video bitrate cdf")
fig, ax = plot_cdf_video('bit_rate1', label='first bitrate',style=styles[0], marker=markers[0])
fig, ax = plot_cdf_video('bit_rate2', label='second bitrate', style=styles[1], marker=markers[1], fig_ax = (fig, ax))
fig, ax = plot_cdf_video('bit_rate3', x_label='Video birates(Kbps)', label='third bitrate', style=styles[2], marker=markers[2], fig_ax = (fig, ax))
fig.savefig(os.path.join(draw_dir, 'cdf_video_bitrate'+TYPE))

print("*video productivity pdf")
video_detail['date'] = video_detail.create_time.dt.date
video_detail['delta'] = base_date - video_detail.date
video_detail['last_day'] = video_detail.delta.dt.days
fig, ax = plot_pdf_video('last_day', x_label='Recently created(days)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
fig.savefig(os.path.join(draw_dir, 'pdf_video_create_time'+TYPE))

print("*video productivity cdf")
fig, ax = plot_cdf_video('last_day', x_label='Recently created(days)', style=styles[0], marker=markers[0], x_scale='linear')
fig.savefig(os.path.join(draw_dir, 'cdf_video_create_time'+TYPE))

print("*video resolution pie")
ratio_freq = video_detail.groupby('ratio', as_index=False)['weight'].sum()
print(ratio_freq)
fig, ax = evaluation.draw_pie(ratio_freq.ratio, ratio_freq.weight)
fig.savefig(os.path.join(draw_dir, 'pie_video_resolution'+TYPE))
