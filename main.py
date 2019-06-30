#--coding=utf-8--

from evaluation import Evaluation, figsize
from matplotlib import pyplot as plt
from datetime import timedelta

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
user_detail['born_year'] = user_detail.birthday.str.split('-', expand=True)[0].astype("float")
print(user_detail[user_detail.following_count.isnull()].uid)
print(user_detail.describe())

# Plot pdf

def plot_pdf(col_name='follower_count',label=None, x_label='# of followers', y_label='p(X)',
             fig_ax=None, condition=None, style='y--', marker='+', fit_function='power_law'):
    if condition:
        if fit_function:
            sequence = user_detail[user_detail[condition[0]] == condition[1]][user_detail[col_name] > 0][[col_name, 'weight']]
        else:
            sequence = user_detail[user_detail[condition[0]] == condition[1]][[col_name, 'weight']]
    else:
        if fit_function:
            sequence = user_detail[user_detail[col_name] > 0][[col_name, 'weight']]
        else:
            sequence = user_detail[[col_name, 'weight']]
    sequence = sequence[~sequence[col_name].isnull()]
    fig, ax = evaluation.draw_pdf(sequence[col_name], weights=sequence['weight'], fig_ax=fig_ax,
                                  fit_function=fit_function, y_label=y_label, x_label=x_label,
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

## favouriting pdf
print("*favoriting pdf")
fig, ax = plot_pdf('favoriting_count', label=None, x_label='# of liked videos')
fig.savefig(os.path.join(draw_dir, 'pdf_favoriting'+TYPE))

print("*user born year hist")
#fig, ax = plot_pdf_video('duration', x_label='Video duration(seconds)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
fig, ax = plot_pdf('born_year', x_label="User's year of birth", fit_function=None)
ax.set_xlim(1900, 2020)
#ax.set_xticks(range(0, 65, 5))
fig.savefig(os.path.join(draw_dir, 'pdf_user_born_year'+TYPE))

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
online_date = video_detail['date'].min()
video_detail['delta_after'] = video_detail.date - online_date
video_detail['delta_last'] =  base_date - video_detail.date
video_detail['days_after'] = video_detail.delta_after.dt.days
video_detail['days_last'] = video_detail.delta_last.dt.days
choose_num = video_detail.shape[0]
print("[Video] total: %d, valid: %d, choose: %d(before %s)"%(total_num, valid_num, choose_num, str(base_date)))
#video_detail['weight'] = 1
#print(user_detail[user_detail.following_count.isnull()].uid)
print(video_detail.columns)
print(video_detail.describe())

# Plot pdf
def plot_pdf_video(col_name='digg_count',label=None, x_label='# of likes', y_label='p(X)',
             fig_ax=None, condition=None, style='y--', marker='+', fit_function='power_law',
             x_scale='log', y_scale='log', linear_bins=False, density=True):
    if condition:
        if fit_function:
            sequence = video_detail[(video_detail[condition[0]] >= condition[1]) & (video_detail[condition[0]] <= condition[2])][video_detail[col_name] > 0][[col_name, 'weight']]
        else:
            sequence = video_detail[video_detail[condition[0]] >= condition[1]][[col_name, 'weight']]
        print("!Special filter: %d" % sequence.shape[0])
    else:
        if fit_function:
            sequence = video_detail[video_detail[col_name] > 0][[col_name, 'weight']]
        else:
            sequence = video_detail[[col_name, 'weight']]
    sequence = sequence[~sequence[col_name].isnull()]
    fig, ax = evaluation.draw_pdf(sequence[col_name], weights=sequence['weight'], fig_ax=fig_ax,
                                  fit_function=fit_function, y_label=y_label, x_label=x_label,
                                  style=style, marker=marker, legend_label=label, xmin=1,
                                  x_scale=x_scale, y_scale=y_scale, linear_bins=linear_bins, density=density)
    return fig, ax

# Plot cdf
def plot_cdf_video(col_name='digg_count',label=None, x_label='# of likes', y_label='CDF',
             fig_ax=None, condition=None, style='y--', marker='+', x_scale='log', y_scale='linear'):
    if condition:
        sequence = video_detail[(video_detail[condition[0]] >= condition[1]) & (video_detail[condition[0]] <= condition[2])][[col_name, 'weight']]
        print("!Special filter: %d" % sequence.shape[0])
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


## digg_count pdf
print("*digg_count|comment_count|share_count pdf")
fig, ax = plot_pdf_video('digg_count', label='like',style=styles[0], marker=markers[0])
fig, ax = plot_pdf_video('comment_count', label='comment', style=styles[1], marker=markers[1], fig_ax = (fig, ax))
fig, ax = plot_pdf_video('share_count', x_label='Video popularity', label='share', style=styles[2], marker=markers[2], fig_ax = (fig, ax))

fig.savefig(os.path.join(draw_dir, 'pdf_video_pop'+TYPE))

print("*digg_count|comment_count|share_count cdf")
fig, ax = plot_cdf_video('digg_count', label='like',style=styles[0], marker=markers[0])
fig, ax = plot_cdf_video('comment_count', label='comment', style=styles[1], marker=markers[1], fig_ax = (fig, ax))
fig, ax = plot_cdf_video('share_count', x_label='Video popularity', label='share', style=styles[2], marker=markers[2], fig_ax = (fig, ax))
fig.savefig(os.path.join(draw_dir, 'cdf_video_pop'+TYPE))

print("*video duration hist")
#fig, ax = plot_pdf_video('duration', x_label='Video duration(seconds)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
fig, ax = plot_pdf_video('duration', x_label='Video duration(seconds)', fit_function=None)
ax.set_xlim(0, 62)
ax.set_xticks(range(0, 65, 5))
fig.savefig(os.path.join(draw_dir, 'pdf_video_duration'+TYPE))

print("*video(created after Apr. 23rd) duration hist")
#fig, ax = plot_pdf_video('duration', x_label='Video duration(seconds)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
new_date = pd.datetime(2019, 4, 23).date()
fig, ax = plot_pdf_video('duration', x_label='Video duration(seconds)', fit_function=None,
                         condition=('date', new_date, base_date))
ax.set_xlim(0, 62)
ax.set_xticks(range(0, 65, 5))
fig.savefig(os.path.join(draw_dir, 'pdf_video_duration_latest'+TYPE))

print("*video duration cdf")
fig, ax = plot_cdf_video('duration', x_label='Video duration(seconds)', style=styles[0], marker=markers[0])
fig.savefig(os.path.join(draw_dir, 'cdf_video_duration'+TYPE))

print("*video(created after Apr. 23rd) duration cdf")
fig, ax = plot_cdf_video('duration', x_label='Video duration(seconds)', style=styles[0], marker=markers[0],
                         condition=('date', new_date, base_date))
fig.savefig(os.path.join(draw_dir, 'cdf_video_duration_latest'+TYPE))

print("*video bitrate hist")
#fig, ax = plot_pdf_video('duration', x_label='Video duration(seconds)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
fig, ax = plot_pdf_video('bit_rate1', fit_function=None)
fig, ax = plot_pdf_video('bit_rate2', fit_function=None, fig_ax=(fig, ax))
fig, ax = plot_pdf_video('bit_rate3', fit_function=None, fig_ax=(fig, ax), x_label='Video bitrates(kbps)')
ax.set_xlim(0, 500)
fig.savefig(os.path.join(draw_dir, 'pdf_video_bitrate'+TYPE))

print("*video bitrate cdf")
fig, ax = plot_cdf_video('bit_rate1', label='first bitrate', style=styles[0], marker=markers[0])
fig, ax = plot_cdf_video('bit_rate2', label='second bitrate', style=styles[1], marker=markers[1], fig_ax = (fig, ax))
fig, ax = plot_cdf_video('bit_rate3', x_label='Video birates(kbps)', label='third bitrate', style=styles[2], marker=markers[2], fig_ax = (fig, ax))
fig.savefig(os.path.join(draw_dir, 'cdf_video_bitrate'+TYPE))

print("*video productivity hist")
print(base_date, online_date)
#fig, ax = plot_pdf_video('last_day', x_label='Recently created(days)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
fig, ax = plot_pdf_video('days_last', x_label='Recently created(days)', fit_function=None)
#fig, ax = plot_pdf_video('days_after', x_label='Created date', fit_function=None)
#time_ranges = range(0, video_detail.days_after.max()+1, 100)
#plt.xticks(time_ranges,
#           [(online_date+timedelta(days=i)).strftime('%y/%m/%d') for i in time_ranges],
#           rotation=60)
fig.savefig(os.path.join(draw_dir, 'pdf_video_create_time'+TYPE))

print("*video productivity cdf")
fig, ax = plot_cdf_video('days_last', x_label='Recently created(days)')
fig.savefig(os.path.join(draw_dir, 'cdf_video_create_time'+TYPE))

print("*digg_count pdf different time")
fig, ax = plot_pdf_video('digg_count', x_label="# of likes", label='1-3 days old', style=styles[0], marker=markers[0],
                         condition=('days_last', 1, 3),  density=False)
fig, ax = plot_pdf_video('digg_count', x_label="# of likes", label='7-10 days old', style=styles[1], marker=markers[1],
                         condition=('days_last', 7, 10), fig_ax=(fig, ax), density=False)
fig, ax = plot_pdf_video('digg_count', x_label="# of likes", y_label='relative # of videos', label='101-103 days old', style=styles[2], marker=markers[2],
                         condition=('days_last', 101, 103), fig_ax=(fig, ax), density=False)
fig.savefig(os.path.join(draw_dir, 'pdf_video_likes_different'+TYPE))

print("*comment_count pdf different time")
fig, ax = plot_pdf_video('comment_count', x_label="# of comments", label='1-3 days old', style=styles[0], marker=markers[0],
                         condition=('days_last', 1, 3),  density=False)
fig, ax = plot_pdf_video('comment_count', x_label="# of comments", label='7-10 days old', style=styles[1], marker=markers[1],
                         condition=('days_last', 7, 10), fig_ax=(fig, ax), density=False)
fig, ax = plot_pdf_video('comment_count', x_label="# of comments", y_label='relative # of videos', label='101-103 days old', style=styles[2], marker=markers[2],
                         condition=('days_last', 101, 103), fig_ax=(fig, ax), density=False)
fig.savefig(os.path.join(draw_dir, 'pdf_video_comments_different'+TYPE))

print("*share_count pdf different time")
fig, ax = plot_pdf_video('share_count', x_label="# of shares", label='1-3 days old', style=styles[0], marker=markers[0],
                         condition=('days_last', 1, 3),  density=False)
fig, ax = plot_pdf_video('share_count', x_label="# of shares", label='7-10 days old', style=styles[1], marker=markers[1],
                         condition=('days_last', 7, 10), fig_ax=(fig, ax), density=False)
fig, ax = plot_pdf_video('share_count', x_label="# of shares", y_label='relative # of videos', label='101-103 days old', style=styles[2], marker=markers[2],
                         condition=('days_last', 101, 103), fig_ax=(fig, ax), density=False)
fig.savefig(os.path.join(draw_dir, 'pdf_video_shares_different'+TYPE))


print("*video resolution pie")
ratio_freq = video_detail.groupby('ratio', as_index=False)['weight'].sum()
print(ratio_freq)
fig, ax = evaluation.draw_pie(ratio_freq.ratio, ratio_freq.weight)
fig.savefig(os.path.join(draw_dir, 'pie_video_resolution'+TYPE))
