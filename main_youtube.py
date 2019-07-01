#--coding=utf-8--

from evaluation import Evaluation, figsize
from matplotlib import pyplot as plt
from datetime import timedelta
from statsmodels.stats.weightstats import DescrStatsW

import pandas as pd
import os
import math
import time
import json
import numpy as np
import seaborn as sns



# data sets to be used if True
user_sample_file_name = 'user_sample.txt'
user_detail_file_name = 'user_detail_all.txt'

TYPE = '.png'
styles = ['b-', 'y--', 'g:', 'r-.']
markers = ['o', '+', 'x', '*']
color_list = ['peru', 'dodgerblue', 'brown', 'dimgray', 'springgreen',
              'fuchsia', 'greenyellow', 'hotpink', 'cyan', 'red',
              'grey', 'slateblue']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)

user_dir = os.path.join(os.getcwd(), 'dataset', 'douyin', 'user')
video_dir = os.path.join(os.getcwd(), 'dataset', 'youtube', 'video', '080327')
video_dir2 = os.path.join(os.getcwd(), 'dataset', 'youtube', 'video', '0628')
draw_dir = os.path.join(os.getcwd(), TYPE[1:]+'_youtube')
evaluation = Evaluation()

# preprocess
'''
video_detail_path = os.path.join(video_dir, 'all.txt')
items = []
line_style = ("%s\t"*9)[:-1] + '\n' #"%s\t%s\t%d\t%s\t%d\t%d\t%f\t%d\t%d"
with open(video_detail_path) as f:
    lines = f.readlines()
    for line in lines:
        item = tuple([col.strip() for col in line.split('\t')[:9]])
        try:
            items.append(line_style % item)
        except TypeError as e:
            print("TypeError:", item)
video_detail_path = os.path.join(video_dir, 'all_processed.txt')
with open(video_detail_path, 'w') as f:
    f.writelines(items)
'''
# video analysis
data_type = {'uid': np.object}
video_detail_path = os.path.join(video_dir, 'all_processed.txt')
video_detail = pd.read_csv(video_detail_path, delimiter='\t', dtype=data_type, header=0,
                           names=['video ID', 'uploader', 'age', 'category', 'length',
                                  'views', 'rate', 'ratings', 'comments'])
video_added_path = os.path.join(video_dir2, 'rate.txt')
video_added = pd.read_csv(video_added_path, delimiter='\t', header=0,
                          names=['col1', 'col2', 'bitrate'])
video_added = video_added[(video_added.bitrate<=1000) & (video_added.bitrate>=1)]
video_detail = pd.concat([video_detail, video_added], axis=1)
total_num = video_detail.shape[0]
#video_detail = video_detail[(video_detail.digg_count>=0) & (video_detail.share_count>=0) & (video_detail.comment_count>=0) & (video_detail.duration<301000) & (~video_detail.bit_rate1.isnull())]
valid_num = video_detail.shape[0]
choose_num = video_detail.shape[0]
print("[Video] total: %d, valid: %d, choose: %d"%(total_num, valid_num, choose_num))
video_detail['weight'] = 1
#print(user_detail[user_detail.following_count.isnull()].uid)
print(video_detail.columns)
print(video_detail.describe())

# Plot pdf
def plot_pdf_video(col_name='digg_count',label=None, x_label='# of likes', y_label='p(X)',
             fig_ax=None, condition=None, style='y--', marker='+', fit_function='power_law',
             x_scale='log', y_scale='log', linear_bins=False, density=True, xmax=None):
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
                                  style=style, marker=marker, legend_label=label, xmin=1, xmax=xmax,
                                  x_scale=x_scale, y_scale=y_scale, density=density)
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

d = DescrStatsW(video_detail[['age', 'views']], weights=video_detail['weight'])
print(d.corrcoef)
d = DescrStatsW(video_detail[['age', 'ratings']], weights=video_detail['weight'])
print(d.corrcoef)
d = DescrStatsW(video_detail[['age', 'comments']], weights=video_detail['weight'])
print(d.corrcoef)


## digg_count pdf
print("*views|ratings|comments pdf")
fig, ax = plot_pdf_video('views', label='views',style=styles[0], marker=markers[0])
fig, ax = plot_pdf_video('ratings', label='ratings', style=styles[1], marker=markers[1], fig_ax = (fig, ax))
fig, ax = plot_pdf_video('comments', x_label='Video popularity', label='comments', style=styles[2], marker=markers[2], fig_ax=(fig, ax))

fig.savefig(os.path.join(draw_dir, 'y_pdf_video_pop'+TYPE))


print("*views|ratings|comments cdf")
fig, ax = plot_cdf_video('views', label='views',style=styles[0], marker=markers[0])
fig, ax = plot_cdf_video('ratings', label='ratings', style=styles[1], marker=markers[1], fig_ax = (fig, ax))
fig, ax = plot_cdf_video('comments', x_label='Video popularity', label='comments', style=styles[2], marker=markers[2], fig_ax = (fig, ax))
fig.savefig(os.path.join(draw_dir, 'y_cdf_video_pop'+TYPE))



print("*video duration hist")
fig, ax = plot_pdf_video('length', x_label='Video duration(seconds)', fit_function=None, xmax=700)
fig.savefig(os.path.join(draw_dir, 'y_pdf_video_duration'+TYPE))

print("*video duration cdf")
fig, ax = plot_cdf_video('length', x_label='Video duration(seconds)', style=styles[0], marker=markers[0])
fig.savefig(os.path.join(draw_dir, 'y_cdf_video_duration'+TYPE))

print("*video bitrate hist")
fig, ax = plot_pdf_video('bitrate', fit_function=None, x_label='Video bitrates(kbps)',xmax=500)
#ax.set_xlim(0, 500)
fig.savefig(os.path.join(draw_dir, 'y_pdf_video_bitrate'+TYPE))

print("*video bitrate cdf")
fig, ax = plot_cdf_video('bitrate', x_label='Video birates(kbps)',style=styles[0], marker=markers[0])
fig.savefig(os.path.join(draw_dir, 'y_cdf_video_bitrate'+TYPE))
exit(0)

print("*video productivity hist")
print(base_date, online_date)
#fig, ax = plot_pdf_video('last_day', x_label='Recently created(days)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
fig, ax = plot_pdf_video('days_last', x_label='Recently created(days)', fit_function=None)
#fig, ax = plot_pdf_video('days_after', x_label='Created date', fit_function=None)
#time_ranges = range(0, video_detail.days_after.max()+1, 100)
#plt.xticks(time_ranges,
#           [(online_date+timedelta(days=i)).strftime('%y/%m/%d') for i in time_ranges],
#           rotation=60)
fig.savefig(os.path.join(draw_dir, 'y_pdf_video_create_time'+TYPE))

print("*video productivity cdf")
fig, ax = plot_cdf_video('days_last', x_label='Recently created(days)')
fig.savefig(os.path.join(draw_dir, 'y_cdf_video_create_time'+TYPE))

print("*digg_count pdf different time")
fig, ax = plot_pdf_video('digg_count', x_label="# of likes", label='1-3 days old', style=styles[0], marker=markers[0],
                         condition=('days_last', 1, 3),  density=False)
fig, ax = plot_pdf_video('digg_count', x_label="# of likes", label='7-10 days old', style=styles[1], marker=markers[1],
                         condition=('days_last', 7, 10), fig_ax=(fig, ax), density=False)
fig, ax = plot_pdf_video('digg_count', x_label="# of likes", y_label='relative # of videos', label='101-103 days old', style=styles[2], marker=markers[2],
                         condition=('days_last', 101, 103), fig_ax=(fig, ax), density=False)
fig.savefig(os.path.join(draw_dir, 'y_pdf_video_likes_different'+TYPE))

print("*comment_count pdf different time")
fig, ax = plot_pdf_video('comment_count', x_label="# of comments", label='1-3 days old', style=styles[0], marker=markers[0],
                         condition=('days_last', 1, 3),  density=False)
fig, ax = plot_pdf_video('comment_count', x_label="# of comments", label='7-10 days old', style=styles[1], marker=markers[1],
                         condition=('days_last', 7, 10), fig_ax=(fig, ax), density=False)
fig, ax = plot_pdf_video('comment_count', x_label="# of comments", y_label='relative # of videos', label='101-103 days old', style=styles[2], marker=markers[2],
                         condition=('days_last', 101, 103), fig_ax=(fig, ax), density=False)
fig.savefig(os.path.join(draw_dir, 'y_pdf_video_comments_different'+TYPE))

print("*share_count pdf different time")
fig, ax = plot_pdf_video('share_count', x_label="# of shares", label='1-3 days old', style=styles[0], marker=markers[0],
                         condition=('days_last', 1, 3),  density=False)
fig, ax = plot_pdf_video('share_count', x_label="# of shares", label='7-10 days old', style=styles[1], marker=markers[1],
                         condition=('days_last', 7, 10), fig_ax=(fig, ax), density=False)
fig, ax = plot_pdf_video('share_count', x_label="# of shares", y_label='relative # of videos', label='101-103 days old', style=styles[2], marker=markers[2],
                         condition=('days_last', 101, 103), fig_ax=(fig, ax), density=False)
fig.savefig(os.path.join(draw_dir, 'y_pdf_video_shares_different'+TYPE))


print("*video resolution pie")
ratio_freq = video_detail.groupby('ratio', as_index=False)['weight'].sum()
print(ratio_freq)
fig, ax = evaluation.draw_pie(ratio_freq.ratio, ratio_freq.weight)
fig.savefig(os.path.join(draw_dir, 'pie_video_resolution'+TYPE))
