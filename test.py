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
########
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

########
print("*user born year hist")
#fig, ax = plot_pdf_video('duration', x_label='Video duration(seconds)', fit_function=None, x_scale='linear', y_scale='linear', linear_bins=True)
fig, ax = plot_pdf('born_year', x_label="User's year of birth", fit_function=None)
ax.set_xlim(1900, 2020)
#ax.set_xticks(range(0, 65, 5))
fig.savefig(os.path.join(draw_dir, 'pdf_user_born_year'+TYPE))


print("user province pdf")
#location_freq = user_detail.groupby('province', as_index=False)['weight'].sum().sort_values(by='weight', ascending=False)
#location_freq['weight'] = location_freq.weight/location_freq.weight.sum()*100
#fig, ax = evaluation.draw_bar(location_freq['province'][:10], location_freq['weight'][:10]) 
#fig.savefig(os.path.join(draw_dir, 'bar_user_province'+TYPE))
