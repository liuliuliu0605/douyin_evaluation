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
user_detail_file_name = 'user_detail_all.txt'

TYPE = '.eps'
styles = ['b-', 'y--', 'g:', 'r-.']
markers = ['o', '+', '^', '*']
color_list = ['peru', 'dodgerblue', 'brown', 'dimgray', 'springgreen',
              'fuchsia', 'greenyellow', 'hotpink', 'cyan', 'red',
              'grey', 'slateblue']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)

user_dir = os.path.join(os.getcwd(), 'dataset', 'douyin', 'user')
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
'''
data_type = {'uid': np.object, 'short_id': np.object}
user_detail_path = os.path.join(user_dir, 'user_detail_all.txt')
user_detail = pd.read_csv(user_detail_path, delimiter='\t', dtype=data_type)
user_detail = user_detail[~user_detail.following_count.isnull() & ~user_detail.follower_count.isnull()]
#user_detail['weight'] = 1
print(user_detail[user_detail.following_count.isnull()].uid)
#print(user_detail.describe())

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
fig, ax = plot_pdf('follower_count', label='All', x_label='# of followers')
#fig, ax = plot_pdf('follower_count', label='Government', x_label='# of followers',
#                   fig_ax=(fig,ax), condition=('is_gov_media_vip',1))
fig.savefig(os.path.join(draw_dir, 'pdf_follower'+TYPE))

## following pdf
print("*following pdf")
fig, ax = plot_pdf('following_count', label='All', x_label='# of followings')
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
fig, ax = plot_cdf('aweme_count', label=None, x_label='# of published videos')
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
'''