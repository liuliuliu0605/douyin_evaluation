import json
import os
import powerlaw
import numpy

from evaluation import Evaluation
from matplotlib import pyplot as plt

base_dir = os.getcwd()
file_path_list = os.listdir(os.path.join(base_dir, 'dataset', 'feed'))
parent_dir = os.path.join(base_dir, 'dataset', 'feed')

digg_count = []
comment_count = []
share_count = []
styles = ['b-', 'y--', 'g:','r-.']
markers = ['o', '+', '^', '*']

videos = {}
for path in file_path_list:
    with open(os.path.join(parent_dir, path), 'rb') as f:
        data = json.load(f)
        aweme_list = data.get('aweme_list')
        for aweme in aweme_list:
            if aweme.get('aweme_id') not in videos:
                videos[aweme.get('aweme_id')] = 1
                digg_count.append(aweme.get('statistics').get('digg_count'))
                comment_count.append(aweme.get('statistics').get('comment_count'))
                share_count.append(aweme.get('statistics').get('share_count'))
print(comment_count)
#fit = powerlaw.Fit(digg_count)
powerlaw.plot_pdf(digg_count,  color='b')
plt.show()

'''
eval = Evaluation()
fig, ax = eval.draw_pdf(digg_count, y_label="Counts", style=styles[0], marker=markers[0],
                        legend_label='like')
fig, ax = eval.draw_pdf(comment_count, y_label="Counts", style=styles[1], marker=markers[1],
                        fig_ax=(fig,ax), legend_label='comment')
fig, ax = eval.draw_pdf(share_count, y_label="Counts", style=styles[2], marker=markers[2],
                        fig_ax=(fig,ax), legend_label='share')
fig.show()
'''
