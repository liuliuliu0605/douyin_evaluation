from evaluation import Evaluation
from matplotlib import pyplot as plt

import pandas as pd
import os
import math
import time
import powerlaw

TYPE = '.png'
SAVE = True
base_dir = os.getcwd()
attribute_file_path = os.path.join(base_dir, 'dataset', 'attributes.txt')
network_file_path = os.path.join(base_dir, 'dataset', 'network.txt')
own_video_file_path = os.path.join(base_dir, 'dataset', 'user_sample_own_video.txt')
like_video_file_path = os.path.join(base_dir, 'dataset', 'user_sample_like_video.txt')

attributes = pd.read_csv(attribute_file_path)
networks = pd.read_csv(network_file_path)
own_videos = pd.read_csv(own_video_file_path)
like_videos = pd.read_csv(like_video_file_path)
styles = ['b-', 'y--', 'g:','r-.']
markers = ['o', '+', '^', '*']
eval = Evaluation()


# Show basic statistic
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 120)
reset_num = attributes[attributes.is_reset == 1].shape[0]
valid_num = attributes.shape[0]-reset_num
attributes = attributes[attributes.is_reset == 0]
attributes_unique = attributes.groupby(['user_id']).head(1)
own_videos = own_videos[own_videos.duration.isnull().values == False]
own_videos['create_time'] = own_videos['create_time'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
like_videos = like_videos[like_videos.duration.isnull().values == False]
like_videos['create_time'] = like_videos['create_time'].apply(
    lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x)))
if SAVE:
    attributes.to_csv(attribute_file_path + '.processed')
    networks.to_csv(network_file_path + '.processed')
    own_videos.to_csv(own_video_file_path + '.processed')
    like_videos.to_csv(like_video_file_path + '.processed')

print("[Statistics]")
print("Total: %d samples, %d reset samples, %d owned videos and %d liked videos" %
      (reset_num + valid_num, reset_num, own_videos.shape[0], like_videos.shape[0]))
print("Valid: %d users and %d unique users" %
      (valid_num, attributes.user_id.nunique()))
#print("All user:")
#print(attributes.describe())
#print("Unique user:")
#print(attributes_unique.describe())
print("Own videos:")
print(own_videos.describe())
print("Like videos:")
print(like_videos.describe())

#print("\n[Data samples]")
#print("User:")
#print(attributes.head())
#print("Video:")
#print(own_videos.head())

'''
# Plot follower count pdf
followers = attributes[attributes.follower_count>0].follower_count.tolist()
followers_unique = attributes[attributes.follower_count>0].groupby(['user_id']).follower_count.head(1).tolist()
fig, ax = eval.draw_pdf(followers_unique, y_label="# of user's followers",
                        style=styles[0], marker=markers[0], legend_label='unique')
fig, ax = eval.draw_pdf(followers, fig_ax=(fig, ax), y_label="p(X)", x_label='# of followers',
                        style=styles[1], marker=markers[1], legend_label='all')
fig.savefig('./eps/pdf/followers_pdf' + TYPE)

# Plot follower count cdf
print("[User follower cdf]")
followers = attributes.follower_count.tolist()
followers_unique = attributes.groupby(['user_id']).follower_count.head(1).tolist()
fig, ax = eval.draw_cdf(followers_unique, x_label="# of user's followers",
                        style=styles[0], marker=markers[0], legend_label='unique')
fig, ax = eval.draw_cdf(followers, fig_ax=(fig, ax), x_label="# of user's followers",
                        style=styles[1], marker=markers[1], legend_label='all')
fig.savefig('./eps/cdf/followers_cdf' + TYPE)
'''

'''
# Plot following count pdf
followings = attributes.following_count.tolist()
followings_unique = attributes.groupby(['user_id']).following_count.head(1).tolist()
fig, ax = eval.draw_pdf(followings_unique, y_label="# of user's followings",
                        style=styles[0], marker=markers[0], legend_label='unique')
fig, ax = eval.draw_pdf(followings, fig_ax=(fig, ax), y_label="# of user's followings",
                        style=styles[1], marker=markers[1], legend_label='all')
fig.savefig('./eps/pdf/followings_pdf' + TYPE)

# Plot following count cdf
print("[User following cdf]")
fig, ax = eval.draw_cdf(followings_unique, x_label="# of user's followings",
                        style=styles[0], marker=markers[0], legend_label='unique')
fig, ax = eval.draw_cdf(followings, fig_ax=(fig, ax), x_label="# of user's followings",
                        style=styles[1], marker=markers[1], legend_label='all')
fig.savefig('./eps/cdf/followings_cdf' + TYPE)

# Plot total favorited count pdf
eval = Evaluation()
total_favorited = attributes.total_favorited.tolist()
total_favorited_unique = attributes.groupby(['user_id']).total_favorited.head(1).tolist()
fig, ax = eval.draw_pdf(total_favorited_unique, y_label="# of total liked",
                        style=styles[0], marker=markers[0], legend_label='unique')
fig, ax = eval.draw_pdf(total_favorited, fig_ax=(fig, ax), y_label="# of user's total_favorited",
                        style=styles[1], marker=markers[1], legend_label='all')
fig.savefig('./eps/pdf/total_favorited_pdf' + TYPE)

# Plot total favorited count cdf
print("[User total favorited cdf]")
fig, ax = eval.draw_cdf(total_favorited_unique, x_label="# of total liked",
                        style=styles[0], marker=markers[0], legend_label='unique')
fig, ax = eval.draw_cdf(total_favorited, fig_ax=(fig, ax), x_label="# of user's total_favorited",
                        style=styles[1], marker=markers[1], legend_label='all')
fig.savefig('./eps/cdf/total_favorited_cdf' + TYPE)

# Plot followers and followings
followers = attributes.follower_count.values
followings = attributes.following_count.values
fig, ax = eval.draw_realation(followers, followings)
fig.savefig('./eps/relation/follower_following' + TYPE)


# Plot aweme count pdf
awemes = attributes[attributes.aweme_count>0].aweme_count.tolist()
awemes_unique = attributes[attributes.aweme_count>0].groupby(['user_id']).aweme_count.head(1).tolist()
fig, ax = eval.draw_pdf(awemes_unique, y_label="# of video clips",
                        style=styles[0], marker=markers[0], legend_label='unique')
fig, ax = eval.draw_pdf(awemes, fig_ax=(fig, ax), y_label="p(X)", x_label='# of Videos',
                        style=styles[1], marker=markers[1], legend_label='all')
fig.savefig('./eps/pdf/awemes_pdf' + TYPE)


# Plot aweme count cdf
print("[User aweme cdf]")
awemes = attributes.aweme_count.tolist()
awemes_unique = attributes.groupby(['user_id']).aweme_count.head(1).tolist()
fig, ax = eval.draw_cdf(awemes_unique, x_label="# of video clips",
                        style=styles[0], marker=markers[0], legend_label='unique')
fig, ax = eval.draw_cdf(awemes, fig_ax=(fig, ax), x_label="# of video clips",
                        style=styles[1], marker=markers[1], legend_label='all')
fig.savefig('./eps/cdf/awemes_cdf' + TYPE)



# Plot video duration pdf
video_duration = (own_videos.duration/1000).tolist()
video_duration2 = (like_videos.duration/1000).tolist()
fig, ax = eval.draw_pdf2(video_duration, style=styles[0], legend_label="Publish")
fig, ax = eval.draw_pdf2(video_duration2, fig_ax=(fig, ax), style=styles[1], legend_label="Like")
fig.savefig('./eps/pdf/video_duration_pdf' + TYPE)

# Plot video duration cdf
print("[Video duration cdf]")
fig, ax = eval.draw_cdf(video_duration, x_label="Video Duration (s)", legend_label="Publish",
                        style=styles[0], marker=markers[0], is_log=False)
fig, ax = eval.draw_cdf(video_duration2, x_label="Video Duration (s)",legend_label="Like",
                        style=styles[1], marker=markers[1], is_log=False, fig_ax=(fig, ax))
fig.savefig('./eps/cdf/video_duration_cdf' + TYPE)


videos = like_videos
digg_count = videos[videos.digg_count>0].digg_count.tolist()
comment_count = videos[videos.comment_count>0].comment_count.tolist()
share_count = videos[videos.share_count>0].share_count.tolist()


fig, ax = eval.draw_pdf(digg_count, y_label="p(X)",
                        style=styles[0], marker=markers[0], legend_label='like')
fig, ax = eval.draw_pdf(comment_count, fig_ax=(fig, ax), y_label="p(X)",
                        style=styles[1], marker=markers[1], legend_label='comment')
fig, ax = eval.draw_pdf(share_count, fig_ax=(fig, ax), y_label="p(X)",
                        style=styles[2], marker=markers[2], legend_label='share')
fig.savefig('./eps/pdf/video_popularity_pdf' + TYPE)

digg_count = videos.digg_count.tolist()
comment_count = videos.comment_count.tolist()
share_count = videos.share_count.tolist()

# Plot video digg_count cdf
print("[Video digg_count cdf]")
fig, ax = eval.draw_cdf(digg_count, x_label="counts", legend_label='like',
                        style=styles[0], marker=markers[0])
fig, ax = eval.draw_cdf(comment_count, fig_ax=(fig, ax), x_label="counts", legend_label='comment',
                        style=styles[1], marker=markers[1])
fig, ax = eval.draw_cdf(share_count, fig_ax=(fig, ax), x_label="counts", legend_label='share',
                        style=styles[2], marker=markers[2])
fig.savefig('./eps/cdf/video_popularity_cdf' + TYPE)
'''
# Plot video bit_rate pdf
videos = own_videos
bit_rate1 = (videos.bit_rate1/1000).tolist()
bit_rate2 = (videos.bit_rate2.dropna()/1000).tolist()
fig, ax = eval.draw_pdf(bit_rate1, y_label="bit rates (Mbps)",
                        style=styles[0],  legend_label='first bitrate')
fig, ax = eval.draw_pdf(bit_rate2, fig_ax=(fig, ax), y_label="bit rates (Kbps)",
                        style=styles[1],  legend_label='second bitrate')
fig.savefig('./eps/pdf/bit_rate_pdf' + TYPE)

# Plot video bit_rate1 cdf
print("[Video bit_rate cdf]")
fig, ax = eval.draw_cdf(bit_rate1, x_label="bit rates (Mbps)",
                        style=styles[0], marker=markers[0], legend_label='first bitrate')
fig, ax = eval.draw_cdf(bit_rate2, fig_ax=(fig, ax), x_label="bit rates (Kbps)",
                        style=styles[1], marker=markers[1], legend_label='second bitrate')
fig.savefig('./eps/cdf/bit_rate_cdf' + TYPE)

own_videos_no_2 = own_videos[own_videos.bit_rate2.isnull()]
#print(own_videos_no_2.describe())
'''

# Plot resolution pie
videos = like_videos
ratio = videos.ratio.tolist()
fig, ax = eval.draw_pie(ratio)
fig.savefig('./eps/pie/resolution' + TYPE)
'''





