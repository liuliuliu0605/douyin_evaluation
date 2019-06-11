import powerlaw_revised
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

user_detail_file_name = 'user.txt.processed'
data_dir = os.path.join(os.getcwd(), 'dataset')
user_detail_file_path = os.path.join(data_dir, user_detail_file_name)
data_type = {'uid': np.object, 'short_id': np.object}
user_detail = pd.read_csv(user_detail_file_path, delimiter='\t', dtype=data_type)
followers = user_detail[user_detail.follower_count > 0][['follower_count', 'weight']]
sequence = np.array(followers.follower_count) #!!!
weights = np.array(followers.weight) #!!!
weights /= sum(weights)
fit = powerlaw_revised.Fit(sequence, weights=weights, xmin=1)
powerlaw_revised.plot_pdf(sequence, weights=weights)
print(type(fit.power_law))
print("alpha:", fit.power_law.alpha)
print("xmin:", fit.power_law.xmin)
#print("mu:", fit.lognormal.mu)
#print("sigma:", fit.lognormal.sigma)
fit.power_law.plot_pdf()
#fit.exponential.plot_pdf()
plt.show()