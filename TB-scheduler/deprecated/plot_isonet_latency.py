# %matplotlib inline
import sys
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import numpy as np
from random import randint
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages

from palettable.colorbrewer.sequential import YlGnBu_5

model_name = 'mobilenet_v2'
total_mac = 'isonet'
data_file = './gen_data/plot_data/' + model_name + '/' + total_mac \
              + '/isonet_latency_' + model_name + '_.csv'
result_dir = './gen_data/plot_data/' + model_name + '/' + total_mac + '/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_file = result_dir + '/isonet_latency_' + model_name + '.pdf'

# Reading data
df = pd.read_csv(data_file)

# Plot type
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(10, 8))
plt.gcf().subplots_adjust(bottom=0.20)
plt.xticks(rotation=90)

# Set limits for X and Y axises
# plt.xlim(-0.5, 10.5)
plt.ylim(0, 1)


N = len(df['layer'])
ind = np.arange(N)
width = 0.6

c = -1
for x in np.arange(N):
   if(x%3 == 0):
       c = c + 1
   ind[x] = c
   c = c + 1

print(ind)

# bar.hatch -> puting patterns on the colors
# opbars = ax.bar(ind, df['ref'].values.tolist(), width, ecolor='k',
#         color=YlGnBu_5.hex_colors[0], edgecolor='k', hatch='//');

opbars = ax.bar(ind, df['value'].values.tolist(), width, ecolor='k',
        color=YlGnBu_5.hex_colors[4], edgecolor='k');



ax.set_ylabel('Latency (Lower is better)',fontsize=30)
ax.yaxis.label.set_color('black')
ax.set_xticks(ind)

# Adding extra name to the x labels
# rotation='degree' for rotating the text
ax.set_xticklabels(df['layer'], fontsize=16)
# t = 10
ax.text(0.5, -3.05, '|', fontsize=20)
ax.text(2.9, -3.9, '|', fontsize=20)
ax.text(2.9, -5.9, '|', fontsize=20)
ax.text(2.9, -7.9, '|', fontsize=20)


ax.text(7.2, -1.9, '|', fontsize=20)
ax.text(7.2, -3.9, '|', fontsize=20)
ax.text(7.2, -5.9, '|', fontsize=20)
ax.text(7.2, -7.9, '|', fontsize=20)


# ax.text(0, -1, 'Early Layers', fontsize=22)
ax.text(0.7, -0.25, 'Small', fontsize=26)
ax.text(4.5, -0.25, 'Large', fontsize=26)


### Style
# Set the background color
ax.set_facecolor('whitesmoke')

plt.gca().xaxis.grid(False)
plt.gca().yaxis.grid(True, color='black')

plt.tick_params( axis='x', which='both', bottom=False, top=False, colors='black', labelsize=26)
plt.tick_params( axis='y', which='both', right=False, colors='black', labelsize=30 )

plt.tick_params(axis='both', which='major', direction='in', 
                length=6, width=3,color='black')
plt.grid(linestyle='--')

ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')

# Adding legend and the position
# ax.legend((pbars[0], opbars[0], cbars[0], prec[0]), ('A', 'B', 'C', 'D'), bbox_to_anchor=(1, 0.92), fontsize=22)
plt.show()
fig.savefig(result_file, facecolor=fig.get_facecolor(), bbox_inches=None)

