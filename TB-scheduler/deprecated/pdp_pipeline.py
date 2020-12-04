# %matplotlib inline
import sys
import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import numpy as np
from random import randint
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages

from palettable.colorbrewer.sequential import YlGnBu_5


# Reading data
df = pd.read_csv('../plot_data/pdp_pipeline.csv')

# Plot type
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(32, 8))
# plt.xticks(rotation=90)

# Set limits for X and Y axises
# plt.xlim(-0.5, 10.5)
plt.ylim(0, 1)


N = len(df['layer'])
ind = np.arange(N)
width = 0.6

c = -1
for x in np.arange(N):
   if(x%10 == 0):
       c = c + 1
   ind[x] = c
   c = c + 1

# print(ind)

# bar.hatch -> puting patterns on the colors
# opbars = ax.bar(ind, df['ref'].values.tolist(), width, ecolor='k',
#         color=YlGnBu_5.hex_colors[0], edgecolor='k', hatch='//');

colour = ['#ffeda0','#feb24c','#f03b20']

opbars = ax.bar(ind, df['avg_cycles'].values.tolist(), width, ecolor='k',
        color=colour[0], edgecolor='k');

opbars = ax.bar(ind, df['middle'].values.tolist(), width, ecolor='k',
        color=colour[1], edgecolor='k');

opbars = ax.bar(ind, df['late'].values.tolist(), width, ecolor='k',
        color=colour[2], edgecolor='k');

opbars = ax.bar(ind, df['total_cycles'].values.tolist(), width, ecolor='k',
        color=YlGnBu_5.hex_colors[4], edgecolor='k', hatch='//');



ax.set_ylabel('Latency',fontsize=32)
ax.yaxis.label.set_color('black')
ax.set_xticks(ind);

# Adding extra name to the x labels
# rotation='degree' for rotating the text
ax.set_xticklabels(df['layer'], fontsize=16, rotation=90)
t = 10



ax.text(0, -0.25, 'Early', fontsize=32)
ax.text(3, -0.25, 'Middle', fontsize=32)
ax.text(6.5, -0.25, 'Late', fontsize=32)


ax.text(10, -0.05, '|', fontsize=20)
ax.text(10, -0.08, '|', fontsize=20)
ax.text(10, -0.11, '|', fontsize=20)
ax.text(10, -0.14, '|', fontsize=20)

ax.text(11, -0.25, 'Early', fontsize=32)
ax.text(14, -0.25, 'Middle', fontsize=32)
ax.text(17.5, -0.25, 'Late', fontsize=32)

ax.text(21, -0.05, '|', fontsize=20)
ax.text(21, -0.08, '|', fontsize=20)
ax.text(21, -0.11, '|', fontsize=20)
ax.text(21, -0.14, '|', fontsize=20)

ax.text(22, -0.25, 'Early', fontsize=32)
ax.text(25, -0.25, 'Middle', fontsize=32)
ax.text(28.5, -0.25, 'Late', fontsize=32)


# ax.text(0, -2, 'Early Layers', fontsize=22)
# ax.text(4, -2, 'Middle Layers', fontsize=22)
# ax.text(8, -2, 'Late Layers', fontsize=22)

ax.text(3, 1.05, 'pdp-0', fontsize=32)
ax.text(14, 1.05, 'pdp-1', fontsize=32)
ax.text(24, 1.05, 'pdp-2', fontsize=32)


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

fig.savefig('test.pdf',facecolor=fig.get_facecolor(), bbox_inches='tight')
