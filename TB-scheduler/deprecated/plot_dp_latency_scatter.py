# %matplotlib inline
import sys
import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from random import randint
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_pdf import PdfPages
from palettable.colorbrewer.sequential import YlGnBu_5

model_names = ['mobilenet','xception','nasnetamobile']
model_name = model_names[1]
# model_name = 'pdp'
total_mac = '800'
data_folder = './gen_data/plot_data/' + model_name + '/' + total_mac + '/latency.csv'
result_dir = './gen_data/plot_data/' + model_name + '/' + total_mac + '/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_file = result_dir + '/latency_scatter_' + model_name + '.pdf'
# Plot type
plt.style.use('ggplot')

#Loading data
df = pd.read_csv(data_folder)

df['total_mac_cycles'] = df['total_mac_cycles'] / df['total_mac_cycles'].max()
df['total_dma_accesses'] = df['total_dma_accesses'] / df['total_dma_accesses'].max()
df['total_memory_mb'] = df['total_memory_mb'] / df['total_memory_mb'].max()
df['total_memory_mb'] = [math.pow(x,6) for x in df['total_memory_mb']]

# delete rows with column name 'schedule_name' = pdp
indexNames = df[df['schedule_name'] == 'pdp'].index
# Delete these row indexes from dataFrame
df.drop(indexNames, inplace=True)
df['schedule_name'] = df['schedule_name'].replace(['per_layer','dp','pdp'],['Base','DP-PT','PT-DP-PT'])

fig, ax = plt.subplots(figsize=(12, 8))
plt.gcf().subplots_adjust(bottom=0.16, right=0.99, top=0.88)
plt.xticks(rotation=90)


# plt.xlim(-0.5, 10)

N = len(df[model_name])
ind = np.arange(N)
# width = 0.8

df_sorted = df.sort_values(by=['total_dma_accesses'])


majorLocator = MultipleLocator(0.2)
minorLocator = MultipleLocator(0.2)
# majorFormatter = FormatStrFormatter('%d')
ax.xaxis.set_major_locator(majorLocator)
ax.xaxis.set_minor_locator(minorLocator)
# ax.xaxis.set_major_formatter(majorFormatter)


# cl = {'PT-DP-PT':'blue','DP-PT':'orange','Base':'green'}
cl = {'PT-DP-PT':'blue','DP-PT':'red','Base':'green'}
marker = {'DP-PT':'x','Base':'+'}

for sch in list(set(df_sorted['schedule_name'])):
    vals = df_sorted[df_sorted['schedule_name'] == sch]
    ax.scatter(vals['total_dma_accesses'],vals['total_mac_cycles'],
                s=1000, marker = marker[sch],edgecolor='k', color=cl[sch],label=sch,
              alpha=1, edgecolors='k')

    # ax.scatter(vals['total_dma_accesses'], vals['total_mac_cycles'],
    #             s=vals['total_memory_mb']*1024*3, marker = 'o',edgecolor='k', color=cl[sch],label=sch,
    #           alpha=0.3, edgecolors='none')

# for n in np.arange(len(df['name'])):
#     ax.text(df['Area'][n]-0.06, df['Power'][n]-0.07, df['name'][n], fontsize=16)

# Put limit on Y axis
plt.ylim(0, 1.1)
plt.xlim(0.01, 1.1)

# Set X label values
ax.set_ylabel('Latency (Lower is better)',fontsize=38, color='black')
ax.set_xlabel('DRAM Energy (Lower is better)',fontsize=38, color='black')

# ax.set_xticks(ind+0.8);

# Put the labels from 'app' coulmn
# ax.set_xticklabels(u.rename(df['name']))
ax.set_facecolor('whitesmoke')

plt.gca().xaxis.grid(True, color='black')
plt.gca().yaxis.grid(True, color='black')

plt.tick_params( axis='x', which='both', bottom=False, top=False, colors='black')
plt.tick_params( axis='y', which='both', right=False, colors='black' )
plt.tick_params(axis='both', which='major', direction='in', 
                length=6, width=3,color='black', labelsize=28)
plt.grid(linestyle='--')

# plt.legend(numpoints=1)
plt.legend(bbox_to_anchor=(0.90, 1.20), ncol=3, fontsize=35)



ax.spines['bottom'].set_color('gray')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
ax.spines['left'].set_color('gray')

plt.show()
# Saving the plot
fig.savefig(result_file,facecolor=fig.get_facecolor(), bbox_inches='tight')