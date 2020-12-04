import pandas as pd
import numpy as np
import helpers as u
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from collections import *
from operator import add
import pprint
# import all_model_names
from sklearn import preprocessing

def normalize(raw):
    norm = [float(i)/sum(raw) for i in raw]
    return norm





if __name__ == '__main__':
    # pd.set_option('display.mpl_style', 'default')
    plt.rcParams['figure.figsize'] = (15, 9)
    mpl.rc('font', family='serif')
    mpl.rcParams['xtick.major.pad'] = '12'
    mpl.rcParams['ytick.major.pad'] = '8'
    fig, ax = plt.subplots()

    # -----------------------------------------------------------
    filename = 'gen_data/plot_data/mobilenet_v2/800/latency.csv'
    df = df = pd.read_csv(filename)
    indices = list(range(0, len(df)))
    data = df.values.transpose().tolist()
    # print(df)
    print(df.keys())
    # removing index names  and app names
    # K = test_data.pop(0)

    # get header names
    headers = list(df.keys())
    names = data.pop(0)
    data[0] = normalize(data[0])
    data[1] = normalize(data[1])
    data[1] = [x * 1000 for x in data[1]]
    # acc_list = create_list(all_model_names.model_accuracy_0_top1, names)
    y_offset = np.array([0.0] * len(names))
    print(data)
    # -----------------------------------------------------------
    colour = u.getpalette()
    colour = ['#d7191c', '#fdae61', '#ffffbf', '#abdda4', '#2b83ba', '#acacac']
    cell_text = headers[1:]
    c = 0
    a = [1,2,3]

    plt.scatter(indices, data[0], s=data[1]*100, color=colour[c], edgecolor='k', label=cell_text[c])

        # y_offset += row
        # c += 1
        # cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])

    # rects1 = [rect for rect in ax.get_children() if isinstance(rect, mpl.patches.Rectangle)]
    # autolabel(rects1,headers)
    # lgd = plt.legend(bbox_to_anchor=(0.0, 1.05, 1, 0.1), loc=9, ncol=4,
    #                  columnspacing=1, borderaxespad=0, frameon=False, fontsize=28)
    plt.show()

    # plt.xlim(-0.5, len(names)+0.25)
    # plt.ylim(0,100)
    # ax.set_ylabel('Total MACs', fontsize=28)
    # ax.set_xticks(indices)
    # ax.set_xticklabels(names)
    # for tick in ax.get_xticklabels():
    #     tick.set_rotation(90)
    # plt.tick_params(axis='x', which='both', bottom='off', top='off')
    # plt.tick_params(axis='y', which='both', right='off')
    # # plt.xticks(rotation=90)
    # plt.tick_params(axis='both', which='major', labelsize=28)
    # ax.tick_params(axis='both', which='major', labelsize=28)
    # plt.tight_layout()
    #
    # # ax.spines['bottom'].set_color('black')
    # # ax.spines['top'].set_color('black')
    # # ax.spines['right'].set_color('black')
    # # ax.spines['left'].set_color('black')
    # # ax.set_axis_bgcolor('white')
    # plt.show()
    # fig.savefig('../plot_data/plots/cycles.pdf', bbox_inches='tight')

    print('end')