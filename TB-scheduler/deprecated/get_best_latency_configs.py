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
    # -----------------------------------------------------------
    # model_name = 'mobilenet_v2'
    # model_name = 'xception'
    model_name = 'mnasnet1_0'
    filename = 'gen_data/plot_data/' + model_name + '/800/latency.csv'
    df = df = pd.read_csv(filename)
    indices = list(range(0, len(df)))
    df = df.sort_values(by=['total_mac_cycles'])
    # df['total_mac_cycles'].astype(int)
    # data = df.values.transpose().tolist()

    pdp=0
    dp = 0
    p=0
    sched_idx = 4
    for idx, row in df.iterrows():
        if row[sched_idx] == 'pdp' and pdp == 0:
            print(row[0], row[sched_idx])
            pdp += 1
        if row[sched_idx] == 'dp' and dp == 0:
            dp +=1
            print(row[0], row[sched_idx])

        if row[sched_idx] == 'per_layer' and p == 0:
            p +=1
            print(row[0], row[sched_idx])

    print('end')