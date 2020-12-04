import pandas as pd
from load_model import Net
import os



def load_df(dnn_name, result_file_name):

    df = pd.read_csv(result_file_name, index_col=0, header=None).T
    net_df = pd.read_csv(dnn_name)
    net = Net(net_df)
    layer_name = []
    layer_attr_type = []

    for k, v in net.layers.items():
        layer_name.append(k)
        layer_attr_type.append(v.attr_type)

    mac_utilization = df['cumm_mac_cycles'] / df['theoretical_max_mac_cycles'] * 100
    padd_utilization = df['padd_total'] / df['theoretical_max_padd_total'] * 100
    df['mac_utilization'] = (mac_utilization)
    df['padd_utilization'] = padd_utilization
    df['name'] = layer_name
    df['attr_type'] = layer_attr_type
    # df[~df['attr_type'].str.contains("FC")]
    # df['mac_utilization'].fillna(0, inplace=True)
    df = df[~df['attr_type'].str.contains("3d")]
    df['padd_utilization'].fillna(0, inplace=True)

    return df


def calculate_total_mac_utilization(df):
    cumm_mac_cycles = df['cumm_mac_cycles'].sum()
    theoretical_max_mac_cycles = df['theoretical_max_mac_cycles'].sum()
    return cumm_mac_cycles/theoretical_max_mac_cycles*100


def extract_filename(result_file_name):
    # extract filename
    filename = result_file_name.split('/')[-1]
    params = filename.split('_')
    schedule_name = ''
    for idx, x in enumerate(params):
        if x == 'schedule':
            if params[idx - 1] != 'hwcf':
                raise Exception('Only hwc schedule supported')

            if params[idx + 1].isdigit():
                schedule_name = 'per_layer'
            elif params[idx + 1] == 'pdp':
                schedule_name = 'pdp'
            elif params[idx + 1] == 'dw':
                schedule_name = 'dp'
            elif params[idx + 1] == 'pw':
                schedule_name = 'pd'
            else:
                raise Exception('Unknown schedule')
            break
    return schedule_name
