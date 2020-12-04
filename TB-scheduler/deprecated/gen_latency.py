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

    total_cycles = df['cycles_total'].sum()
    total_dma_accesses = df['in_dma_act'].sum() + df['out_dma_act'].sum() + df['in_dma_wgt'].sum()
    mem_wgt = df['mem_wgt'].max()
    mem_in_act = df['mem_in_act'].max()
    mem_out_act = df['mem_out_act'].max()
    mem_partial_product = df['mem_partial_product'].max()
    total_memory = mem_wgt + mem_in_act + mem_out_act + mem_partial_product
    total_memory_mb = round(total_memory / (1024 * 1024), 2)

    return total_cycles, total_dma_accesses, total_memory_mb

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


if __name__ == '__main__':

    model_name ='mobilenet_v2'
    total_mac = 'iso_920'
    dnn_name = './raw_data/benchmarks/' + model_name + '.csv'
    data_folder = './gen_data/benchmarks_results/' + model_name + '/' + total_mac + '/'

    result_dir = './gen_data/plot_data/' + model_name + '/' + total_mac + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_dir + 'latency.csv', 'w') as f:
        print(model_name + ',total_cycles,total_dma_accesses,total_memory_mb,schedule_name')
        f.write(model_name + ',total_cycles,total_dma_accesses,total_memory_mb,schedule_name')

        f.write('\n')
        for file in os.listdir(data_folder):
            if file.endswith(".csv"):
                result_file_name = os.path.join(data_folder, file)
                schedule_name = extract_filename(result_file_name)
                total_cycles, total_dma_accesses, total_memory_mb = load_df(dnn_name, result_file_name)
                f.write('{},{},{},{},{}'.format(file, int(total_cycles), int(total_dma_accesses), round(total_memory_mb, 2), schedule_name))
                f.write('\n')
                print('{},{},{},{},{}'.format(file, int(total_cycles), int(total_dma_accesses), round(total_memory_mb, 2), schedule_name))
        f.close()

