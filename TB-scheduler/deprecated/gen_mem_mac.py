import pandas as pd

from deprecated.df_utility import *

if __name__ == '__main__':
    model_name = 'mobilenet_v2'
    total_mac = 800
    dnn_name = './raw_data/benchmarks/' + model_name + '.csv'
    data_folder = 'gen_data/benchmarks_results/mobilenet_v2/top1_' + str(total_mac)
    result_dir = 'gen_data/plot_data/mobilenet_v2/' + str(total_mac)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = result_dir + '/plot_mem_mac.csv'


    mem_per_total_df = pd.DataFrame()
    i=0
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            result_file_name = os.path.join(data_folder, file)
            schedule_name = extract_filename(result_file_name)
            df = load_df(dnn_name, result_file_name)
            df = df.reset_index()
            if i==0:
                mem_per_total_df['name'] = df['name']
                mem_per_total_df['attr_type'] = df['attr_type']
                i+=1
            mem_per_total_df[schedule_name] = df['is_mac_cycle_selected'] / (df['is_dma_cycle_selected'] + df['is_mac_cycle_selected']) * 100
            mem_per_total_df[schedule_name + '_mac_util'] = df['mac_cycles']/(df['cycles_total']) *100
            mem_per_total_df[schedule_name + '_dma_util'] = 100 - mem_per_total_df[schedule_name + '_mac_util']
            print('end')
    print('end')
    mem_per_total_df = mem_per_total_df.reindex(columns=['name', 'attr_type',
                                                         'per_layer', 'per_layer_mac_util', 'per_layer_dma_util',
                                                         'dp', 'dp_mac_util', 'dp_dma_util',
                                                         'pdp', 'pdp_mac_util', 'pdp_dma_util'])
    mem_per_total_df.to_csv(result_file,index=False, float_format='%.1f')