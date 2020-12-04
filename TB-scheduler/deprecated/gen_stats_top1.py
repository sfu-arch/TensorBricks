import pandas as pd

from deprecated.df_utility import *

if __name__ == '__main__':
    model_name = 'mobilenet_v2'
    dnn_name = './raw_data/benchmarks/' + model_name + '.csv'
    data_folder = 'gen_data/benchmarks_results/mobilenet_v2/800_top1'
    column_list = [['name','cycles_total', 'total_dma_accesses', 'total_memory_mb','mac_utilization', 'padd_utilization']]
    stat_df = pd.DataFrame(columns=column_list)
    i=0

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            result_file_name = os.path.join(data_folder, file)
            schedule_name = extract_filename(result_file_name)
            df = load_df(dnn_name, result_file_name)
            df = df.reset_index()
            cycles_total = int(df['cycles_total'].sum())
            total_dma_accesses = int(df['in_dma_act'].sum() + df['out_dma_act'].sum() + df['in_dma_wgt'].sum())

            mem_wgt = df['mem_wgt'].max()
            mem_in_act = df['mem_in_act'].max()
            mem_out_act = df['mem_out_act'].max()
            mem_partial_product = df['mem_partial_product'].max()
            total_memory = mem_wgt + mem_in_act + mem_out_act + mem_partial_product
            total_memory_mb = round(total_memory/(1024*1024),2)
            mac_utilization = df['cumm_mac_cycles'].sum() / df['theoretical_max_mac_cycles'].sum() * 100
            mac_utilization = round(mac_utilization,2)
            padd_utilization = df['padd_total'].sum() / df['theoretical_max_padd_total'].sum() * 100
            padd_utilization = round(padd_utilization,2)
            column_list.append([schedule_name, cycles_total, total_dma_accesses,
                                total_memory_mb, mac_utilization, padd_utilization])

            print('end')

    outputfile = 'gen_data/plot_data/mobilenet_v2/800/plot_top1_stats.csv'
    with open(outputfile, 'w') as file:
        for row in column_list:
            csvstring = (',').join([str(x) for x in row])
            print(csvstring)
            file.write(csvstring)
            file.write('\n')
        file.close()