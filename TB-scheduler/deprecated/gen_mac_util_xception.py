from deprecated.df_utility import *

def get_feature_xception(i,j,df):
    if not (df['attr_type'].iloc[i] == 'DW' and df['attr_type'].iloc[j] == 'PW'):
        raise Exception('Wrong indices')
    cumm_mac_cycles = df['mac_cycles'].iloc[i] + df['mac_cycles'].iloc[j]
    dw_cycle = df['cycles_total'].iloc[i]
    pw_cycle = df['cycles_total'].iloc[j]
    cumm_cycles = dw_cycle + pw_cycle

    if pw_cycle < dw_cycle:
        print('pw1<dw')
        mac_util0 = df['cumm_mac_cycles'].iloc[i]/df['theoretical_max_mac_cycles'].iloc[i]*100
        mac_util1 = df['cumm_mac_cycles'].iloc[j] / df['theoretical_max_mac_cycles'].iloc[j] * 100



    return dw_cycle, pw_cycle, cumm_mac_cycles, cumm_cycles

def get_average_cycle_mac_util(df,model_name):
    if model_name != 'xception':
        raise Exception('Wrong network: only xception supported')

    total_bins = 6
    total_phases = 3
    early = 0
    middle = 1
    late = 2
    cycles = [0]*total_bins
    cumm_mac = [0]*total_phases
    total_cycles = [0]*total_phases
    num_layers = [0, 0, 0]

    early_layer_list = [(0,1),(2,3),(5,6),(7,8)]
    middle_layer_list = [(21,22),(23,24),(25,26),(27,28)]
    late_layer_list = [(63,64),(65,66),(68,69),(70,71)]

    for i,j in early_layer_list:
        print('layer : {}'.format(i))

        dw_cycle, pw_cycle, cumm_mac_cycles, cumm_cycles = get_feature_xception(i,j, df)
        cycles[0] += dw_cycle
        cycles[1] += pw_cycle
        cumm_mac[early] += cumm_mac_cycles
        total_cycles[early] += cumm_cycles
        num_layers[early] +=2

        for i, j in middle_layer_list:
            # print('layer : {}'.format(i))

            dw_cycle, pw_cycle, cumm_mac_cycles, cumm_cycles = get_feature_xception(i, j, df)
            cycles[2] += dw_cycle
            cycles[3] += pw_cycle
            cumm_mac[middle] += cumm_mac_cycles
            total_cycles[middle] += cumm_cycles
            num_layers[middle] += 2

        for i, j in late_layer_list:
            # print('layer : {}'.format(i))

            dw_cycle, pw_cycle, cumm_mac_cycles, cumm_cycles = get_feature_xception(i, j, df)
            cycles[4] += dw_cycle
            cycles[5] += pw_cycle
            cumm_mac[late] += cumm_mac_cycles
            total_cycles[late] += cumm_cycles
            num_layers[late] += 2


    # avg_mac_utilization = [round(x/y*100,2) for x,y in zip(cumm_mac,total_cycles)]
    avg_cycles = [0]*total_bins
    for i, x in enumerate(cycles):
        if i <= 1:
            avg_cycles[i] = round(x/num_layers[early],2)

        elif i <= 3:
            avg_cycles[i] = round(x / num_layers[middle],2)
        elif i <= 5:
            avg_cycles[i] = round(x / num_layers[late],2)

    # norm_avg_cycle = [round(x/max(avg_cycles)*100,2) for x in avg_cycles]

    return total_cycles, cumm_mac


if __name__ == '__main__':
    # ------------------------
    model_names = ['mobilenet_v2', 'xception', 'mnasnet1_0']
    model_name = model_names[1]
    total_mac = '/top1_800/'
    data_folder = 'gen_data/benchmarks_results/' + model_name + total_mac+'/'
    dnn_file = './raw_data/benchmarks/' + model_name + '.csv'
    result_dir = './gen_data/plot_data/' + model_name + '/' + total_mac + '/'
    result_file = result_dir + '/mac_mem_ratio.csv'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # name = 'pdp-'
    name_idx = 0
    # list_strings_to_print = []
    # list_total_cycles = []
    # header_string = 'name,e-pw1,e-dw,e-pw2,m-pw1,m-dw,m-pw2,l-pw1,l-dw,l-pw2,m0,m1,m2,total_cycles'
    header_string = 'layer,avg_cycles, bench_name, mac_util,total_cycles'
    print(header_string)
    # iso_dict = {'Base': {}, 'D-P': {},'P-D-P': {}}
    iso_dict = {}
    iso_dict_cycles = {}
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            result_file_name = os.path.join(data_folder, file)
            schedule_name = extract_filename(result_file_name)
            bench_name = schedule_name
            iso_dict[bench_name] = {}
            df = load_df(dnn_file, result_file_name)
            total_cycles = int(df['cycles_total'].sum())
            total_dma = df['in_dma_act'].sum() + df['in_dma_wgt'].sum() + df['out_dma_act'].sum()
            total_mac_cycles = int(df['mac_cycles'].sum())
            padd_cycles = int(df['padd_total'].sum()/800)
            new_mac = total_mac_cycles - padd_cycles
            new_mac_util = round(new_mac/total_cycles*100,2)
            total_mac_util = round(total_mac_cycles/total_cycles*100,2)
            # memory KB
            total_memory = round((df['mem_wgt'].max() + df['mem_in_act'].max() \
                           + df['mem_out_act'].max() + df['mem_partial_product'].max())/1024,2)
            partial_memory = round(df['mem_partial_product'].max()/1024,2)
            # iso_dict[bench_name]['total_cycles'] = total_cycles
            print('{}, total_mac_cycles: {}, total_cycles: {}, total_mac_util: {}, new_mac: {},'
                  'total_dma: {}, total onchip memory:{} KB, partial_memory: {} KB'.format(bench_name,
                total_mac_cycles, total_cycles, total_mac_util,new_mac_util, total_dma, total_memory, partial_memory))

            total_cycles, avg_mac_util = get_average_cycle_mac_util(df,model_name)
            iso_dict[bench_name] = avg_mac_util
            iso_dict_cycles[bench_name] = total_cycles
            name_idx += 1

    print(iso_dict)
    # for i in range(0,2):
    #     iso_dict['dp'][i] = iso_dict['dp'][i]/iso_dict['per_layer'][0]
    #     iso_dict['per_layer'][i] = iso_dict['per_layer'][0]/iso_dict['per_layer'][0]

    # for key, value in iso_dict.items():
    #     if key == 'dp':
    for i in range(0,3):
        iso_dict['dp'][i] =iso_dict['dp'][i]/ iso_dict_cycles['per_layer'][i]
        iso_dict_cycles['dp'][i] = iso_dict_cycles['dp'][i]/iso_dict_cycles['per_layer'][i]
        iso_dict['per_layer'][i] = iso_dict['per_layer'][i] / iso_dict_cycles['per_layer'][i]

    for i in range(0,3):
        iso_dict_cycles['per_layer'][i] = iso_dict_cycles['per_layer'][i] / iso_dict_cycles['per_layer'][i]
    # Transpose mac_dma_ratio
    csv_string_list = []
    early = 0
    middle = 1
    late = 2
    csv_string_list.append('layer,dma,mac')
    csv_string_list.append('Base,{:.2f},{:.2f}'.format(iso_dict_cycles['per_layer'][early],iso_dict['per_layer'][early]))
    csv_string_list.append('DP-PT,{:.2f},{:.2f}'.format(iso_dict_cycles['dp'][early],iso_dict['dp'][early]))
    # csv_string_list.append('PT-DP-PT,100,{:.2f}'.format(iso_dict['pdp'][early]))
    # middle
    csv_string_list.append('Base,{:.2f},{:.2f}'.format(iso_dict_cycles['per_layer'][middle], iso_dict['per_layer'][middle]))
    csv_string_list.append('DP-PT,{:.2f},{:.2f}'.format(iso_dict_cycles['dp'][middle], iso_dict['dp'][middle]))
    # csv_string_list.append('PT-DP-PT,100,{:.2f}'.format(iso_dict['pdp'][middle]))

    # late
    csv_string_list.append('Base,{:.2f},{:.2f}'.format(iso_dict_cycles['per_layer'][late], iso_dict['per_layer'][late]))
    csv_string_list.append('DP-PT,{:.2f},{:.2f}'.format(iso_dict_cycles['dp'][late], iso_dict['dp'][late]))
    # csv_string_list.append('PT-DP-PT,100,{:.2f}'.format(iso_dict['pdp'][late]))

    with open(result_file, 'w') as f:
        for row in csv_string_list:
            print(row)
            f.write(row)
            f.write('\n')
        f.close()


    # # -- Generate csv --
    # with open(result_dir + 'pdp_pipeline.csv', 'w') as f:
    #     f.write(header_string)
    #     f.write('\n')
    #     for bench_name, value in iso_dict.items():
    #         mac_util = value['avg_mac_util']
    #         norm_total_cycles = round(value['total_cycles']/max_total_cycle,2)
    #         for i, x in enumerate(value['avg_cycles']):
    #             norm_cycles = round(x/all_max_cycles,2)
    #             if i< 3:
    #                 mac_util = value['avg_mac_util'][0]
    #             elif 3 <= i < 6:
    #                 mac_util = value['avg_mac_util'][1]
    #             elif 6<= i < 9:
    #                 mac_util = value['avg_mac_util'][2]
    #             csv_str = ''
    #             if i % 3 == 0:
    #                csv_str = 'PT-1,{},{},{},{}'.format(norm_cycles, bench_name, mac_util,norm_total_cycles)
    #             elif i % 3 == 1:
    #                 csv_str = 'DP-2,{},{},{},{}'.format(norm_cycles, bench_name, mac_util,norm_total_cycles)
    #             elif i % 3 == 2:
    #                 csv_str = 'PT-3,{},{},{},{}'.format(norm_cycles, bench_name, mac_util,norm_total_cycles)
    #             f.write(csv_str)
    #             f.write('\n')


    #     f.write(header_string)
    #     f.write('\n')
    #     norm_total_cycles = [round(x/max(list_total_cycles),2) for x in list_total_cycles]
    #     # print(norm_total_cycles)
    #     for cycle, row in zip(norm_total_cycles, list_strings_to_print):
    #         f.write(row + ',' + str(cycle))
    #         print(row + ',' + str(cycle))
    #         f.write('\n')

        # f.close()
