import pandas as pd
from deprecated.df_utility import *

def get_feature(i,df):

    pw1_key = 'features.' + str(i) + '.conv.0.0'
    dw_key =  'features.' + str(i) + '.conv.1.0'
    pw2_key = 'features.' + str(i) + '.conv.2'

    pw1_row = df.loc[df['name'] == pw1_key]
    dw_row = df.loc[df['name'] == dw_key]
    pw2_row = df.loc[df['name'] == pw2_key]

    pw1_cycle = int(pw1_row['cycles_total'].iloc[0])
    dw_cycle = int(dw_row['cycles_total'].iloc[0])
    pw2_cycle = int(pw2_row['cycles_total'].iloc[0])

    cumm_mac_cycles = pw1_row['mac_cycles'].iloc[0] + dw_row['mac_cycles'].iloc[0] + pw2_row['mac_cycles'].iloc[0]
    cumm_cycles = pw1_cycle + dw_cycle + pw2_cycle

    # if pw1_cycle < dw_cycle:
    #     print('pw1<dw')
    # if pw1_cycle < pw2_cycle:
    #     print('pw1<pw2')
    # if pw2_cycle < dw_cycle :
    #     print('dw > pw2')

    return pw1_cycle,dw_cycle, pw2_cycle, cumm_mac_cycles, cumm_cycles


def get_average_cycle_mac_util(df):
    early = 0
    middle = 1
    late = 2
    cycles = [0]*9
    cumm_mac = [0]*3
    total_cycles = [0]*3
    num_layers = [0, 0, 0]
    for i in range(2, 17):
        # print('layer : {}'.format(i))
        pw1_cycle,dw_cycle, pw2_cycle, cumm_mac_cycles, cumm_cycles = get_feature(i, df)
        if 2 <= i <= 4:
            cycles[0] += pw1_cycle
            cycles[1] += dw_cycle
            cycles[2] += pw2_cycle
            cumm_mac[early] += cumm_mac_cycles
            total_cycles[early] += cumm_cycles
            num_layers[early] +=1
        if 5 <= i <= 13:
            cycles[3] += pw1_cycle
            cycles[4] += dw_cycle
            cycles[5] += pw2_cycle
            cumm_mac[middle] += cumm_mac_cycles
            total_cycles[middle] += cumm_cycles
            num_layers[middle] +=1
        if 14 <= i <= 16:
            cycles[6] += pw1_cycle
            cycles[7] += dw_cycle
            cycles[8] += pw2_cycle
            cumm_mac[late] += cumm_mac_cycles
            total_cycles[late] += cumm_cycles
            num_layers[late] +=1

    avg_mac_utilization = [round(x/y*100,2) for x,y in zip(cumm_mac,total_cycles)]
    avg_cycles = [0]*9
    for i, x in enumerate(cycles):
        if i <= 2:
            avg_cycles[i] = round(x/num_layers[early],2)

        elif i <= 5:
            avg_cycles[i] = round(x / num_layers[middle],2)
        elif i <= 8:
            avg_cycles[i] = round(x / num_layers[late],2)

    # norm_avg_cycle = [round(x/max(avg_cycles)*100,2) for x in avg_cycles]

    return avg_cycles, avg_mac_utilization


if __name__ == '__main__':
    # ------------------------

    model_name = 'mobilenet_v2'
    total_mac = 920
    data_folder = 'gen_data/benchmarks_results/mobilenet_v2/iso_' + str(total_mac)+'/'
    dnn_file = './raw_data/benchmarks/' + model_name + '.csv'
    result_dir = './gen_data/plot_data/' + model_name + '/' + str(total_mac) + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    name = 'pdp-'
    name_idx = 0
    # list_strings_to_print = []
    # list_total_cycles = []
    # header_string = 'name,e-pw1,e-dw,e-pw2,m-pw1,m-dw,m-pw2,l-pw1,l-dw,l-pw2,m0,m1,m2,total_cycles'
    header_string = 'layer,avg_cycles, bench_name, mac_util,total_cycles'
    print(header_string)
    # iso_dict = {'Base': {}, 'D-P': {},'P-D-P': {}}
    iso_dict = {}

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            result_file_name = os.path.join(data_folder, file)
            schedule_name = extract_filename(result_file_name)
            bench_name = name + str(name_idx)
            iso_dict[bench_name] = {}
            df = load_df(dnn_file, result_file_name)
            total_cycles = int(df['cycles_total'].sum())
            iso_dict[bench_name]['total_cycles'] = total_cycles
            print('{} : {}'.format(bench_name,total_cycles))
            new_df = pd.DataFrame()
            new_df['attr_type'] = df.loc[4:51,'attr_type']
            new_df['cycles_total'] = df.loc[4:51,'cycles_total']

            # ax = new_df.plot.bar(x='attr_type', y='cycles_total', rot=90)
            # plt.show()

            result_list = []
            new_df = pd.DataFrame(columns=['c0','c1' 'c2','m0','m1','m2'])
            avg_cycles, avg_mac_util = get_average_cycle_mac_util(df)
            iso_dict[bench_name]['avg_cycles'] = avg_cycles
            iso_dict[bench_name]['avg_mac_util'] = avg_mac_util
            # cyc_string = ",".join(str(round(x,2)) for x in norm_avg_cycles)
            # mac_string = ",".join(str(round(x,2)) for x in avg_mac_util)
            #
            # total_string = bench_name + cyc_string + mac_string
            # # print(total_string)
            # list_strings_to_print.append(total_string)
            # list_total_cycles.append(total_cycles)
            name_idx += 1

    print(iso_dict)
    # max_cycles = []
    total_cycles = []
    for key, value in iso_dict.items():
        # max_cycles.append(max(value['avg_cycles']))
        total_cycles.append(value['total_cycles'])

    # all_max_cycles = max(max_cycles)
    max_total_cycle = max(total_cycles)
    # -- Generate csv --
    with open(result_dir + 'pdp_pipeline.csv', 'w') as f:
        f.write(header_string)
        f.write('\n')
        for bench_name, value in iso_dict.items():
            mac_util = value['avg_mac_util']
            norm_total_cycles = round(value['total_cycles']/max_total_cycle,2)

            for i, x in enumerate(value['avg_cycles']):
                # norm_cycles = round(x/all_max_cycles,2)
                norm_cycles = round(x / max_total_cycle, 2)
                if i< 3:
                    mac_util = value['avg_mac_util'][0]
                elif 3 <= i < 6:
                    mac_util = value['avg_mac_util'][1]
                elif 6<= i < 9:
                    mac_util = value['avg_mac_util'][2]
                csv_str = ''
                if i % 3 == 0:
                   csv_str = 'PT-1,{},{},{},{}'.format(norm_cycles, bench_name, mac_util,norm_total_cycles)
                elif i % 3 == 1:
                    csv_str = 'DP-2,{},{},{},{}'.format(norm_cycles, bench_name, mac_util,norm_total_cycles)
                elif i % 3 == 2:
                    csv_str = 'PT-3,{},{},{},{}'.format(norm_cycles, bench_name, mac_util,norm_total_cycles)
                f.write(csv_str)
                f.write('\n')


    #     f.write(header_string)
    #     f.write('\n')
    #     norm_total_cycles = [round(x/max(list_total_cycles),2) for x in list_total_cycles]
    #     # print(norm_total_cycles)
    #     for cycle, row in zip(norm_total_cycles, list_strings_to_print):
    #         f.write(row + ',' + str(cycle))
    #         print(row + ',' + str(cycle))
    #         f.write('\n')

        # f.close()
