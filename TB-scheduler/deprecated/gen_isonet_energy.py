import pandas as pd
import operator
from deprecated.df_utility import *


def normalize_dict(isonet_dict):
    new_dict = {}
    for folder, value in isonet_dict.items():
        new_dict[folder] = {}
        max_key = max(value.items(), key=operator.itemgetter(1))[0]
        norm_dict = {k: v/value[max_key] for k, v in value.items()}
        print(norm_dict)
        new_dict[folder] = norm_dict

    return new_dict


if __name__ == '__main__':
    model_name = 'mobilenet_v2'
    folders_list = ['top1_800', 'top1_2200']
    dnn_name = './raw_data/benchmarks/' + model_name + '.csv'
    data_folder_prefix = 'gen_data/benchmarks_results/' + str(model_name) + '/'
    result_dir = 'gen_data/plot_data/' + str(model_name) + '/isonet/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = result_dir + '/isonet_energy_'+ model_name +'_.csv'

    isonet_dict = {}
    isonet_energy = pd.DataFrame()
    for folder in folders_list:
        isonet_dict[folder] = {}
        current_folder = data_folder_prefix + folder
        i=0
        for file in os.listdir(current_folder):
            if file.endswith(".csv"):
                result_file_name = os.path.join(current_folder, file)
                schedule_name = extract_filename(result_file_name)
                df = load_df(dnn_name, result_file_name)
                total_dma = int(df['in_dma_act'].sum()) + int(df['in_dma_wgt'].sum())
                print('{},{},{}'.format(folder, schedule_name, total_dma))
                isonet_dict[folder][schedule_name] = total_dma

    isonet_dict = normalize_dict(isonet_dict)

    with open(result_file,'w') as f:
        header='layer,value,ref'
        f.write(header)
        f.write('\n')
        for folder,value in isonet_dict.items():
            f.write('Base,{},{}'.format(value['per_layer'], folder))
            f.write('\n')
            f.write('D-P,{},{}'.format(value['dp'], folder))
            f.write('\n')
            f.write('P-D-P,{},{}'.format(value['pdp'], folder))
            f.write('\n')
        f.close()