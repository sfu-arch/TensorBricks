import pandas as pd
from load_model import Net
from dnn_schedules.hwcf import HWCFSchedule
import datetime
import argparse
import os
from plots.get_constraints.get_params import *
from attr import AttrDict

def select_schedule(_net, _model_name, _result_dir, _verbose, val, _hardware_dict):
    if val == 0:
        schedule = HWCFSchedule(_net, _model_name, _result_dir, _verbose, hardware_dict=_hardware_dict)
    else:
        raise Exception('wrong val')

    schedule.run_model()
    schedule.print_stats()
    return


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, required=True,
                        help="0=HWCFSchedule")
    parser.add_argument('-m', type=int, required=True,
                        help="total macs = 800")

    args = parser.parse_args()
    schedule_value = args.s
    # total_mac = args.m
    total_padds = 800
    step_size = 400
    total_mac= 'one_for_all_800'
    model_names = ['mnasnet1_0','vgg16', 'mobilenet_v2', 'squeezenet1_0','resnet152', 'xception', 'alexnet']
    data_folder = './raw_data/benchmarks/'
    hardware_yaml = None

    verbose = False

    file = 'per_layer_constraint_800_for_all.csv'
    new_df = pd.read_csv("plots/get_constraints/results_selected/" + file)

    for model_name in model_names[0:1]:
        df = pd.read_csv(data_folder + model_name + '.csv')
        result_dir = './generated/benchmarks_results/' + model_name + '/' + str(total_mac) + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        net = Net(df)

        hw_dict = AttrDict()
        hw_dict_list = []

        for index, row in new_df.iterrows():
            print(index, row)

            hw_dict_list = get_one_per_layer_params(row['wxx'], row['cxx'], row['fx'], row['cx'], row['wx'],total_padds)
            schedule_value = 0


            for hw_dict in hw_dict_list:
                select_schedule(net, model_name, result_dir, verbose, schedule_value, hw_dict)

        print(datetime.datetime.now())

