import pandas as pd
from load_model import Net
from dnn_schedules.cross_layer.hwcf_dw_pw import HWCFScheduleDWPW
from dnn_schedules.deprecated.hwcf_schedule import HWCFSchedule
from dnn_schedules.cross_layer.hwcf_pw_dw import HWCFSchedulePWDW
from dnn_schedules.cross_layer.pdp_pipeline import HWCFSchedulePDP
import datetime
import argparse
import os


def select_schedule(_net, _model_name, _result_dir, _verbose, val, _hardware_dict):
    if val == 0:
        schedule = HWCFSchedule(_net, _model_name, _result_dir, _verbose, hardware_dict=_hardware_dict)
    elif val == 1:
        schedule = HWCFScheduleDWPW(_net, _model_name, _result_dir, _verbose, hardware_dict=_hardware_dict)
    elif val == 2:
        schedule = HWCFSchedulePWDW(_net, _model_name, _result_dir, _verbose, hardware_dict=_hardware_dict)
    elif val == 3:
        schedule = HWCFSchedulePDP(_net, _model_name, _result_dir, _verbose, hardware_dict=_hardware_dict)
    else:
        raise Exception('wrong val')

    schedule.run_model()
    schedule.print_stats()
    return


if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, required=True,
                        help="0=HWCFSchedule, 1=HWCFScheduleDWPW, 2=HWCFSchedulePWDW, 3=HWCFSchedulePDP")
    parser.add_argument('-m', type=int, required=True,
                        help="total macs = 100, 500, 1000, 3000")

    args = parser.parse_args()
    schedule_value = args.s
    total_mac = args.m
    total_padds = total_mac
    step_size = 400

    # model_names = ['pdp', 'efficientnet-b3', 'xception', 'densenet161',
    #                  'inception_v3', 'resnet152', 'efficientnet-b0', 'resnet50',
    #                  'mobilenet_v2', 'nasnetamobile', 'mnasnet1_0', 'vgg16',
    #                  'mobilenet', 'resnet18', 'shufflenet_v2_x1_0',
    #                  'squeezenet1_0', 'alexnet']

    model_names = ['mobilenet_v2', 'xception', 'efficientnet-b0', 'nasnetamobile', 'mnasnet1_0',
                     'mobilenet', 'resnet18', 'shufflenet_v2_x1_0',
                     'squeezenet1_0', 'densenet161','efficientnet-b3','inception_v3']

    data_folder = './raw_data/benchmarks/'
    hardware_yaml = None

    verbose = False
    for model_name in model_names[1:2]:
        df = pd.read_csv(data_folder + model_name + '.csv')
        result_dir = './gen_data/benchmarks_results/' + model_name + '/' + str(total_mac) + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        net = Net(df)

        constraint_files = ['per_layer_constraint_' + str(total_mac)+'.csv',
                            'dp_constraint_' + str(total_mac)+'.csv', 'pdp_constraint' + str(total_mac)+'.csv']
        per_layer_param_list = ['wxx', 'cxx', 'fx', 'cx', 'wx']
        dp_param_list = ['wx', 'cxx2', 'fx2', 'cx']
        pdp_param_list = ['wxx', 'fx', 'cxx', 'fx2']

        if schedule_value == 0:
            file = 'per_layer_constraint_' + str(total_mac) + '.csv'
            params = per_layer_param_list
        elif schedule_value == 1:
            file = 'dp_constraint_' + str(total_mac) + '.csv'
            params = dp_param_list
        elif schedule_value == 3:
            file = 'pdp_constraint_' + str(total_mac) + '.csv'
            params = pdp_param_list
        else:
            raise Exception('Value unsupported')

        data = pd.read_csv("get_plots/get_constraints/results_selected/" + file)
        # new_df = select_pdp_constraints(data)
        # new_df = select_min_max_constraints(data, params)
        new_df = data
        hw_dict = AttrDict()
        hw_dict_list = []

        for index, row in new_df.iterrows():
            print(index, row)
            if file == 'per_layer_constraint_' + str(total_mac) + '.csv':
                hw_dict_list = get_per_layer_params(row['wxx'], row['cxx'], row['fx'], row['cx'], row['wx'],total_padds)
                schedule_value = 0
            elif file == 'dp_constraint_' + str(total_mac) + '.csv':
                hw_dict_list = get_dp_params(row['wx'], row['cxx2'], row['fx2'], row['cx'], total_padds)
                schedule_value = 1
            elif file == 'pdp_constraint_' + str(total_mac) + '.csv':
                hw_dict_list = get_pdp_params(row['wxx'], row['fx'], row['cxx'], row['fx2'], total_padds, step_size)
                schedule_value = 3
            else:
                raise Exception('constraint file not known')

            for hw_dict in hw_dict_list:
                select_schedule(net, model_name, result_dir, verbose, schedule_value, hw_dict)

        print(datetime.datetime.now())

