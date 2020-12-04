import pandas as pd
from load_model import Net
from attrdict import AttrDict
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

    args = parser.parse_args()
    schedule_value =args.s

    model_names = ['pdp', 'efficientnet-b3', 'xception', 'densenet161',
                     'inception_v3', 'resnet152', 'efficientnet-b0', 'resnet50',
                     'mobilenet_v2', 'nasnetamobile', 'mnasnet1_0', 'vgg16',
                     'mobilenet', 'resnet18', 'shufflenet_v2_x1_0',
                     'squeezenet1_0', 'alexnet']

    data_folder = './raw_data/benchmarks/'
    hardware_yaml = None
    hardware_dict = AttrDict({'HWConfig': {
        # depthwise
        'hx': 7, 'wx': 7, 'cx': 6, 'dma_cycles': 1.7,
        # for pointwise
        'fx': 6, 'padd_cycles': 1, 'hxx': 7, 'wxx': 7, 'cxx': 3, 'padd_unit': 1,
        # for pointwise 2
        'fx2': 3, 'padd_cycles2': 1, 'hxx2': 5,  'wxx2': 5, 'padd_unit2': 1, 'cxx2': 6}})


    hxx_min = 5
    hxx_max = 20
    hxx_step = 5

    wxx_min = 5
    wxx_max = 20
    wxx_step = 5
    
    cxx_min = 5
    cxx_max = 10
    cxx_step = 5

    fx_min = 5
    fx_max = 10
    fx_step = 5

    fx2_min = 5
    fx2_max = 10
    fx2_step = 5

    verbose = False
    for model_name in model_names[0:1]:
        df = pd.read_csv(data_folder + model_name + '.csv')
        result_dir = './gen_data/benchmarks_results/' + model_name + '/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        net = Net(df)
        # for hxx in range(hxx_min, hxx_max, hxx_step):
        #     for wxx in range(wxx_min, wxx_max, wxx_step):
        #         for cxx in range(cxx_min, cxx_max, cxx_step):
        #             for fx in range(fx_min, fx_max, fx_step):
        #                 for fx2 in range(fx2_min, fx2_max, fx2_step):
        #                     # ---- Point-wise1 params
        #                     hardware_dict['HWConfig']['hxx'] = hxx
        #                     hardware_dict['HWConfig']['wxx'] = wxx
        #                     hardware_dict['HWConfig']['cxx'] = cxx
        #                     hardware_dict['HWConfig'][fx] = fx
        #                     hardware_dict['HWConfig']['padd_unit'] = wxx * 5
        #                     # ---- Depth-wise params
        #                     hardware_dict['HWConfig']['hx'] = hxx - 3 + 1
        #                     hardware_dict['HWConfig']['wx'] = wxx - 3 + 1
        #                     hardware_dict['HWConfig']['cx'] = fx
        #
        #                     # ---- Point-wise2 params
        #                     hardware_dict['HWConfig']['hxx2'] = hxx - 3 + 1
        #                     hardware_dict['HWConfig']['wxx2'] = wxx - 3 + 1
        #                     hardware_dict['HWConfig']['cxx2'] = fx
        #                     hardware_dict['HWConfig']['fx2'] = fx2
        #                     hardware_dict['HWConfig']['padd_unit2'] = (wxx - 3 + 1) * 5
        select_schedule(net, model_name, result_dir, verbose, schedule_value, hardware_dict)
        # print("hxx ={}, wxx= {}, cxx= {}, fx= {}, fx2= {}".format(hxx, wxx, cxx, fx, fx2))

        print(datetime.datetime.now())

