import pandas as pd
from load_model import Net
import pathlib

#-- Per LAYER Schedules
from dnn_schedules.per_layer.cfhw_schedule import CFHWSchedule
from dnn_schedules.per_layer.hwcf_schedule import HWCFSchedule
from dnn_schedules.per_layer.hwcf_schedule2 import HWCFSchedule2
from dnn_schedules.per_layer.fchw_schedule import FCHWSchedule
from dnn_schedules.per_layer.hwfc_schedule import HWFCSchedule

#-- CROSS LAYER PDP Schedules
from dnn_schedules.cross_layer.pdp_pipeline.fchw_pdp import FCHW_SchedulePDP
from dnn_schedules.cross_layer.pdp_pipeline.hwfc_pdp import HWFC_SchedulePDP
from dnn_schedules.cross_layer.pdp_pipeline.hwcf_pdp import HWCF_SchedulePDP
from dnn_schedules.cross_layer.pdp_pipeline.cfhw_pdp import CFHW_SchedulePDP

#-- TWO LAYER DP Schedules
from dnn_schedules.cross_layer.two_layer.hwcf_dp import HWCFScheduleDP

#-- TWO LAYER CC Schedules
from dnn_schedules.cross_layer.two_layer.hwfc_cc import HWFCScheduleCC
from dnn_schedules.cross_layer.two_layer.hwcf_cc import HWCFScheduleCC
from dnn_schedules.cross_layer.two_layer.fchw_cc import FCHWScheduleCC
from dnn_schedules.cross_layer.two_layer.cfhw_cc import CFHWScheduleCC

from dnn_schedules.cross_layer.two_layer.tangram import Tangram
from dnn_schedules.cross_layer.two_layer.fuselayer import FuseLayer
import datetime

TwoLayerDataflow=['FuseLayer','Tangram']
second_dataflow = ['hwfc','hwcf','cfhw','fchw']

if __name__ == '__main__':
    print(datetime.datetime.now())

    model_names = ['DepthSeparable','Depthwise','pdp','mobilenet']
    data_folder='./raw_data/test_data/'
    # hardware_yaml = "params/systolic_config.yaml"
    # hardware_yaml = "params/tangram_config.yaml"
    hardware_yaml = "params/fuse_config.yaml"
    hardware_dict = None

    verbose = True
    for model_name in model_names[0:1]:
        df = pd.read_csv(data_folder + model_name + '.csv')
        net = Net(df)
        result_dir = './generated/test_data_results/'
        pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)

        # for dataflow in SingleLayerDataflow:
        #     # 'cf_cfhw' is inner dataflow and is fixed.
        #     str_schedule = '{}(\'cf_cfhw\', net, model_name, result_dir, verbose, ' \
        #                    'hardware_yaml=hardware_yaml, hardware_dict=hardware_dict)'.format(dataflow)
        #     schedule = eval(str_schedule)
        #     print(schedule)
        #     schedule.run_model()
        #     schedule.print_stats()

        for dataflow in TwoLayerDataflow[0:1]:
            # 'cf_cfhw' is inner dataflow and is fixed.
            # 'cfhw' can be replaced with any value from second_dataflow
            str_schedule = '{}(\'tangram\', \'cfhw\', net, model_name, result_dir, verbose, ' \
                           'hardware_yaml=hardware_yaml, hardware_dict=hardware_dict)'.format(dataflow)

            schedule = eval(str_schedule)
            print(schedule)
            schedule.run_model()
            schedule.print_stats()

        # for dataflow in ThreeLayerDataflow:
        #     # 'cf_cfhw' is inner dataflow and is fixed.
        #     # 'hwfc' can be replaced with any value from second_dataflow
        #     str_schedule = '{}(\'cf_cfhw\', \'hwfc\', net, model_name, result_dir, verbose, ' \
        #                    'hardware_yaml=hardware_yaml, hardware_dict=hardware_dict)'.format(dataflow)

            schedule = eval(str_schedule)
            print(schedule)
            schedule.run_model()
            schedule.print_stats()

    print(datetime.datetime.now())

