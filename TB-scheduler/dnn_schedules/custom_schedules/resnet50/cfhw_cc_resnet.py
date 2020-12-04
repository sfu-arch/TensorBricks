
from dnn_schedules.cross_layer.two_layer.cfhw_cc import CFHWScheduleCC as OrigCFHWScheduleCC
from dnn_schedules.cross_layer.two_layer.cfhw_cc import conv_conv

from dnn_schedules.schedule import Schedule
from dnn_schedules.cross_layer.cross_layer_utils import second_layer_dataflow, init_cc_stats, init_dc_stats
from dnn_schedules.per_layer.hwfc_schedule import conv2d_pw as hwfc_conv2d_pw
from dnn_schedules.per_layer.hwcf_schedule2 import conv2d_pw as hwcf_conv2d_pw
from dnn_schedules.per_layer.fchw_schedule import conv2d_pw as fchw_conv2d_pw
from dnn_schedules.per_layer.cfhw_schedule import conv2d_pw as cfhw_conv2d_pw
from attrdict import AttrDict
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw_block
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw  as hwc_conv2d_dw


class CFHWScheduleCC(OrigCFHWScheduleCC):

    def __init__(self,hw_type ,second_pw_dataflow, net, model_name, result_dir, verbose,
                 hardware_yaml=None, hardware_dict=None):
        super().__init__(hw_type ,second_pw_dataflow, net, model_name, result_dir,
                         verbose, hardware_yaml, hardware_dict)
        self.second_pw_dataflow = second_pw_dataflow
        assert self.second_pw_dataflow in second_layer_dataflow, 'dataflow not present for last layer'

    def __str__(self):
        return 'Resnet_cfhw_{}_schedule_cc_{}'.format(self.second_pw_dataflow, self.hw_type)

    """ Resnet-50 has the structure a->|X|->(PW,3d,PW)|- (+).  X-->Downsample->(+)-- Y. 
    You can only pipeline PW,3d or 3d PW. We chose to pipeline PW--3d 
    """
    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0
        while idx < len(items):
            current_layer = items[idx][1]
            if idx + 1 < len(items):
                next_layer = items[idx+1][1]

                if (current_layer.attr_type == 'PW' and next_layer.attr_type == '3d'):
                    self.onchip_mem.clear()
                    conv_conv(self, items[idx][1], items[idx+1][1])
                    self.layer_names.append(current_layer.name)
                    idx += 2
                    continue

            if current_layer.attr_type == 'DW':
                raise ValueError('DW not supported for CC benchmarks.')

            if current_layer.attr_type == 'PW':
                self.onchip_mem.clear()
                self.layer_names.append(current_layer.name)
                pw_layer_hw_params = self.load_hw_params_pointwise(True, True)
                eval_layer = '{}_conv2d_pw(self, current_layer, pw_layer_hw_params)'.format(self.second_pw_dataflow)
                _ = eval(eval_layer)
            if current_layer.attr_type == '3d':
                self.onchip_mem.clear()
                per_layer_hw_params = self.load_hw_params_conv(True, True)
                eval_layer = '{}_conv2d_pw(self, current_layer, per_layer_hw_params)'.format(self.second_pw_dataflow)
                _ = eval(eval_layer)
                self.layer_names.append(current_layer.name)

            idx += 1
        return