from dnn_schedules.schedule import Schedule
from dnn_schedules.cross_layer.cross_layer_utils import second_layer_dataflow, init_cc_stats
from dnn_schedules.per_layer.hwfc_schedule import conv2d_pw as hwfc_conv2d_pw
from dnn_schedules.per_layer.hwcf_schedule2 import conv2d_pw as hwcf_conv2d_pw
from dnn_schedules.per_layer.fchw_schedule import conv2d_pw as fchw_conv2d_pw
from dnn_schedules.per_layer.cfhw_schedule import conv2d_pw as cfhw_conv2d_pw
from attrdict import AttrDict
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw  as hwc_conv2d_dw

from  dnn_schedules.cross_layer.two_layer.fchw_cc import conv_conv, dw_conv

class Tangram(Schedule):

    def __init__(self,hw_type ,second_pw_dataflow, net, model_name, result_dir, verbose,
                 hardware_yaml=None, hardware_dict=None):
        assert hw_type == 'tangram', 'Wrong hardware type {} != tangram'.format(hw_type)
        super().__init__(hw_type,net, model_name, result_dir, verbose, hardware_yaml, hardware_dict)
        self.second_pw_dataflow = second_pw_dataflow
        assert self.second_pw_dataflow in second_layer_dataflow, 'dataflow not present for last layer'
        self.conv2d_dw = hwc_conv2d_dw

    def __str__(self):
        return 'tangram_fchw_{}_schedule_cc_{}'.format(self.second_pw_dataflow, self.hw_type)

    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0
        while idx < len(items):
            current_layer = items[idx][1]
            if idx + 1 < len(items):
                next_layer = items[idx+1][1]

                if (current_layer.attr_type == 'PW' and next_layer.attr_type == 'PW') or \
                    (current_layer.attr_type == '3d' and next_layer.attr_type == 'PW') or \
                    (current_layer.attr_type == 'PW' and next_layer.attr_type == '3d') or \
                    (current_layer.attr_type == '3d' and next_layer.attr_type == '3d'):
                    self.onchip_mem.clear()
                    conv_conv(self, items[idx][1], items[idx+1][1])
                    self.layer_names.append(current_layer.name)
                    idx += 2
                    continue

                if (current_layer.attr_type == 'DW' and next_layer.attr_type == 'PW') or \
                    (current_layer.attr_type == 'DW' and next_layer.attr_type == '3d'):
                    self.onchip_mem.clear()
                    dw_conv(self, items[idx][1], items[idx+1][1])
                    self.layer_names.append(current_layer.name)
                    idx += 2
                    continue


            if current_layer.attr_type == 'DW':
                self.onchip_mem.clear()
                self.layer_names.append(current_layer.name)
                dw_layer_hw_params = self.load_hw_params_depthwise()
                self.conv2d_dw(self, current_layer, dw_layer_hw_params)
            if current_layer.attr_type == 'PW':
                self.onchip_mem.clear()
                self.layer_names.append(current_layer.name)
                pw_layer_hw_params = self.load_hw_params_pointwise(True, True)
                # self.conv2d_pw(current_layer, pw_layer_hw_params)
                eval_layer = '{}_conv2d_pw(self, current_layer, pw_layer_hw_params)'.format(self.second_pw_dataflow)
                _ = eval(eval_layer)
            if current_layer.attr_type == '3d':
                self.onchip_mem.clear()
                per_layer_hw_params = self.load_hw_params_conv(True, True)
                # self.conv2d_pw(current_layer, per_layer_hw_params)
                eval_layer = '{}_conv2d_pw(self, current_layer, per_layer_hw_params)'.format(self.second_pw_dataflow)
                _ = eval(eval_layer)
                self.layer_names.append(current_layer.name)

            idx += 1
        return
