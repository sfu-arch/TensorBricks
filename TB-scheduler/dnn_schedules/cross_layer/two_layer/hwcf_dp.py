from dnn_schedules.schedule import Schedule
from dnn_schedules.cross_layer.cross_layer_utils import second_layer_dataflow

from attrdict import AttrDict

from dnn_schedules.per_layer.hwfc_schedule import conv2d_pw as hwfc_conv2d_pw
from dnn_schedules.per_layer.hwcf_schedule2 import conv2d_pw as hwcf_conv2d_pw
from dnn_schedules.per_layer.fchw_schedule import conv2d_pw as fchw_conv2d_pw
from dnn_schedules.per_layer.cfhw_schedule import conv2d_pw as cfhw_conv2d_pw

from dnn_schedules.cross_layer.cross_layer_utils import second_layer_dataflow, init_dp_stats
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw_block
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw  as hwc_conv2d_dw
# from dnn_schedules.per_layer.hwcf_schedule import conv_conv
# from dnn_schedules.per_layer.hwcf_schedule import conv2d_pw_block as hwcf_pw_block



class HWCFScheduleDP(Schedule):

    def __init__(self,hw_type ,second_pw_dataflow, net, model_name, result_dir, verbose,
                 hardware_yaml=None, hardware_dict=None):
        super().__init__(hw_type,net, model_name, result_dir, verbose, hardware_yaml, hardware_dict)
        self.second_pw_dataflow = second_pw_dataflow
        assert self.second_pw_dataflow in second_layer_dataflow, 'dataflow not present for last layer'
        self.conv2d_dw = hwc_conv2d_dw
    def __str__(self):
        return 'hwc_{}_schedule_dp_{}'.format(self.second_pw_dataflow, self.hw_type)

    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0
        while idx < len(items):
            current_layer = items[idx][1]
            if idx + 1 < len(items):
                next_layer = items[idx+1][1]

                if current_layer.attr_type == 'DW' and next_layer.attr_type == 'PW':
                    self.onchip_mem.clear()
                    depth_separable_conv(self, items[idx][1], items[idx+1][1])
                    self.layer_names.append(current_layer.name)
                    idx += 2
                    continue

                # # P-D, P-P, P-3d, 3d-P
                # elif current_layer.type == 'Conv2d' and next_layer.type == 'Conv2d':
                #
                #     self.conv_conv(items[idx][1], items[idx+1][1])
                #     idx += 2
                #     continue

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


def depth_separable_conv(cls, first_layer, second_layer):
    first_layer_hw_params, second_layer_hw_params = init_dp_stats(cls, first_layer, second_layer)

    # -- schedule loop --
    batch_cycles_1 = {}
    batch_cycles_2 = {}

    time_idx_1 = 0
    time_idx_2 = 0

    start_hout_idx = 0
    for hin in range(0, first_layer.Hin, first_layer_hw_params.hx):
        assert (first_layer_hw_params.hx - first_layer.Kx + 1 >= 0), \
            'Increase value of hx, hx ({}) - layer_attr.Kx ({}) + 1 <0'.format(
                first_layer_hw_params.hx, first_layer.Kx)

        # Adjust hin indices which will be used from previous convolutions
        if hin != 0:
            hin = hin - first_layer.Kx + 1

        end_hin_idx = min(hin + first_layer_hw_params.hx, first_layer.Hin) - 1
        num_hin = end_hin_idx - hin + 1

        if num_hin < first_layer.Kx:
            num_h_convs = 1
        else:
            # In case of last values -- need to add padding information,
            #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
            num_h_convs = int(num_hin - first_layer.Kx / first_layer.Sx) + 1

        end_hout_idx = start_hout_idx + num_h_convs - 1
        start_wout_idx = 0
        cls.debug_message('=====')
        for win in range(0, first_layer.Win, first_layer_hw_params.wx):
            assert (first_layer_hw_params.wx - first_layer.Ky + 1 >= 0), \
                'Increase value of wx, wx ({}) - layer_attr.Ky ({}) + 1 <0'.format(
                    first_layer.wx, first_layer.Ky)

            if win != 0:
                win = win - first_layer.Ky + 1

            end_win_idx = min(win + first_layer_hw_params.wx, first_layer.Win) - 1
            num_win = end_win_idx - win + 1

            # note: # macs connections will differ for stride = 2
            if num_win < first_layer.Ky:
                num_w_convs = 1
            else:
                num_w_convs = int((num_win - first_layer.Ky) / first_layer.Sy) + 1

            end_wout_idx = start_wout_idx + num_w_convs - 1

            start_cout_idx = 0
            for cin in range(0, first_layer.Cin, first_layer_hw_params.cx):
                end_cin_idx = min(cin + first_layer_hw_params.cx, first_layer.Cin) - 1
                num_cin = end_cin_idx - cin + 1
                num_cout = num_cin * first_layer.Depth_multiplier
                end_cout_idx = start_cout_idx + num_cout - 1
                cls.debug_message('=== Layer 1: DW ===')
                cycles_1 = conv2d_dw_block(cls, first_layer, first_layer_hw_params, cin, win, hin,
                                                start_hout_idx, start_wout_idx,
                                                num_cross_layers=2,
                                                layer_position_idx=0)

                if time_idx_1 in batch_cycles_1:
                    batch_cycles_1[time_idx_1] += cycles_1
                else:
                    batch_cycles_1[time_idx_1] = cycles_1
                # --------------------------------------------------------------------------------------------------
                # -- start of pointwise 2 convolution
                # --------------------------------------------------------------------------------------------------
                time_idx_2 = time_idx_1 + 1
                cls.debug_message('=== Layer 2: PW ===')
                cls.debug_message('  --second layer (hwc) PW input_act[{}:{}][{}:{}][{}:{}]'.format(
                    start_hout_idx, end_hout_idx,
                    start_wout_idx, end_wout_idx,
                    start_cout_idx, end_cout_idx,

                ))

                # Runs in HW|C|F for all filters of Layer 2 PW

                hin_2 = start_hout_idx
                win_2 = start_wout_idx
                cin_2 = start_cout_idx
                end_hin_idx_2 = end_hout_idx
                end_win_idx_2 = end_wout_idx
                end_cin_idx_2 = end_cout_idx
                pw2_start_indices = AttrDict({'hin': hin_2, 'win': win_2, 'cin': cin_2,
                                              'hout': start_hout_idx, 'wout': start_wout_idx, 'cout': 0,
                                              'end_hin': end_hin_idx_2 + 1,
                                              'end_win': end_win_idx_2 + 1,
                                              'end_cin': end_cin_idx_2 + 1,
                                              'end_cout': second_layer.Cout
                                              })
                # cycles_2 = cls.conv2d_pw(cls, second_layer, second_layer_hw_params, pw2_start_indices,
                #                           num_cross_layers=2, layer_position_idx=1)
                eval_second_layer = '{}_conv2d_pw(cls, second_layer, second_layer_hw_params, pw2_start_indices, ' \
                                    'num_cross_layers=2, layer_position_idx=1)'.format(cls.second_pw_dataflow)
                cycles_2 = eval(eval_second_layer)
                if time_idx_2 in batch_cycles_2:
                    batch_cycles_2[time_idx_2] += cycles_2
                else:
                    batch_cycles_2[time_idx_2] = cycles_2

                start_cout_idx = end_cout_idx + 1
                time_idx_1 += 1
            # end cin
            start_wout_idx = end_wout_idx + 1
            cls.debug_message(' --- ')
        # end win
        start_hout_idx = end_hout_idx + 1
    # end hin
    cross_layer_cycles, cycles_DW1, cycles_PW2 = cls.get_global_cycles_two_layer(batch_cycles_1,
                                                                                  batch_cycles_2,
                                                                                  time_idx_2)
    # Add cross_layer_cycles to 'global_cycles'
    # Note: Since, the global cycles will be added across layers to get total cycles
    # Only add cycles to last of the cross layers
    cls.insert_max_stats('global_cycles', first_layer.layer_idx, 0)
    cls.insert_max_stats('global_cycles', second_layer.layer_idx, cross_layer_cycles)

    cls.insert_max_stats('timing_cycles', first_layer.layer_idx, cycles_DW1)
    cls.insert_max_stats('timing_cycles', second_layer.layer_idx, cycles_PW2)

    return