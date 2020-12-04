from dnn_schedules.schedule import Schedule
from dnn_schedules.cross_layer.cross_layer_utils import second_layer_dataflow, init_cc_stats
from dnn_schedules.per_layer.hwfc_schedule import conv2d_pw as hwfc_conv2d_pw
from dnn_schedules.per_layer.hwcf_schedule2 import conv2d_pw as hwcf_conv2d_pw
from dnn_schedules.per_layer.fchw_schedule import conv2d_pw as fchw_conv2d_pw
from dnn_schedules.per_layer.cfhw_schedule import conv2d_pw as cfhw_conv2d_pw
from attrdict import AttrDict
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw_block
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw  as hwc_conv2d_dw

class HWCFScheduleCC(Schedule):

    def __init__(self,hw_type ,second_pw_dataflow, net, model_name, result_dir, verbose,
                 hardware_yaml=None, hardware_dict=None):
        super().__init__(hw_type,net, model_name, result_dir, verbose, hardware_yaml, hardware_dict)
        self.second_pw_dataflow = second_pw_dataflow
        assert self.second_pw_dataflow in second_layer_dataflow, 'dataflow not present for last layer'
        self.conv2d_dw = hwc_conv2d_dw

    def __str__(self):
        return 'hwcf_{}_schedule_cc_{}'.format(self.second_pw_dataflow, self.hw_type)

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

def conv_conv(cls, first_layer, second_layer):
    first_layer_hw_params, second_layer_hw_params = init_cc_stats(cls, first_layer, second_layer)

    # -- schedule loop --
    start_hout_idx = 0
    start_hout_idx_2 = 0
    end_hout_idx_2 = 0

    batch_cycles_1 = {}
    batch_cycles_2 = {}

    time_idx_1 = 0
    time_idx_2 = 0

    for hin in range(0, first_layer.Hin, first_layer_hw_params.hxx):

        if hin != 0:
            INIT_START_HIN_IDX= -1
        else:
            INIT_START_HIN_IDX = None
        hin, end_hin_idx, end_hout_idx, num_hin, num_h_convs, num_hout = cls.h_params_calculation(hin, first_layer, first_layer_hw_params,
                             INIT_START_HIN_IDX,  first_layer.Hin, start_hout_idx)

        start_wout_idx = 0
        start_wout_idx_2 = 0
        end_wout_idx = 0
        end_wout_idx_2 = 0

        for win in range(0, first_layer.Win, first_layer_hw_params.wxx):
            if win != 0:
                INIT_START_WIN_IDX = -1
            else:
                INIT_START_WIN_IDX = None

            win, end_wout_idx, end_win_idx, num_win, num_w_convs, num_wout = cls.w_params_calculation(win, first_layer, first_layer_hw_params,
                                 INIT_START_WIN_IDX, first_layer.Win, start_wout_idx)
            for cin in range(0, first_layer.Cin, first_layer_hw_params.cxx):
                num_cin, end_cin_idx = cls.c_params_calculation(cin, first_layer_hw_params, first_layer.Cin)
                f=0
                end_cout_idx, num_cout = cls.f_params_calculation(f, first_layer_hw_params, first_layer.Cout)

                cls.debug_message('=== Layer 1: PW ===')
                pw_block_start_indices_1 = AttrDict({'hin': hin, 'end_hin': end_hin_idx + 1,
                                                     'win': win, 'end_win': end_win_idx + 1,
                                                     'cin': cin, 'end_cin': end_cin_idx + 1,
                                                     'hout': start_hout_idx, 'wout': start_wout_idx,
                                                     'cout': 0, 'end_cout': first_layer.Cout})

                cycles_1 = hwfc_conv2d_pw(cls, first_layer,
                                          first_layer_hw_params, pw_block_start_indices_1,
                                          num_cross_layers=2,
                                          layer_position_idx=0)



                if time_idx_1 in batch_cycles_1:
                    batch_cycles_1[time_idx_1] += cycles_1
                else:
                    batch_cycles_1[time_idx_1] = cycles_1

                # end f

                # --------------------------------------------------------------------------------------------------
                # -- start of pointwise 2 convolution
                # --------------------------------------------------------------------------------------------------
                time_idx_2 = time_idx_1 + 1
                cls.debug_message('  --second layer (hwc) PW input_act[{}:{}][{}:{}][{}:{}]'.format(
                    start_hout_idx, end_hout_idx,
                    start_wout_idx, end_wout_idx,
                    f, end_cout_idx - 1
                ))

                block_cin_2 = num_cout
                block_win_2 = num_w_convs
                block_hin_2 = num_h_convs
                block_cout_2 = second_layer.Cout

                if block_win_2 < second_layer.Ky:
                    num_w_convs_2 = 1
                else:
                    num_w_convs_2 = int((block_win_2 - second_layer.Ky) / second_layer.Sy) + 1

                if block_hin_2 < second_layer.Kx:
                    num_h_convs_2 = 1
                else:
                    num_h_convs_2 = int(num_hin - first_layer.Kx / first_layer.Sx) + 1

                # hin3, win3, cin3, hout3, wout3, cout3
                hin_2 = start_hout_idx
                win_2 = start_wout_idx
                cin_2 = f
                end_hin_idx_2 = hin_2 + block_hin_2 - 1
                end_win_idx_2 = win_2 + block_win_2 - 1
                end_cin_idx_2 = end_cout_idx

                end_hout_idx_2 = start_hout_idx_2 + num_h_convs_2 - 1
                end_wout_idx_2 = start_wout_idx_2 + num_w_convs_2 - 1
                # Stage 3 PW2 - executes all filters.
                start_cout_idx_2 = 0
                end_cout_idx_2 = start_cout_idx_2 + block_cout_2 - 1

                # Runs in HW|C|F for all filters of Layer 3
                cls.debug_message('=== Layer 2: PW ===')
                pw2_start_indices = AttrDict({'hin': hin_2, 'win': win_2, 'cin': cin_2,
                                              'hout': start_hout_idx, 'wout': start_wout_idx, 'cout': 0,
                                              'end_hin': end_hin_idx_2 + 1,
                                              'end_win': end_win_idx_2 + 1,
                                              'end_cin': end_cin_idx_2 + 1,
                                              'end_cout': second_layer.Cout
                                              })

                eval_second_layer = '{}_conv2d_pw(cls, second_layer, second_layer_hw_params, pw2_start_indices, ' \
                                    'num_cross_layers=2, layer_position_idx=1)'.format(cls.second_pw_dataflow)
                cycles_2 = eval(eval_second_layer)

                if time_idx_2 in batch_cycles_2:
                    batch_cycles_2[time_idx_2] += cycles_2
                else:
                    batch_cycles_2[time_idx_2] = cycles_2

                time_idx_1 += 1

            # end cin

            start_wout_idx = end_wout_idx + 1
            # update start_wout indices for second/third layer
            start_wout_idx_2 = end_wout_idx_2 + 1
        # end w
        start_hout_idx = end_hout_idx + 1
        start_hout_idx_2 = end_hout_idx_2 + 1

    # end h
    cross_layer_cycles, cycles_PW1, cycles_DW2 = cls.get_global_cycles_two_layer(batch_cycles_1,
                                                                                  batch_cycles_2, time_idx_2)
    # Add cross_layer_cycles to 'global_cycles'
    # Note: Since, the global cycles will be added across layers to get total cycles
    # Only add cycles to last of the cross layers
    cls.insert_max_stats('global_cycles', first_layer.layer_idx, 0)
    cls.insert_max_stats('global_cycles', second_layer.layer_idx, cross_layer_cycles)

    cls.insert_max_stats('timing_cycles', first_layer.layer_idx, cycles_PW1)
    cls.insert_max_stats('timing_cycles', second_layer.layer_idx, cycles_DW2)

    return
