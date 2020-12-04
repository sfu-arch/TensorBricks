from dnn_schedules.schedule import Schedule
from dnn_schedules.cross_layer.cross_layer_utils import second_layer_dataflow, init_cc_stats, init_dc_stats
from dnn_schedules.per_layer.hwfc_schedule import conv2d_pw as hwfc_conv2d_pw
from dnn_schedules.per_layer.hwcf_schedule2 import conv2d_pw as hwcf_conv2d_pw
from dnn_schedules.per_layer.fchw_schedule import conv2d_pw as fchw_conv2d_pw
from dnn_schedules.per_layer.cfhw_schedule import conv2d_pw as cfhw_conv2d_pw
from attrdict import AttrDict
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw_block
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw  as hwc_conv2d_dw

class FCHWScheduleCC(Schedule):

    def __init__(self,hw_type ,second_pw_dataflow, net, model_name, result_dir, verbose,
                 hardware_yaml=None, hardware_dict=None):
        super().__init__(hw_type,net, model_name, result_dir, verbose, hardware_yaml, hardware_dict)
        self.second_pw_dataflow = second_pw_dataflow
        assert self.second_pw_dataflow in second_layer_dataflow, 'dataflow not present for last layer'
        self.conv2d_dw = hwc_conv2d_dw

    def __str__(self):
        return 'fchw_{}_schedule_cc_{}'.format(self.second_pw_dataflow, self.hw_type)

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
    batch_cycles_1 = {}
    batch_cycles_2 = {}

    time_idx_1 = 0
    time_idx_2 = 0


    for f in range(0, first_layer.Cout, first_layer_hw_params.fx):
        end_cout_idx, num_cout = cls.f_params_calculation(f, first_layer_hw_params, first_layer.Cout)
        cls.debug_message('=== Layer 1: PW ===')
        pw_block_start_indices_1 = AttrDict({'hin': 0, 'end_hin': first_layer.Hin,
                                             'win': 0, 'end_win': first_layer.Win,
                                             'cin': 0, 'end_cin': first_layer.Cin,
                                             'hout': 0, 'wout': 0,
                                             'cout': f, 'end_cout': end_cout_idx + 1})

        cycles_1 = hwfc_conv2d_pw(cls, first_layer,
                                  first_layer_hw_params, pw_block_start_indices_1,
                                  num_cross_layers=2,
                                  layer_position_idx=0)



        if time_idx_1 in batch_cycles_1:
            batch_cycles_1[time_idx_1] += cycles_1
        else:
            batch_cycles_1[time_idx_1] = cycles_1

        # end cin

        # --------------------------------------------------------------------------------------------------
        # -- start of pointwise 2 convolution
        # --------------------------------------------------------------------------------------------------
        time_idx_2 = time_idx_1 + 1
        cls.debug_message('  --second layer (hwc) PW input_act[{}:{}][{}:{}][{}:{}]'.format(
            0, first_layer.Hout-1,
            0, first_layer.Wout-1,
            f, end_cout_idx - 1
        ))

        cin_2 = f
        end_cin_idx_2 = end_cout_idx

        # Stage 3 PW2 - executes all filters.
        # Runs in HW|C|F for all filters of Layer 3
        cls.debug_message('=== Layer 2: PW ===')
        pw2_start_indices = AttrDict({'hin': 0, 'win': 0, 'cin': cin_2,
                                      'hout': 0, 'wout': 0, 'cout': 0,
                                      'end_hin': first_layer.Hout,
                                      'end_win': first_layer.Wout,
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

    # end f

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

# This one is for Tangram; Pipeline DW (KCXY) +PW (CKXY; currently supports any dataflow)
# TODO: create a separate class for Tangram and import these functions.
def dw_conv(cls, first_layer, second_layer):
    first_layer_hw_params, second_layer_hw_params = init_dc_stats(cls, first_layer, second_layer)

    # -- schedule loop --
    batch_cycles_1 = {}
    batch_cycles_2 = {}

    time_idx_1 = 0
    time_idx_2 = 0

    # Since, it is running DW on CONV hardware. fx can be anything actually,
    # but per iteration only #depth_multiplier will be used.
    for f in range(0, first_layer.Cout, first_layer_hw_params.fx):
        end_cout_idx, num_cout = cls.f_params_calculation(f, first_layer_hw_params, first_layer.Cout)
        cls.debug_message('=== Layer 1: DW ===')
        dw_block_start_indices_1 = AttrDict({'hin': 0, 'end_hin': first_layer.Hin,
                                             'win': 0, 'end_win': first_layer.Win,
                                             'cin': 0, 'end_cin': first_layer.Cin,
                                             'hout': 0, 'wout': 0,
                                             'cout': f, 'end_cout': end_cout_idx + 1})

        cycles_1 = hwc_conv2d_dw(cls, first_layer,
                                  first_layer_hw_params, dw_block_start_indices_1,
                                  num_cross_layers=2,
                                  layer_position_idx=0)



        if time_idx_1 in batch_cycles_1:
            batch_cycles_1[time_idx_1] += cycles_1
        else:
            batch_cycles_1[time_idx_1] = cycles_1

        # end cin

        # --------------------------------------------------------------------------------------------------
        # -- start of pointwise 2 convolution
        # --------------------------------------------------------------------------------------------------
        time_idx_2 = time_idx_1 + 1
        cls.debug_message('  --second layer (hwc) PW input_act[{}:{}][{}:{}][{}:{}]'.format(
            0, first_layer.Hout-1,
            0, first_layer.Wout-1,
            f, end_cout_idx - 1
        ))

        cin_2 = f
        end_cin_idx_2 = end_cout_idx

        # Stage 3 PW2 - executes all filters.
        # Runs in HW|C|F for all filters of Layer 3
        cls.debug_message('=== Layer 2: PW ===')
        pw2_start_indices = AttrDict({'hin': 0, 'win': 0, 'cin': cin_2,
                                      'hout': 0, 'wout': 0, 'cout': 0,
                                      'end_hin': first_layer.Hout,
                                      'end_win': first_layer.Wout,
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

    # end f

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


def conv_conv_old(cls, first_layer, second_layer):
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
        # Adjust hin indices which will be used from previous convolutions
        # Note: no such assumption is made for 'w' dimension
        assert (first_layer_hw_params.hxx - first_layer.Kx + 1 >= 0), \
            'Increase value of hxx, hxx ({}) - layer_attr.Kx ({}) + 1 <0'.format(
                first_layer_hw_params.hxx, first_layer.Kx)
        if hin != 0:
            hin = hin - first_layer.Kx + 1

        end_hin_idx = min(hin + first_layer_hw_params.hxx, first_layer.Hin) - 1
        num_hin = end_hin_idx - hin + 1
        # In case of last values -- need to add padding information,
        #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
        if num_hin < first_layer.Kx:
            num_h_convs = 1
        else:
            num_h_convs = int(num_hin - first_layer.Kx / first_layer.Sx) + 1

        end_hout_idx = start_hout_idx + num_h_convs - 1
        # num_hout = end_hout_idx - start_hout_idx + 1

        start_wout_idx = 0
        start_wout_idx_2 = 0

        end_wout_idx = 0
        end_wout_idx_2 = 0

        for win in range(0, first_layer.Win, first_layer_hw_params.wxx):
            assert (first_layer_hw_params.wxx - first_layer.Ky + 1 >= 0), \
                'Increase value of wxx, wxx ({}) - layer_attr.Ky ({}) + 1 <0'.format(
                    first_layer_hw_params.wxx, first_layer.Ky)

            # assert (first_layer_hw_params.mac_wxx - first_layer.Ky + 1 > 0), \
            #     'Increase value of mac_wx, mac_wx ({}) - layer_attr.Ky ({}) + 1 <0'.format(first_layer.mac_wxx,
            #                                                                                first_layer.Ky)

            if win != 0:
                win = win - first_layer.Ky + 1

            end_win_idx = min(win + first_layer_hw_params.wxx, first_layer.Win) - 1
            num_win = end_win_idx - win + 1

            if num_win < first_layer.Ky:
                num_w_convs = 1
            else:
                # note: # macs connections will differ for stride = 2
                num_w_convs = int((num_win - first_layer.Ky) / first_layer.Sy) + 1

            end_wout_idx = start_wout_idx + num_w_convs - 1

            for f in range(0, first_layer.Cout, first_layer_hw_params.fx):
                # Note: P1 in PDP is HW|F|C i.e F -> will run Fx idx only
                # However, for P2 in PDP init_start_cout_idx=0, init_end_cout_idx= layer_attr.Cout
                # making it HW|C|F
                end_cout_idx = min(f + first_layer_hw_params.fx, first_layer.Cout) - 1
                num_cout = end_cout_idx - f + 1
                for cin in range(0, first_layer.Cin, first_layer_hw_params.cxx):
                    end_cin_idx = min(cin + first_layer_hw_params.cxx, first_layer.Cin) - 1
                    cls.debug_message('=== Layer 1: PW ===')
                    pw_block_start_indices_1 = AttrDict({'orig_hin': hin, 'end_hin': end_hin_idx + 1,
                                                         'orig_win': win, 'end_win': end_win_idx + 1,
                                                         'orig_cin': cin, 'end_cin': end_cin_idx + 1,
                                                         'hout': start_hout_idx, 'wout': start_wout_idx,
                                                         'cout': f, 'cout_end': end_cout_idx + 1})

                    cycles_1 = cls.conv2d_pw_block(pw_block_start_indices_1, first_layer,
                                                    first_layer_hw_params,
                                                    num_cross_layers=2,
                                                    layer_position_idx=0)
                    if time_idx_1 in batch_cycles_1:
                        batch_cycles_1[time_idx_1] += cycles_1
                    else:
                        batch_cycles_1[time_idx_1] = cycles_1

                # end cin

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

                cycles_2 = cls.conv2d_pw(second_layer, second_layer_hw_params, pw2_start_indices,
                                          num_cross_layers=2, layer_position_idx=1)
                if time_idx_2 in batch_cycles_2:
                    batch_cycles_2[time_idx_2] += cycles_2
                else:
                    batch_cycles_2[time_idx_2] = cycles_2

                time_idx_1 += 1

            # end f

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
