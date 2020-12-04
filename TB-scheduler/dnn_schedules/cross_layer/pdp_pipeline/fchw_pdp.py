from attrdict import AttrDict
from dnn_schedules.schedule import Schedule
from dnn_schedules.per_layer.hwfc_schedule import conv2d_pw as hwfc_conv2d_pw
from dnn_schedules.per_layer.hwcf_schedule2 import conv2d_pw as hwcf_conv2d_pw
from dnn_schedules.per_layer.fchw_schedule import conv2d_pw as fchw_conv2d_pw
from dnn_schedules.per_layer.cfhw_schedule import conv2d_pw as cfhw_conv2d_pw
from dnn_schedules.cross_layer.cross_layer_utils import second_layer_dataflow, init_pdp_stats
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw  as hwc_conv2d_dw

# -- OLD stuff --
# from dnn_schedules.per_layer.hwcf_schedule import conv_conv

class FCHW_SchedulePDP(Schedule):

    def __init__(self,hw_type ,second_pw_dataflow, net, model_name, result_dir, verbose,
                 hardware_yaml=None, hardware_dict=None):
        super().__init__(hw_type, net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)
        self.second_pw_dataflow = second_pw_dataflow
        assert self.second_pw_dataflow in second_layer_dataflow,'dataflow not present for last layer'
        self.conv2d_dw = hwc_conv2d_dw
        self.conv2d_pw = fchw_conv2d_pw


    def __str__(self):
        return 'fchw_hwc_{}_schedule_pdp_{}'.format(self.second_pw_dataflow, self.hw_type)

    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0

        while idx < len(items):
            current_layer = items[idx][1]

            if idx + 2 < len(items):
                next_layer = items[idx+1][1]
                next_next_layer = items[idx + 2][1]


                if current_layer.attr_type == 'PW' and next_layer.attr_type == 'DW' \
                        and next_next_layer.attr_type == 'PW':
                    self.onchip_mem.clear()
                    self.layer_names.append(current_layer.name)
                    self.pdp_conv(current_layer, next_layer, next_next_layer)
                    idx += 3
                    continue


            # -------------------------------------------
            # if idx + 1 < len(items):
            #     next_layer = items[idx+1][1]
            #
            #     if current_layer.attr_type == 'DW' and next_layer.attr_type == 'PW':
            #         self.onchip_mem.clear()
            #         depth_separable_conv(self, items[idx][1], items[idx+1][1])
            #         self.layer_names.append(current_layer.name)
            #         idx += 2
            #         continue
                # # P-D, P-P, P-3d, 3d-P
                # elif current_layer.type == 'Conv2d' and next_layer.type == 'Conv2d':
                #
                #     conv_conv(self, items[idx][1], items[idx+1][1])
                #     idx += 2
                #     continue

#-------------------------------------------
            if current_layer.attr_type == 'DW':
                self.onchip_mem.clear()
                self.layer_names.append(current_layer.name)
                dw_layer_hw_params = self.load_hw_params_depthwise()
                hwc_conv2d_dw(self, current_layer, dw_layer_hw_params)
            if current_layer.attr_type == 'PW':
                self.onchip_mem.clear()
                self.layer_names.append(current_layer.name)
                pw_layer_hw_params = self.load_hw_params_pointwise(True, True)
                fchw_conv2d_pw(self, current_layer, pw_layer_hw_params)

            if current_layer.attr_type == '3d':
                self.onchip_mem.clear()
                # self.stats['orig_idx'][layer_attr.layer_idx] = orig_idx - 1
                per_layer_hw_params = self.load_hw_params_conv(True, True)
                fchw_conv2d_pw(self, current_layer, per_layer_hw_params)
                self.layer_names.append(current_layer.name)

            idx += 1
        return

    def pdp_conv(self, first_layer, second_layer, third_layer):
        first_layer_hw_params, second_layer_hw_params, \
        third_layer_hw_params = init_pdp_stats(self, first_layer, second_layer, third_layer)

        # -- schedule loop --


        batch_cycles_1 = {}
        batch_cycles_2 = {}
        batch_cycles_3 = {}
        time_idx_1 = 0
        time_idx_2 = 0
        time_idx_3 = 0


        for f in range(0, first_layer.Cout, first_layer_hw_params.fx):
            # Note: layer1 in F|C i.e F -> will run Fx idx only
            # However, if 'cout'=0 'cout_end'= layer_attr.Cout, it becomes C|F again
            end_cout_idx = min(f + first_layer_hw_params.fx, first_layer.Cout) - 1
            num_cout = end_cout_idx - f + 1
            for cin in range(0, first_layer.Cin, first_layer_hw_params.cxx):
                end_cin_idx = min(cin + first_layer_hw_params.cxx, first_layer.Cin) - 1

                self.debug_message('=== Layer 1: PW ===')
                pw_block_start_indices_1 = AttrDict({'hin': 0, 'end_hin': first_layer.Hin,
                                                     'win': 0, 'end_win':  first_layer.Win,
                                                     'cin': cin, 'end_cin': end_cin_idx + 1,
                                                     'hout': 0, 'wout': 0,
                                                     'cout': f, 'end_cout': end_cout_idx + 1})

                cycles_1 = fchw_conv2d_pw(self, first_layer,
                                     first_layer_hw_params, pw_block_start_indices_1,
                                     num_cross_layers=3,
                                     layer_position_idx=0)
                if time_idx_1 in batch_cycles_1:
                    batch_cycles_1[time_idx_1] += cycles_1
                else:
                    batch_cycles_1[time_idx_1] = cycles_1

            # end cin

            # At this point Layer 1 input: H*W*C, Fx --> Layer 2: input: H*W*Fx --> Layer 2 output: H*W*(Fx*Depth_multiplier)
            # --------------------------------------------------------------------------------------------------
            # -- start of depthwise convolution
            # --------------------------------------------------------------------------------------------------
            time_idx_2 = time_idx_1 + 1
            self.debug_message('  --second layer (hwc) DW input_act[{}:{}][{}:{}][{}:{}]'.format(
                0, first_layer.Hout-1,
                0, first_layer.Wout-1,
                f, end_cout_idx
            ))


            # -- H Block params starts --
            # Adjust hin indices which will be used from previous convolutions
            # Note: no such assumption is made for 'w' dimension
            assert (first_layer_hw_params.hxx - first_layer.Kx + 1 >= 0), \
                'Increase value of hxx, hxx ({}) - layer_attr.Kx ({}) + 1 <0'.format(
                    first_layer_hw_params.hxx, first_layer.Kx)

            end_hin_idx = first_layer.Hin - 1
            num_hin = end_hin_idx + 1
            end_hout_idx = first_layer.Hout - 1
            num_hout = end_hout_idx + 1


            # -- W Block params starts --
            assert (first_layer_hw_params.wxx - first_layer.Ky + 1 >= 0), \
                'Increase value of wxx, wxx ({}) - layer_attr.Ky ({}) + 1 <0'.format(
                    first_layer_hw_params.wxx, first_layer.Ky)

            assert (first_layer_hw_params.mac_wxx - first_layer.Ky + 1 > 0), \
                'Increase value of mac_wx, mac_wx ({}) - layer_attr.Ky ({}) + 1 <0'.format(first_layer.mac_wxx,
                                                                                           first_layer.Ky)
            end_win_idx = first_layer.Win - 1
            num_win = end_win_idx + 1
            end_wout_idx = first_layer.Wout - 1
            num_wout = end_wout_idx + 1
            # -- W Block params ends --

            block_cin_2 = num_cout
            block_win_2 = num_wout
            block_hin_2 = num_hout

            block_cout_2 = block_cin_2*second_layer.Depth_multiplier
            num_h_convs_2 = second_layer.Hout
            num_w_convs_2 = second_layer.Wout

            hin_2 = 0
            win_2 = 0
            cin_2 = f
            end_cin_idx_2 = cin_2 + num_cout - 1

            end_hout_idx_2 = second_layer.Hout - 1
            end_wout_idx_2 = second_layer.Wout - 1
            cout_2 = (cin_2 + 1)*second_layer.Depth_multiplier - 1
            end_cout_idx_2 = cout_2 + block_cout_2 - 1
            num_cout_2 = end_cout_idx_2 + 1

            second_partial_layer = AttrDict({'name': second_layer.name, 'layer_idx': second_layer.layer_idx,
                                             'type': second_layer.type,
                                             'Kx': second_layer.Kx, 'Ky': second_layer.Ky, 'K': second_layer.K,
                                             'Cin': block_cin_2, 'Win': block_win_2, 'Hin': block_hin_2,
                                             'Cout': block_cout_2,
                                             'Depth_multiplier': second_layer.Depth_multiplier,
                                              # 'Wout': block_wout_2, 'Hout': block_hout_2,
                                              'attr_type': second_layer.attr_type,
                                              'Sx': second_layer.Sx, 'Sy': second_layer.Sy,
                                              'Px': second_layer.Px, 'Py': second_layer.Py
                                              # 'Bias': second_layer.Bias
                                            })
            self.debug_message('=== Layer 2: DW ===')
            dw_start_indices_2 = AttrDict({'hin': 0, 'win': 0, 'cin': cin_2,
                                           'hout': 0, 'wout': 0})
            cycles_2 = hwc_conv2d_dw(self, second_partial_layer, second_layer_hw_params,dw_start_indices_2, num_cross_layers=3, layer_position_idx=1)

            if time_idx_2 in batch_cycles_2:
                batch_cycles_2[time_idx_2] += cycles_2
            else:
                batch_cycles_2[time_idx_2] = cycles_2

            # At this point Layer 1 input: H*W*C, Fx
            # Layer 2: input: H*W*Fx
            # Layer 3 input: H*W*(Fx*Depth_multiplier) --> output: H*W*third_layer.Cout
            # --------------------------------------------------------------------------------------------------
            # -- start of pointwise 2 convolution
            # --------------------------------------------------------------------------------------------------
            time_idx_3 = time_idx_2 + 1
            self.debug_message('  --third layer (hwc) PW input_act[{}:{}][{}:{}][{}:{}]'.format(
                hin_2, end_hout_idx_2,
                win_2, end_wout_idx_2,
                cin_2, end_cin_idx_2
            ))

            block_cin_3 = block_cout_2
            block_win_3 = num_w_convs_2
            block_hin_3 = num_h_convs_2
            block_cout_3 = third_layer.Cout

            num_h_convs_3 = third_layer.Hout
            num_w_convs_3 = third_layer.Wout

            # hin3, win3, cin3, hout3, wout3, cout3
            cin_3 = cout_2
            end_hin_idx_3 = block_hin_3 - 1
            end_win_idx_3 = block_win_3 - 1
            end_cin_idx_3 = end_cout_idx_2

            end_hout_idx_3 = 0 + num_h_convs_3 - 1
            end_wout_idx_3 = 0 + num_w_convs_3 - 1

            # Stage 3 PW2 - executes all filters.
            # Runs in C|F|HW for all filters of Layer 3
            # TODO: verify this. Since, orig pw_conv runs HW|C|F
            self.debug_message('=== Layer 3: PW ===')
            pw3_start_indices = AttrDict({'hin': 0, 'win': 0, 'cin': cin_3,
                'hout': 0, 'wout': 0, 'cout': 0,
                'end_hin': end_hin_idx_3 + 1,
                'end_win': end_win_idx_3 + 1,
                'end_cin':end_cin_idx_3 + 1,
                'end_cout': third_layer.Cout
                })
            # CFHWSchedule

            eval_second_layer = '{}_conv2d_pw(self, third_layer, third_layer_hw_params, pw3_start_indices, ' \
                                 'num_cross_layers=3, layer_position_idx=2)'.format(self.second_pw_dataflow)
            cycles_3 = eval(eval_second_layer)
            if time_idx_3 in batch_cycles_3:
                batch_cycles_3[time_idx_3] += cycles_3
            else:
                batch_cycles_3[time_idx_3] = cycles_3
            time_idx_1 += 1


        # end f
        cross_layer_cycles,  cycles_PW1, cycles_DW2, cycles_PW3 = self.get_global_cycles_three_layer(batch_cycles_1,
                                                                batch_cycles_2,
                                                                batch_cycles_3,
                                                                time_idx_3)
        # Add cross_layer_cycles to 'global_cycles'
        # Note: Since, the global cycles will be added across layers to get total cycles
        # Only add cycles to last of the cross layers
        self.insert_max_stats('global_cycles', first_layer.layer_idx, 0)
        self.insert_max_stats('global_cycles', second_layer.layer_idx, 0)
        self.insert_max_stats('global_cycles',third_layer.layer_idx, cross_layer_cycles)

        self.insert_max_stats('timing_cycles', first_layer.layer_idx, cycles_PW1)
        self.insert_max_stats('timing_cycles', second_layer.layer_idx, cycles_DW2)
        self.insert_max_stats('timing_cycles', third_layer.layer_idx, cycles_PW3)



        return

