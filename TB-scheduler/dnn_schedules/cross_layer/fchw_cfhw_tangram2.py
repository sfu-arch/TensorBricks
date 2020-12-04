from dnn_schedules.per_layer.hwcf_schedule import HWCFSchedule
# from dnn_schedules.cfhw.cfhw_eyeriss import run_tangram
from attrdict import AttrDict


class FCHWScheduleCFHWTangram2(HWCFSchedule):
    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'fchw_cfhw_schedule'

    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0
        while idx < len(items):
            current_layer = items[idx][1]
            if idx + 1 < len(items):
                next_layer = items[idx+1][1]

                if current_layer.type == 'Conv2d' and next_layer.type == 'Conv2d':
                # if (current_layer.attr_type == 'DW' or current_layer.attr_type == 'PW') \
                #         and (next_layer.attr_type == 'DW' or next_layer.attr_type == 'PW') :
                    self.conv_conv(items[idx][1], items[idx+1][1])
                    idx += 2
                    continue

            if current_layer.type == 'Conv2d':
            # if (current_layer.attr_type == 'DW' or current_layer.attr_type == 'PW'):
                per_layer_hw_params = self.load_hw_params_conv(True, False, config=3)
                per_layer_hw_params['hxx'] = current_layer.Hin
                per_layer_hw_params['wxx'] = current_layer.Win
                self.conv2d_pw(current_layer, per_layer_hw_params)
                self.layer_names.append(current_layer.name)
                self.debug_message('===================================================')
                self.debug_message(' LAYER NAME: {} LAYER IDX: {}'.format(current_layer.name, current_layer.layer_idx))

            idx += 1


    def conv_conv(self, first_layer, second_layer):

        first_layer_hw_params = self.load_hw_params_conv(True, False, config=3)
        first_layer_hw_params['hxx'] = first_layer.Hin
        first_layer_hw_params['wxx'] = first_layer.Win
        # --- Pointwise 1 stats
        self.debug_message('cin= {} cout= {}'.format(first_layer.Cin, first_layer.Cout))
        self.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
        first_num_macs_w_units = first_layer_hw_params.mac_wxx * (first_layer_hw_params.mac_wxx_type ** 2)
        first_layer_mac_units = first_layer_hw_params.mac_cxx * first_num_macs_w_units * first_layer_hw_params.mac_fx
        first_layer_padd_units = first_layer_hw_params.mac_wxx * (first_layer_hw_params.mac_wxx_type **2) * first_layer_hw_params.mac_fx
        self.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
        self.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)

        # --- Pointwise 2 stats
        second_layer_hw_params = self.load_hw_params_conv(False, False, config=3)
        second_layer_hw_params['hxx'] = second_layer.Hin
        second_layer_hw_params['wxx'] = second_layer.Win
        self.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
        second_num_macs_w_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_wxx_type
        second_layer_mac_units = second_layer_hw_params.mac_cxx * second_num_macs_w_units * second_layer_hw_params.mac_fx
        self.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)
        second_layer_padd_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_fx
        self.insert_max_stats('padd_units_available', second_layer.layer_idx, second_layer_padd_units)

        # adding mac units
        total_mac_units = first_layer_mac_units + second_layer_mac_units
        self.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
        self.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)

        # print('total mac units: {} 1: {} 3: {}'.format(total_mac_units, first_layer_mac_units, second_layer_mac_units))
        self.insert_max_stats('total_padd_units', first_layer.layer_idx, first_layer_padd_units)
        self.insert_max_stats('total_padd_units', second_layer.layer_idx, second_layer_padd_units)

        # -- schedule loop --
        if first_layer.attr_type == 'DW':
            start_f = 0
            end_f = 1
            f_tile = first_layer_hw_params.fx
        else:
            start_f = 0
            end_f = first_layer.Cout
            f_tile = first_layer_hw_params.fx

        batch_cycles_1 = {}
        batch_cycles_2 = {}
        time_idx_1 = 0
        time_idx_2 = 0
        for f in range(start_f, end_f, f_tile):

            # Note: layer1 in F|C i.e F -> will run Fx idx only
            # However, if 'cout'=0 'cout_end'= layer_attr.Cout, it becomes C|F again
            end_cout_idx = min(f + first_layer_hw_params.fx, first_layer.Cout) - 1
            # num_cout = end_cout_idx - f + 1
            for cin in range(0, first_layer.Cin, first_layer_hw_params.cxx):
                end_cin_idx = min(cin + first_layer_hw_params.cxx, first_layer.Cin) - 1
                self.debug_message('=== Layer 1: PW/DW/CONV ===')
                pw_block_start_indices_1 = AttrDict({'orig_hin': 0, 'end_hin': first_layer.Hin,
                                                     'orig_win': 0, 'end_win': first_layer.Win,
                                                     'orig_cin': cin, 'end_cin': end_cin_idx + 1,
                                                     'hout': 0, 'wout': 0,
                                                     'cout': f, 'cout_end': end_cout_idx + 1})
                if first_layer.attr_type == 'DW':
                    pw_block_start_indices_1['cout'] = cin
                    pw_block_start_indices_1['cout_end'] = cin + first_layer.Depth_multiplier


                cycles_1 = self.conv2d_pw_block(pw_block_start_indices_1, first_layer,
                                     first_layer_hw_params,
                                     num_cross_layers=3,
                                     layer_position_idx=0)
                if time_idx_1 in batch_cycles_1:
                    batch_cycles_1[time_idx_1] += cycles_1
                else:
                    batch_cycles_1[time_idx_1] = cycles_1
            # end cin
            time_idx_2 = time_idx_1+1

            # --------------------------------------------------------------------------------------------------
            # -- start of CONV/PW/DW 2 convolution
            # --------------------------------------------------------------------------------------------------
            cin_start_2 = f
            cin_end_2_idx = end_cout_idx
            self.debug_message('  --second layer (hwc) input_act[{}:{}][{}:{}][{}:{}]'.format(
                0, second_layer.Hin-1,
                0, second_layer.Win-1,
                cin_start_2, cin_end_2_idx
            ))

            # Stage 3 PW2 - executes all filters.
            # Runs in C|F for all filters of Layer 2
            self.debug_message('=== Layer 3: PW ===')
            pw2_start_indices = AttrDict({'hin': 0, 'win': 0, 'cin': cin_start_2,
                                          'hout': 0, 'wout': 0, 'cout': 0,
                                          'end_hin': second_layer.Hin,
                                          'end_win': second_layer.Win,
                                          'end_cin': cin_end_2_idx+1,
                                          'end_cout': second_layer.Cout
                                          })

            cycles_2 = self.conv2d_pw(second_layer, second_layer_hw_params, pw2_start_indices,
                           num_cross_layers=3, layer_position_idx=2)

            if time_idx_2 in batch_cycles_2:
                batch_cycles_2[time_idx_2] += cycles_2
            else:
                batch_cycles_2[time_idx_2] = cycles_2

            time_idx_1 += 1


        # end f
        assert time_idx_1 == time_idx_2, 'For two layers both times should match'
        cross_layer_cycles, cycles_layer1, cycles_layer2 = self.get_global_cycles_two_layer(batch_cycles_1, batch_cycles_2, time_idx_2)

        # Add cross_layer_cycles to 'global_cycles'
        # Note: Since, the global cycles will be added across layers to get total cycles
        # Only add cycles to last of the cross layers
        self.insert_max_stats('global_cycles', first_layer.layer_idx, 0)
        self.insert_max_stats('global_cycles', second_layer.layer_idx, cross_layer_cycles)

        self.insert_max_stats('timing_cycles', first_layer.layer_idx, cycles_layer1)
        self.insert_max_stats('timing_cycles', second_layer.layer_idx, cycles_layer2)
        return

