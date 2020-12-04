from dnn_schedules.per_layer.hwcf_schedule import HWCFSchedule
from attrdict import AttrDict


class HWCFSchedulePWDW(HWCFSchedule):

    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'hwcf_schedule_pw_dw'

    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0
        while idx < len(items):
            current_layer = items[idx][1]
            if idx + 1 < len(items):
                next_layer = items[idx+1][1]

                if current_layer.attr_type == 'PW' and next_layer.attr_type == 'DW':
                    self.pw_dw_conv(items[idx][1], items[idx+1][1])
                    idx += 2
                    continue

            if current_layer.attr_type == 'DW':
                dw_layer_hw_params = self.load_hw_params_depthwise()
                self.conv2d_dw(current_layer, dw_layer_hw_params)
            if current_layer.attr_type == 'PW':
                pw_layer_hw_params = self.load_hw_params_pointwise(True,True)
                self.conv2d_pw(current_layer, pw_layer_hw_params)


            if current_layer.attr_type == '3d':
                self.onchip_mem.clear()
                per_layer_hw_params = self.load_hw_params_conv(True, True)
                self.conv2d_pw(current_layer, per_layer_hw_params)
                self.layer_names.append(current_layer.name)

            idx += 1
        return

    def pw_dw_conv(self, first_layer, second_layer):
        first_layer_hw_params = self.load_hw_params_pointwise(True, False)
        # --- Pointwise 1 stats
        self.debug_message('cin= {} cout= {}'.format(first_layer.Cin, first_layer.Cout))
        self.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
        first_num_macs_w_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_wxx_type
        first_layer_mac_units = first_layer_hw_params.mac_cxx * first_num_macs_w_units * first_layer_hw_params.mac_fx
        first_layer_padd_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_fx
        self.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
        self.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)


        # -- Depthwise stats --
        second_layer_hw_params = self.load_hw_params_depthwise()
        self.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
        second_num_macs_w_units = second_layer_hw_params.mac_wx * second_layer_hw_params.mac_wx_type * second_layer_hw_params.mac_wx_type
        second_layer_mac_units = second_layer_hw_params.mac_cx * second_num_macs_w_units
        self.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)


        # adding mac units
        total_mac_units = first_layer_mac_units + second_layer_mac_units
        self.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
        self.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)


        # print('total mac units: {} 1: {} 2: {} 3: {}'.format(total_mac_units, first_layer_mac_units,
        #                                                      second_layer_mac_units, third_layer_mac_units))
        self.insert_max_stats('total_padd_units', first_layer.layer_idx, first_layer_padd_units)

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
                assert(first_layer_hw_params.wxx - first_layer.Ky +1 >=0), \
                    'Increase value of wxx, wxx ({}) - layer_attr.Ky ({}) + 1 <0'.format(
                        first_layer_hw_params.wxx, first_layer.Ky)

                assert( first_layer_hw_params.mac_wxx - first_layer.Ky +1 >0), \
                    'Increase value of mac_wx, mac_wx ({}) - layer_attr.Ky ({}) + 1 <0'.format(first_layer.mac_wxx, first_layer.Ky)

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

                for f in range(0,first_layer.Cout, first_layer_hw_params.fx):
                    # Note: P1 in PDP is HW|F|C i.e F -> will run Fx idx only
                    # However, for P2 in PDP init_start_cout_idx=0, init_end_cout_idx= layer_attr.Cout
                    # making it HW|C|F
                    end_cout_idx= min(f + first_layer_hw_params.fx, first_layer.Cout) - 1
                    num_cout = end_cout_idx -f + 1
                    for cin in range(0, first_layer.Cin, first_layer_hw_params.cxx):
                        end_cin_idx = min(cin + first_layer_hw_params.cxx, first_layer.Cin)-1
                        self.debug_message('=== Layer 1: PW ===')
                        pw_block_start_indices_1 = AttrDict({'orig_hin': hin, 'end_hin': end_hin_idx+1,
                                                           'orig_win': win, 'end_win': end_win_idx+1,
                                                           'orig_cin': cin, 'end_cin': end_cin_idx+1,
                                                           'hout': start_hout_idx, 'wout': start_wout_idx,
                                                           'cout': f, 'cout_end': end_cout_idx+1})

                        cycles_1  = self.conv2d_pw_block(pw_block_start_indices_1, first_layer,
                                             first_layer_hw_params,
                                             num_cross_layers=3,
                                             layer_position_idx=0)
                        if time_idx_1 in batch_cycles_1:
                            batch_cycles_1[time_idx_1] += cycles_1
                        else:
                            batch_cycles_1[time_idx_1] = cycles_1

                    # end cin
                    # --------------------------------------------------------------------------------------------------
                    # -- start of depthwise convolution
                    # --------------------------------------------------------------------------------------------------
                    time_idx_2 = time_idx_1 + 1
                    self.debug_message('  --second layer (hwc) DW input_act[{}:{}][{}:{}][{}:{}]'.format(
                        start_hout_idx, end_hout_idx,
                        start_wout_idx, end_wout_idx,
                        f, end_cout_idx
                    ))

                    block_cin_2 = num_cout
                    block_win_2 = num_w_convs
                    block_hin_2 = num_h_convs

                    block_cout_2 = block_cin_2*second_layer.Depth_multiplier
                    if block_win_2 < second_layer.Ky:
                        num_w_convs_2 = 1
                    else:
                        num_w_convs_2 = int((block_win_2 - second_layer.Ky) / second_layer.Sy) + 1

                    if block_hin_2 < second_layer.Kx:
                        num_h_convs_2 = 1
                    else:
                        num_h_convs_2 = int(block_hin_2 - second_layer.Kx / second_layer.Sx) + 1

                    # hin2, win2, cin2, hout2, wout2, cout2
                    hin_2 = start_hout_idx
                    win_2 = start_wout_idx
                    cin_2 = f
                    end_hout_idx_2 = start_hout_idx_2 + num_h_convs_2 - 1
                    end_wout_idx_2 = start_wout_idx_2 + num_w_convs_2 - 1
                    end_cin_idx_2 = cin_2 + num_cout - 1
                    cout_2= f
                    end_cout_idx_2 = cout_2 + block_cout_2 - 1


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
                    dw_start_indices_2 = AttrDict({'hin': hin, 'win': win, 'cin': f,
                                                   'hout': start_hout_idx, 'wout': start_wout_idx})
                    cycles_2 = self.conv2d_dw(second_partial_layer, second_layer_hw_params,dw_start_indices_2,
                                              num_cross_layers=3, layer_position_idx=1)

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
        cross_layer_cycles,  cycles_PW1, cycles_DW2 = self.get_global_cycles_two_layer(batch_cycles_1,
                                                                batch_cycles_2, time_idx_2)
        # Add cross_layer_cycles to 'global_cycles'
        # Note: Since, the global cycles will be added across layers to get total cycles
        # Only add cycles to last of the cross layers
        self.insert_max_stats('global_cycles', first_layer.layer_idx, 0)
        self.insert_max_stats('global_cycles', second_layer.layer_idx, cross_layer_cycles)


        self.insert_max_stats('timing_cycles', first_layer.layer_idx, cycles_PW1)
        self.insert_max_stats('timing_cycles', second_layer.layer_idx, cycles_DW2)




        return