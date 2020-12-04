from dnn_schedules.per_layer.hwcf_schedule import HWCFSchedule
from attrdict import AttrDict


class HWCFScheduleFire2Layer(HWCFSchedule):

    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'hwfc_hwc_hwcf_schedule_pdp'

    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0

        while idx < len(items):
            current_layer = items[idx][1]

            if idx + 2 < len(items):
                next_layer = items[idx+1][1]
                next_next_layer = items[idx + 2][1]


                if current_layer.attr_type == 'PW' and next_layer.attr_type == 'PW' \
                        and next_next_layer.attr_type == '3d':
                    # Run PW1, 3d together in fire layer. Since, 3d take the most time.
                    self.onchip_mem.clear()
                    self.layer_names.append(current_layer.name)
                    self.conv_conv(current_layer, next_next_layer)

                    # Run pointwise layer later.
                    # We want to run it on CONV hardware. Since, more resources.
                    per_layer_hw_params = self.load_hw_params_conv(False, False, config=0)
                    self.conv2d_pw(current_layer, per_layer_hw_params)
                    self.layer_names.append(current_layer.name)

                    idx += 3
                    continue

#-------------------------------------------
            if current_layer.attr_type == 'DW':
                self.onchip_mem.clear()
                self.layer_names.append(current_layer.name)
                dw_layer_hw_params = self.load_hw_params_depthwise()
                self.conv2d_dw(current_layer, dw_layer_hw_params)
            if current_layer.attr_type == 'PW':
                self.onchip_mem.clear()
                self.layer_names.append(current_layer.name)
                # Stand alone runs on second CONV hardware.
                pw_layer_hw_params = self.load_hw_params_conv(False,True, config=0)
                self.conv2d_pw(current_layer, pw_layer_hw_params)

            if current_layer.attr_type == '3d':
                self.onchip_mem.clear()
                # self.stats['orig_idx'][layer_attr.layer_idx] = orig_idx - 1
                # Stand alone runs on second CONV hardware.
                per_layer_hw_params = self.load_hw_params_conv(False,True, config=0)
                self.conv2d_pw(current_layer, per_layer_hw_params)
                self.layer_names.append(current_layer.name)

            idx += 1
        return

    def fire_conv(self, first_layer, second_layer, third_layer):
        first_layer_hw_params = self.load_hw_params_pointwise(True, False)
        # --- Pointwise 1 stats
        self.debug_message('cin= {} cout= {}'.format(first_layer.Cin, first_layer.Cout))
        self.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
        first_num_macs_w_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_wxx_type
        first_layer_mac_units = first_layer_hw_params.mac_cxx * first_num_macs_w_units * first_layer_hw_params.mac_fx
        first_layer_padd_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_fx
        self.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
        self.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)


        # -- Pointwise 2 in second unit  --
        second_layer_hw_params = self.load_hw_params_pointwise(False, False)
        self.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
        second_num_macs_w_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_wxx_type
        second_layer_mac_units = second_layer_hw_params.mac_cxx * second_num_macs_w_units * second_layer_hw_params.mac_fx
        self.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)
        second_layer_padd_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_fx
        self.insert_max_stats('padd_units_available', second_layer.layer_idx, second_layer_padd_units)

        #-- 3d CONV in parallel with second unit --
        third_layer_hw_params = self.load_hw_params_conv(False,False, config=4)
        self.debug_message('{} {} {}'.format(third_layer.layer_idx, third_layer.name, third_layer.attr_type))
        third_num_macs_w_units = third_layer_hw_params.mac_wxx * third_layer_hw_params.mac_wxx_type * third_layer_hw_params.mac_wxx_type
        third_layer_mac_units = third_layer_hw_params.mac_cxx * third_num_macs_w_units * third_layer_hw_params.mac_fx
        self.insert_max_stats('mac_units_available', third_layer.layer_idx, third_layer_mac_units)
        third_layer_padd_units = third_layer_hw_params.mac_wxx * third_layer_hw_params.mac_wxx_type * third_layer_hw_params.mac_wxx_type * third_layer_hw_params.mac_fx
        self.insert_max_stats('padd_units_available', third_layer.layer_idx, third_layer_padd_units)

        # adding mac units
        total_mac_units = first_layer_mac_units + second_layer_mac_units + third_layer_mac_units
        self.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
        self.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)
        self.insert_max_stats('total_mac_units', third_layer.layer_idx, total_mac_units)

        # print('total mac units: {} 1: {} 2: {} 3: {}'.format(total_mac_units, first_layer_mac_units,
        #                                                      second_layer_mac_units, third_layer_mac_units))
        self.insert_max_stats('total_padd_units', first_layer.layer_idx, first_layer_padd_units)
        # PW2, 3dCONV runs in parallel
        self.insert_max_stats('total_padd_units', second_layer.layer_idx, second_layer_padd_units)
        self.insert_max_stats('total_padd_units', third_layer.layer_idx, second_layer_padd_units)

        # -- schedule loop --
        start_hout_idx = 0
        start_hout_idx_2 = 0
        start_hout_idx_3 = 0
        end_hout_idx_2 = 0
        end_hout_idx_3 = 0

        batch_cycles_1 = {}
        batch_cycles_2 = {}
        batch_cycles_3 = {}
        time_idx_1 = 0
        time_idx_2 = 0
        time_idx_3 = 0
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
            start_wout_idx_3 = 0
            end_wout_idx = 0
            end_wout_idx_2 = 0
            end_wout_idx_3 = 0
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
##-------------------------------------------------------------------------------
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
                    # -- start of PCONV 2 in parallel with CONV
                    # --------------------------------------------------------------------------------------------------
                    time_idx_2 = time_idx_1 + 1
                    self.debug_message('  --second layer (hwc) PW input_act[{}:{}][{}:{}][{}:{}]'.format(
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
                    self.debug_message('=== Layer 2.1: PW ===')
                    pw2_start_indices = AttrDict({'hin': hin_2, 'win': win_2, 'cin': cin_2,
                                                  'hout': start_hout_idx, 'wout': start_wout_idx, 'cout': 0,
                                                  'end_hin': end_hin_idx_2 + 1,
                                                  'end_win': end_win_idx_2 + 1,
                                                  'end_cin': end_cin_idx_2 + 1,
                                                  'end_cout': second_layer.Cout
                                                  })

                    cycles_2 = self.conv2d_pw(second_layer, second_layer_hw_params, pw2_start_indices,
                                              num_cross_layers=2, layer_position_idx=1)
                    if time_idx_2 in batch_cycles_2:
                        batch_cycles_2[time_idx_2] += cycles_2
                    else:
                        batch_cycles_2[time_idx_2] = cycles_2

                    # --------------------------------------------------------------------------------------------------
                    # -- start of Conv 3 convolution using output  of 1st PCONV
                    # --------------------------------------------------------------------------------------------------
                    time_idx_3 = time_idx_1
                    self.debug_message('  --third layer (hwc) CONV input_act[{}:{}][{}:{}][{}:{}]'.format(
                        start_hout_idx, end_hout_idx,
                        start_wout_idx, end_wout_idx,
                        f, end_cout_idx - 1
                    ))

                    block_cin_3 = num_cout
                    block_win_3 = num_w_convs
                    block_hin_3 = num_h_convs
                    block_cout_3 = third_layer.Cout

                    if block_win_3 < third_layer.Ky:
                        num_w_convs_3 = 1
                    else:
                        num_w_convs_3 = int((block_win_3 - third_layer.Ky) / third_layer.Sy) + 1

                    if block_hin_3 < third_layer.Kx:
                        num_h_convs_3 = 1
                    else:
                        num_h_convs_3 = int(num_hin - first_layer.Kx / first_layer.Sx) + 1



                    # hin3, win3, cin3, hout3, wout3, cout3
                    hin_3 = start_hout_idx
                    win_3 = start_wout_idx
                    cin_3 = f
                    end_hin_idx_3 = hin_3 + block_hin_3 - 1
                    end_win_idx_3 = win_3 + block_win_3 - 1
                    end_cin_idx_3 = end_cout_idx

                    end_hout_idx_3 = start_hout_idx_3 + num_h_convs_3 - 1
                    end_wout_idx_3 = start_wout_idx_3 + num_w_convs_3 - 1
                    # Stage 3 PW2 - executes all filters.
                    start_cout_idx_3 = 0
                    end_cout_idx_3 = start_cout_idx_3 + block_cout_3 - 1

                    # Runs in HW|C|F for all filters of Layer 3
                    self.debug_message('=== Layer 3: PW ===')
                    pw3_start_indices = AttrDict({'hin': hin_3, 'win': win_3, 'cin': cin_3,
                        'hout': start_hout_idx, 'wout': start_wout_idx, 'cout': 0,
                        'end_hin': end_hin_idx_3 + 1,
                        'end_win': end_win_idx_3 + 1,
                        'end_cin':end_cin_idx_3 + 1,
                        'end_cout': third_layer.Cout
                        })

                    cycles_3 = self.conv2d_pw(third_layer, third_layer_hw_params, pw3_start_indices,
                                   num_cross_layers=2, layer_position_idx=1)
                    if time_idx_3 in batch_cycles_3:
                        batch_cycles_3[time_idx_3] += cycles_3
                    else:
                        batch_cycles_3[time_idx_3] = cycles_3
                    time_idx_1 += 1
                # end f
##-------------------------------------------------------------------------------
                start_wout_idx = end_wout_idx + 1
                # update start_wout indices for second/third layer
                start_wout_idx_2 = end_wout_idx_2 + 1
                start_wout_idx_3 = end_wout_idx_3 + 1

            # end w
            start_hout_idx = end_hout_idx + 1
            start_hout_idx_2 = end_hout_idx_2 + 1
            start_hout_idx_3 = end_hout_idx_3 + 1

        # end h
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

