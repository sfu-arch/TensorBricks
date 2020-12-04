from dnn_schedules.per_layer.hwcf_schedule import HWCFSchedule


class HWFCSchedule(HWCFSchedule):

    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'hwfc_schedule'

    def conv2d_pw(self, layer_attr, hw_params, init_start_cout_idx=0, init_start_hout_idx=0, init_start_wout_idx=0,
                  num_cross_layers=1, layer_position_idx=0):

        self.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
        if num_cross_layers==1:
            # This needs to change for PW and CONV
            # mac_units = hw_params.cxx * (hw_params.wxx - layer_attr.Ky + 1) * hw_params.fx
            mac_units = hw_params.mac_cxx * hw_params.mac_wxx *\
                                (layer_attr.Kx*layer_attr.Ky)* hw_params.mac_fx
            self.insert_max_stats('mac_units_available', layer_attr.layer_idx, mac_units)
            self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)


        start_hout_idx = init_start_hout_idx
        for hin in range(0, layer_attr.Hin, hw_params.hxx):
            # Adjust hin indices which will be used from previous convolutions
            # Note: no such assumption is made for 'w' dimension
            assert (hw_params.hxx - layer_attr.Kx + 1 > 0), \
                'Increase value of hxx, hxx ({}) - layer_attr.Kx ({}) + 1 <0'.format(hw_params.hxx, layer_attr.Kx)

            isconv_layer = (layer_attr.Kx !=1) or (layer_attr.Ky != 1)
            if hin !=0:
                hin = hin - layer_attr.Kx + 1

            end_hin_idx = min(hin + hw_params.hxx, layer_attr.Hin) - 1
            num_hin = end_hin_idx - hin + 1
            # In case of last values -- need to add padding information,
            #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
            if num_hin < layer_attr.Kx:
                num_h_convs = 1
            else:
                num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

            end_hout_idx = start_hout_idx + num_h_convs - 1
            num_hout = end_hout_idx - start_hout_idx + 1

            start_wout_idx = init_start_wout_idx
            for win in range(0, layer_attr.Win, hw_params.wxx):
                assert( hw_params.wxx - layer_attr.Ky +1 >0), \
                    'Increase value of wxx, wxx ({}) - layer_attr.Ky ({}) + 1 <0'.format(hw_params.wxx, layer_attr.Ky)

                assert( hw_params.mac_wxx - layer_attr.Ky +1 >0), \
                    'Increase value of mac_wx, mac_wx ({}) - layer_attr.Ky ({}) + 1 <0'.format(hw_params.mac_wxx, layer_attr.Ky)


                if win !=0 :
                    # Retains the previous Ky-1 windows from previous iterations
                    win = win - layer_attr.Ky + 1

                end_win_idx = min(win + hw_params.wxx, layer_attr.Win) - 1
                num_win = end_win_idx - win + 1
                if num_win < layer_attr.Ky:
                    num_w_convs = 1
                else:
                    # note: # macs connections will differ for stride = 2
                    num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

                end_wout_idx = start_wout_idx + num_w_convs - 1
                num_wout = end_wout_idx - start_wout_idx + 1

                for f in range(0,layer_attr.Cout, hw_params.fx):
                    # Note: to run HW|F|C we run the innermost for loop for F only Fx times in HW|C|F
                    # Thus, reusing conv2d_pw_block from HW|C|F dataflow
                    end_cout_idx= min(f + hw_params.fx, layer_attr.Cout) - 1
                    num_cout = num_f = end_cout_idx - f + 1

                    for cin in range(0, layer_attr.Cin, hw_params.cxx):
                        # Note: for HWCF: init_start_cout_idx=0, and init_end_cout_idx = Cout
                        self.conv2d_pw_block(cin, win, hin, start_wout_idx, start_hout_idx, layer_attr,
                                             hw_params,
                                             init_start_cout_idx=f, init_end_cout_idx=end_cout_idx,
                                             num_cross_layers=1,
                                             layer_position_idx=0)
                    # end cin
                    # a) P, b) DP c) not PD d) P2 in PDP e) C2 in CC
                    if (num_cross_layers >1 and layer_position_idx == num_cross_layers - 1) or num_cross_layers == 1:
                        mem_out_act = num_f * num_wout * num_hout
                        self.stats['out_dma_act'][layer_attr.layer_idx] += mem_out_act
                        self.debug_message('outDMA (cwh) op_act[0:{}][{}:{}][{}:{}]'.format(num_f - 1,
                                                                                            start_wout_idx,
                                                                                            end_wout_idx,
                                                                                            start_hout_idx,
                                                                                            end_hout_idx))
                        self.insert_max_stats('mem_out_act', layer_attr.layer_idx,mem_out_act)
                    # a) not P, b) not DP c) PD d) P1 in PDP 3) C1 in CC
                    if (num_cross_layers > 1 and layer_position_idx == 0):
                        mem_out_act = num_f * num_wout * num_hout
                        self.debug_message('mem_out_act op_act[0:{}][{}:{}][{}:{}]'.format(layer_attr.Cout - 1,
                                                                                           start_wout_idx, end_wout_idx,
                                                                                           start_hout_idx,
                                                                                           end_hout_idx))
                        self.insert_max_stats('mem_out_act', layer_attr.layer_idx, mem_out_act)
                #endf
                self.debug_message('====')
                start_wout_idx = end_wout_idx + 1
            # end w
            start_hout_idx = end_hout_idx + 1
        # end h
        return

