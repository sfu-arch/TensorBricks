from dnn_schedules.per_layer.hwc_schedule import  HWCSchedule


class FCHWSchedule(HWCSchedule):
    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'fchw_schedule'

    def run_model(self):
        for layer_name, layer_attr in self.net.layers.items():
            if layer_attr.attr_type == 'DW':
                dw_layer_hw_params = self.load_hw_params_depthwise()
                self.conv2d_dw(layer_attr, dw_layer_hw_params)
            if layer_attr.attr_type == 'PW':
                pw_layer_hw_params = self.load_hw_params_pointwise(True, True)
                self.conv2d_pw(layer_attr, pw_layer_hw_params)

            if layer_attr.attr_type == '3d':
                per_layer_hw_params = self.load_hw_params_conv(True, True)
                self.conv2d_pw(layer_attr, per_layer_hw_params)
                self.layer_names.append(layer_attr.name)
        return

    # pointwise only stores partial products of output  in memory.
    # This implies input activations will be streamed multiple times. Hence, high DMA operations
    # Note this is a bad design, better idea is to store entire input/output activation
    # TODO: what to do with vector adds -- need to have enough to not stall pipeline

    def conv2d_pw(self, layer_attr, hw_params, init_start_cout_idx=0, init_start_hout_idx=0, init_start_wout_idx=0,
                  is_cross_layer=False, is_first_layer=True, is_last_layer=True):

        self.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
        if not is_cross_layer:
            mac_units = hw_params.mac_cxx * (hw_params.mac_wxx - layer_attr.Ky + 1)*\
                        (hw_params.Kx*hw_params.Ky) * hw_params.mac_fx
            self.insert_max_stats('mac_units_available', layer_attr.layer_idx, mac_units)
            self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)

        for f in range(0, layer_attr.Cout, hw_params.fx):

            end_f_idx = min(f + hw_params.fx, layer_attr.Cout) - 1
            num_f = end_f_idx - f + 1

            for cin in range(0, layer_attr.Cin, hw_params.cxx):

                # ------ c parameter calculations
                start_cout_idx = cin
                end_cout_idx = min(start_cout_idx + hw_params.cxx, layer_attr.Cin) - 1
                num_cout = end_cout_idx - start_cout_idx + 1
                end_cin_idx = end_cout_idx
                num_cin = num_cout
                if not is_cross_layer:
                    wgt_volume = layer_attr.Cout * layer_attr.Cin * layer_attr.Kx * layer_attr.Ky
                    self.debug_message('inDMA wgts [{}:{}][{}:{}]'.format(f, end_f_idx, cin, end_cin_idx))
                    self.stats['in_dma_wgt'][layer_attr.layer_idx] = wgt_volume
                    self.insert_max_stats('mem_wgt', layer_attr.layer_idx, wgt_volume)
                    self.stats['cycles_total'][layer_attr.layer_idx] += hw_params.dma_cycles * wgt_volume
                    self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1

                start_hout_idx = init_start_hout_idx
                for hin in range(0, layer_attr.Hin, hw_params.hxx):
                    assert (hw_params.hxx - layer_attr.Kx + 1 >= 0), \
                        'Increase value of hxx, hxx ({}) - layer_attr.Kx ({}) + 1 <0'.format(hw_params.hxx,
                                                                                             layer_attr.Kx)
                    # Adjust hin indices which will be used from previous convolutions
                    # Note: no such assumption is made for 'w' dimension
                    if hin != 0:
                        hin = hin - layer_attr.Kx + 1

                    end_hin_idx = min(hin + hw_params.hxx, layer_attr.Hin) - 1
                    num_hin = end_hin_idx - hin + 1
                    if num_hin < layer_attr.Kx:
                        num_h_convs = 1
                    else:
                        # In case of last values -- need to add padding information,
                        #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
                        num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

                    end_hout_idx = start_hout_idx + num_h_convs - 1
                    num_hout = end_hout_idx - start_hout_idx + 1

                    start_wout_idx = init_start_wout_idx
                    for win in range(0, layer_attr.Win, hw_params.wxx):
                        assert (hw_params.wxx - layer_attr.Ky + 1 >= 0), \
                            'Increase value of wxx, wxx ({}) - layer_attr.Ky ({}) + 1 <0'.format(hw_params.wxx,
                                                                                                 layer_attr.Ky)
                        if win != 0:
                            win = win - layer_attr.Ky + 1

                        end_win_idx = min(win + hw_params.wxx, layer_attr.Win) - 1
                        num_win = end_win_idx - win + 1
                        if num_win < layer_attr.Ky:
                            num_w_convs = 1
                        else:
                            # note: # macs connections will differ for stride = 2
                            num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

                        num_macs_wx = hw_params.wxx - layer_attr.Ky + 1
                        end_wout_idx = start_wout_idx + num_w_convs - 1
                        num_wout = end_wout_idx - start_wout_idx + 1

                        self.debug_message(' -- ')
                        if (is_cross_layer and is_first_layer) or not is_cross_layer:
                            self.debug_message('inDMA ip_act[{}:{}][{}:{}][{}:{}]'.format(cin, end_cin_idx,
                                                                                          win, end_win_idx,
                                                                                          hin, end_hin_idx
                                                                                          ))

                            self.stats['in_dma_act'][layer_attr.layer_idx] += num_cin * num_win * num_hin
                            cur_in_act_memory = num_cin * num_hin * num_win
                            self.insert_max_stats('mem_in_act', layer_attr.layer_idx, cur_in_act_memory)

                        num_mac_cycles_all_filters = 0
                        self.debug_message('====')
                        start_wout_idx = end_wout_idx + 1
                    # end w
                    start_hout_idx = end_hout_idx + 1
                # end h
            # end cin
            #--- This is where partial product calculation happens

        # end f

        return

