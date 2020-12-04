from dnn_schedules.schedule import Schedule

class CWHSchedule1(Schedule):

    def __init__(self, net, model_name, verbose, hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, verbose, hardware_yaml, hardware_dict)

        self.run_model()
        self.print_stats()

    def run_model(self):
        for layer_name, layer_attr in self.net.layers.items():
            if layer_attr.attr_type == 'DW':
                self.conv2d_dw(layer_attr)
            if layer_attr.attr_type == 'PW':
                self.conv2d_pw(layer_attr)

    # Only works for stride =1 and padding = 'valid'
    def conv2d_dw(self, layer_attr):
        hw_params = self.hw_params.HWConfig
        num_macs = hw_params.wx + layer_attr.Ky - 1
        self.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name,layer_attr.attr_type))
        for cin in range(0, layer_attr.Cin, hw_params.cx):
            start_cout_idx = int(cin*layer_attr.Depth_multiplier)
            end_cout_idx = min(int(start_cout_idx + hw_params.cx*layer_attr.Depth_multiplier), layer_attr.Cin)-1
            num_cout = end_cout_idx - start_cout_idx + 1
            # print('inDMA wgts[{}:{}]'.format(start_cout_idx, end_cout_idx))
            self.stats['in_dma_wgt'][layer_attr.layer_idx] += num_cout*layer_attr.Kx*layer_attr.Ky

            prev_wgt_memory = self.stats['mem_wgt'][layer_attr.layer_idx]
            cur_wgt_memory = num_cout*layer_attr.Kx*layer_attr.Ky
            self.stats['mem_wgt'][layer_attr.layer_idx] = max(cur_wgt_memory, prev_wgt_memory)
            # print('=====')
            start_wout_idx = 0
            for win in range(0, layer_attr.Win, hw_params.wx):
                if win != 0:
                    win = win - layer_attr.Ky + 1
                start_hout_idx = 0
                for hin in range(0, layer_attr.Hin, hw_params.hx):

                    end_cin_idx = min(cin + hw_params.cx, layer_attr.Cin) - 1
                    end_win_idx = min(win+hw_params.wx, layer_attr.Win) - 1
                    end_hin_idx = min(hin+hw_params.hx, layer_attr.Hin) - 1
                    # print('inDMA ip_act[{}:{}][{}:{}][{}:{}]'.format(cin, end_cin_idx,
                    #                                                  win, end_win_idx,
                    #                                                  hin, end_hin_idx
                    #                                                  ))
                    num_cin = end_cin_idx - cin + 1
                    self.stats['in_dma_act'][layer_attr.layer_idx] += num_cin*(end_win_idx-win+1)*(end_hin_idx-hin+1)

                    # Adjust hin indices which will be used from previous convolutions
                    # Note: no such assumption is made for 'w' dimension
                    if hin != 0:
                        hin = hin - layer_attr.Kx + 1

                    num_win = end_win_idx - win + 1
                    num_hin = end_hin_idx - hin + 1

                    cur_in_act_memory = num_cin * num_hin * num_win
                    prev_in_act_memory = self.stats['mem_in_act'][layer_attr.layer_idx]
                    self.stats['mem_in_act'][layer_attr.layer_idx] = max(cur_in_act_memory, prev_in_act_memory)

                    # In case of last values -- need to add padding information,
                    #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
                    num_h_convs =  int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1
                    end_hout_idx = start_hout_idx + num_h_convs - 1

                    # note: # macs connections will differ for stride = 2
                    num_w_convs = int((num_win - layer_attr.Ky)/ layer_attr.Sy) + 1
                    end_wout_idx = start_wout_idx + num_w_convs - 1


                    # cycles information
                    self.stats['cycles_total'][layer_attr.layer_idx] += num_h_convs
                    prev_max_cycles_per_batch = self.stats['cycles_max_per_batch'][layer_attr.layer_idx]
                    self.stats['cycles_max_per_batch'][layer_attr.layer_idx] = max(num_h_convs, prev_max_cycles_per_batch)
                    # mac utilization
                    self.stats['cumm_mac_cycles'][layer_attr.layer_idx] += num_w_convs*num_h_convs*num_cin
                    self.stats['theoretical_max_mac_cycles'][layer_attr.layer_idx] += num_macs*num_h_convs*num_cin

                    # print('outDMA[{}:{}][{}:{}][{}:{}]'.format(start_cout_idx, end_cout_idx,
                    #                                            start_wout_idx,end_wout_idx,
                    #                                            start_hout_idx,end_hout_idx))

                    num_wout = end_wout_idx-start_wout_idx+1
                    num_hout = end_hout_idx-start_hout_idx+1
                    self.stats['out_dma_act'][layer_attr.layer_idx] += num_cout*num_wout*num_hout

                    prev_out_act_memory = self.stats['mem_out_act'][layer_attr.layer_idx]
                    cur_out_act_memory = num_cout*num_wout*num_hout
                    self.stats['mem_out_act'][layer_attr.layer_idx] = max(cur_out_act_memory,prev_out_act_memory)



                    # wasted_mac_cycles = int(abs(num_macs - hw_params.k + 1 - num_win)* num_hin)
                    # self.stats['wasted_mac_cycles'][layer_attr.layer_idx] += wasted_mac_cycles
                    # print('wasted_mac_cycles: ', wasted_mac_cycles)
                    start_hout_idx = end_hout_idx + 1

                start_wout_idx = end_wout_idx + 1
                # print(' --- ')

    # pointwise only stores partial products of output  in memory.
    # This implies input activations will be streamed multiple times. Hence, high DMA operations
    # Note this is a bad design, better idea is to store entire input/output activation
    # TODO: what to do with vector adds -- need to have enough to not stall pipeline
    def conv2d_pw(self, layer_attr):
        hw_params = self.hw_params.HWConfig
        num_macs = hw_params.wx + layer_attr.Ky - 1
        self.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name,layer_attr.attr_type))
        for f in range(0, layer_attr.Cout, hw_params.fx):
            end_f_idx = min( f + hw_params.fx, layer_attr.Cout) - 1
            num_f = end_f_idx - f + 1
            for cin in range(0, layer_attr.Cin, hw_params.cx):
                start_cout_idx = cin
                end_cout_idx = min(start_cout_idx + hw_params.cx, layer_attr.Cin)-1
                num_cout = end_cout_idx - start_cout_idx + 1
                self.debug_message('inDMA wgts [{}:{}][{}:{}]'.format(f,end_f_idx,start_cout_idx, end_cout_idx))
                self.stats['in_dma_wgt'][layer_attr.layer_idx] += num_f*num_cout*layer_attr.Kx*layer_attr.Ky

                prev_wgt_memory = self.stats['mem_wgt'][layer_attr.layer_idx]
                cur_wgt_memory = num_f*num_cout*layer_attr.Kx*layer_attr.Ky
                self.stats['mem_wgt'][layer_attr.layer_idx] = max(cur_wgt_memory, prev_wgt_memory)
                self.debug_message(' -- ')
                start_wout_idx = 0
                for win in range(0, layer_attr.Win, hw_params.wx):
                    if win != 0:
                        win = win - layer_attr.Ky + 1
                    start_hout_idx = 0
                    for hin in range(0, layer_attr.Hin, hw_params.hx):

                        end_cin_idx = end_cout_idx
                        end_win_idx = min(win+hw_params.wx, layer_attr.Win) - 1
                        end_hin_idx = min(hin+hw_params.hx, layer_attr.Hin) - 1
                        self.debug_message('inDMA ip_act[{}:{}][{}:{}][{}:{}]'.format(cin, end_cin_idx,
                                                                         win, end_win_idx,
                                                                         hin, end_hin_idx
                                                                         ))
                        num_cin = num_cout
                        self.stats['in_dma_act'][layer_attr.layer_idx] += num_cin*(end_win_idx-win+1)*(end_hin_idx-hin+1)

                        # Adjust hin indices which will be used from previous convolutions
                        # Note: no such assumption is made for 'w' dimension
                        if hin != 0:
                            hin = hin - layer_attr.Kx + 1

                        num_win = end_win_idx - win + 1
                        num_hin = end_hin_idx - hin + 1

                        cur_in_act_memory = num_cin * num_hin * num_win
                        prev_in_act_memory = self.stats['mem_in_act'][layer_attr.layer_idx]
                        self.stats['mem_in_act'][layer_attr.layer_idx] = max(cur_in_act_memory, prev_in_act_memory)

                        # In case of last values -- need to add padding information,
                        #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
                        num_h_convs =  int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1
                        end_hout_idx = start_hout_idx + num_h_convs - 1

                        # note: # macs connections will differ for stride = 2
                        num_w_convs = int((num_win - layer_attr.Ky)/ layer_attr.Sy) + 1
                        end_wout_idx = start_wout_idx + num_w_convs - 1

                        # cycles information
                        self.stats['cycles_total'][layer_attr.layer_idx] += num_h_convs
                        prev_max_cycles_per_batch = self.stats['cycles_max_per_batch'][layer_attr.layer_idx]
                        self.stats['cycles_max_per_batch'][layer_attr.layer_idx] = max(num_h_convs,
                                                                                       prev_max_cycles_per_batch)

                        # mac utilization
                        self.stats['cumm_mac_cycles'][layer_attr.layer_idx] += num_w_convs * num_h_convs * num_f*num_cout
                        self.stats['theoretical_max_mac_cycles'][layer_attr.layer_idx] += num_macs * num_h_convs * num_f*num_cout

                        # out_act = C[f:end_f_idx][start_cout_idx:end_cout_idx]
                        # W[start_wout_idx:end_wout_idx]H[start_hout_idx:end_hout_idx]
                        num_partial_convs = num_f*num_w_convs*num_h_convs
                        num_wout = end_wout_idx-start_wout_idx+1
                        num_hout = end_hout_idx-start_hout_idx+1

                        self.debug_message('partial_out_act = C[{}:{}]cin[{}:{}]'
                              'W[{}:{}]H[{}:{}]'.format(f, end_f_idx, start_cout_idx, end_cout_idx,
                                                        start_wout_idx, end_wout_idx,
                                                        start_hout_idx, end_hout_idx))



                # Add partial products
                # prev_partial_prod_param_list = [f,end_f_idx, start_cout_idx, end_cout_idx,
                #                      start_wout_idx, end_wout_idx,
                #                      start_hout_idx, end_hout_idx]

                partial_product_volume = num_f*num_cout*num_wout*num_hout
                # to add we need previous and current test_data
                cur_partial_prod_memory = partial_product_volume
                self.insert_max_stats('mem_partial_product',layer_attr.layer_idx, cur_partial_prod_memory)
                # addition for partial products requires some cycles
                self.stats['padd_total'][layer_attr.layer_idx] += partial_product_volume
                cur_partial_prod_add_per_batch = partial_product_volume
                self.insert_max_stats('max_padd_ops_per_batch',layer_attr.layer_idx, cur_partial_prod_add_per_batch)
            # out dma the channels which are completed
            self.debug_message('out_dma[{}:{}][{}:{}][{}:{}]'.format(f, end_f_idx, start_wout_idx, end_wout_idx,
                                                     start_hout_idx, end_hout_idx))
            self.stats['out_dma_act'][layer_attr.layer_idx] += num_f*num_wout*num_hout # reduction operation

            # assuming double buffer when partial products are added  pipeline -> a + b (prev_cycle) = c (dma out)
            prev_out_act_memory = self.stats['mem_out_act'][layer_attr.layer_idx]
            cur_out_act_memory = num_f*num_wout*num_hout
            self.stats['mem_out_act'][layer_attr.layer_idx] = max(cur_out_act_memory, prev_out_act_memory)

            self.debug_message('====')
