from dnn_schedules.per_layer.hwc_schedule import HWCSchedule
from tqdm import tqdm


class HWFCScheduleFuseLayer(HWCSchedule):
    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)



    def __str__(self):
        return 'hwfc_schedule_Fuse Layer'

    def run_model(self):
        for layer_name, layer_attr in tqdm(self.net.layers.items()):
            self.onchip_mem.clear()
            self.debug_message('===================================================')
            self.debug_message(' LAYER NAME: {} LAYER IDX: {}'.format(layer_attr.name, layer_attr.layer_idx))

            if layer_attr.type == 'Conv2d':
                per_layer_hw_params = self.load_hw_params_conv(True, False, isfuse_layer=True)
                self.run_fuse_layer(layer_attr, per_layer_hw_params)
                self.layer_names.append(layer_attr.name)
        return

    # pointwise only stores partial products of output  in memory.
    # This implies input activations will be streamed multiple times. Hence, high DMA operations
    # Note this is a bad design, better idea is to store entire input/output activation
    # TODO: what to do with vector adds -- need to have enough to not stall pipeline
    def run_fuse_layer(self, layer_attr, hw_params,
                       init_start_hin_idx=0,
                       init_start_win_idx=0,
                       init_start_cin_idx=0,
                       is_cross_layer=False, is_first_layer=True, is_last_layer=True):

        # self.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
        current_cycles = 0
        for hin in range(0, layer_attr.Hin):
            for win in range(0, layer_attr.Win):
                current_cycles += self.run_fuse_block(hin, win, layer_attr, hw_params,
                                    init_start_hin_idx,
                                    init_start_win_idx,
                                    init_start_cin_idx,
                                    is_cross_layer, is_first_layer, is_last_layer)
                self.debug_message('====')
                # start_wout_idx = end_wout_idx + 1
            # end w
            # start_hout_idx = end_hout_idx + 1
        # end h
        if is_cross_layer==False:
            self.insert_max_stats('global_cycles', layer_attr.layer_idx, current_cycles)

        return current_cycles



    def run_fuse_block(self, hin, win, layer_attr, hw_params,
                        init_start_hin_idx,
                        init_start_win_idx,
                        init_start_cin_idx,
                        is_cross_layer= False, is_first_layer=True, is_last_layer=True):
        current_cycles_block = 0
        if layer_attr.attr_type == 'DW':
            NUM_F = layer_attr.Depth_multiplier

        else:
            NUM_F = layer_attr.Cout


        for f in range(0, NUM_F, hw_params.fx):
            # calculate filter indices
            end_f_idx = min(f + hw_params.fx, NUM_F) - 1
            num_f = end_f_idx - f + 1

            if layer_attr.attr_type == 'DW':
                num_cout = layer_attr.Cout
            else:
                num_cout = num_f

            for cin in range(0, layer_attr.Cin, hw_params.cxx):
                current_cycles_block += self.add_fuse_fc_stats(hin, win, cin, f, end_f_idx, num_f, num_cout,
                               layer_attr, hw_params,
                               init_start_hin_idx,
                               init_start_win_idx,
                               init_start_cin_idx,
                               is_cross_layer, is_first_layer, is_last_layer)
            # end cin
        # end f

        return current_cycles_block

    def calc_fuse_stats(self, hw_params, num_cin, num_filt, filt_h, filt_w):
        # due to reuse - K^2
        # 1*1*Fx values are generated which needs to be added to K^2*Fx values to get one conv.
        # Fuse layer caches previous computations so it is added later.
        mac_cycles = filt_h*filt_w
        mac_util = num_cin*num_filt/(hw_params.cxx*hw_params.fx)
        mac_util_cycles = mac_util*mac_cycles

        return mac_cycles, mac_util_cycles



    def add_fuse_fc_stats(self,hin, win, cin, f,end_f_idx, num_f,num_cout,
                          layer_attr,hw_params,
                          init_start_hin_idx,
                          init_start_win_idx,
                          init_start_cin_idx,
                          is_cross_layer, is_first_layer, is_last_layer):
        current_cycles_stats = 0
        padd_cycles = 0

        # ------ c parameter calculations
        # calculate cin indices
        end_cin_idx = min(cin + hw_params.cxx, layer_attr.Cin) - 1
        num_cin = end_cin_idx - cin + 1

        tmp_wgt_h = init_start_hin_idx+ hin
        tmp_wgt_w = init_start_win_idx + win
        tmp_wgt_c = init_start_cin_idx+cin
        tmp_wgt_c_end = init_start_cin_idx+ end_cin_idx

        if not is_cross_layer and \
                not self.onchip_mem.check_if_wgt_exists(layer_attr.layer_idx, tmp_wgt_h,tmp_wgt_h,
                                                        tmp_wgt_w,tmp_wgt_w,
                                                        tmp_wgt_c,tmp_wgt_c_end,
                                                        f, end_f_idx):
            wgt_volume = num_f * num_cin * 1*1
            self.debug_message(
                'inDMA wgts (h,w,f,c) [{}:{}][{}:{}]'.format(f, end_f_idx, tmp_wgt_c,
                                                         tmp_wgt_c_end))
            self.stats['in_dma_wgt'][layer_attr.layer_idx] = wgt_volume
            self.insert_max_stats('mem_wgt', layer_attr.layer_idx, wgt_volume)
            self.stats['cycles_total'][layer_attr.layer_idx] += hw_params.dma_cycles * wgt_volume
            current_cycles_stats +=  hw_params.dma_cycles * wgt_volume
            self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
            self.onchip_mem.insert_wgt(layer_attr.layer_idx,tmp_wgt_h,tmp_wgt_h,
                                       tmp_wgt_w,tmp_wgt_w,tmp_wgt_c,tmp_wgt_c_end,f,end_f_idx)

        mac_cycles, mac_util_cycles = self.calc_fuse_stats(hw_params, num_cin, num_f,
                                                       layer_attr.Kx, layer_attr.Ky)
        # num_hout, num_wout
        self.debug_message(
            '{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
        self.debug_message('cycles: {} mac util: {}'.format(mac_cycles, mac_util_cycles / mac_cycles * 100))

        if not is_cross_layer:
            mac_units = hw_params.cxx * hw_params.fx
            self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)
        # --------------------------------------
        self.debug_message(' -- ')
        tmp_ip_act_h = init_start_hin_idx+ hin
        tmp_ip_act_w = init_start_win_idx + win
        tmp_ip_act_c = init_start_cin_idx+cin
        tmp_ip_act_c_end = init_start_cin_idx+ end_cin_idx
        if (is_cross_layer and is_first_layer) or not is_cross_layer and \
                not self.onchip_mem.check_if_ip_act_exists(layer_attr.layer_idx, tmp_ip_act_h,tmp_ip_act_h,
                                                        tmp_ip_act_w,tmp_ip_act_w,
                                                        tmp_ip_act_c,tmp_ip_act_c_end):

            self.debug_message('inDMA (c) ip_act[{}:{}]'.format(cin, end_cin_idx))

            self.stats['in_dma_act'][layer_attr.layer_idx] += num_cin
            self.insert_max_stats('mem_in_act', layer_attr.layer_idx, num_cin)
            dma_cycles = num_cin * hw_params.dma_cycles
            self.onchip_mem.insert_ip_act(layer_attr.layer_idx,tmp_ip_act_h,tmp_ip_act_h,tmp_ip_act_w,tmp_ip_act_w,
                                          tmp_ip_act_c,tmp_ip_act_c_end)

        else:
            dma_cycles = 0

        self.debug_message('====')

        if (is_cross_layer and is_last_layer) or not is_cross_layer:
            self.stats['out_dma_act'][layer_attr.layer_idx] += num_cout
            self.debug_message('mem_out_act op_act[0:{}]'.format(num_cout - 1))
            self.insert_max_stats('mem_out_act', layer_attr.layer_idx, num_cout * layer_attr.Hout * layer_attr.Ky)
        if is_cross_layer and not is_last_layer:
            self.debug_message('mem_out_act op_act[0:{}]'.format(num_cout - 1))
            self.insert_max_stats('mem_out_act', layer_attr.layer_idx, num_cout * layer_attr.Hout * layer_attr.Ky)

        self.debug_message('====')

        # If it is DW, then padds are not done in CONV hardware.
        if layer_attr.attr_type != 'DW' and (cin != 0 or (is_cross_layer and (init_start_cin_idx != 0))):
            self.insert_max_stats('padd_cycles_max_per_batch', layer_attr.layer_idx,
                                  num_cout)
            self.stats['padd_ops'][layer_attr.layer_idx] += num_cout
            padd_cycles += num_cout
            self.insert_max_stats('mem_partial_product', layer_attr.layer_idx,
                                  num_cout)

        # cycles information
        current_batch_cycles = mac_cycles + padd_cycles
        self.stats['mac_cycles'][layer_attr.layer_idx] += mac_cycles
        self.stats['mac_util_cycles'][layer_attr.layer_idx] += mac_util_cycles
        self.stats['padd_cycles'][layer_attr.layer_idx] += padd_cycles
        self.stats['dma_cycles'][layer_attr.layer_idx] += dma_cycles

        self.insert_max_stats('cycles_max_per_batch', layer_attr.layer_idx, current_batch_cycles)
        if not is_cross_layer or (is_cross_layer and is_first_layer):
            # if dma cost is higher then add dma cycles
            if dma_cycles > current_batch_cycles:
                self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
                self.stats['cycles_total'][layer_attr.layer_idx] += dma_cycles
                current_cycles_stats += dma_cycles
            else:
                self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
                self.stats['cycles_total'][layer_attr.layer_idx] += current_batch_cycles
                current_cycles_stats += current_batch_cycles
        else:
            self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
            self.stats['cycles_total'][layer_attr.layer_idx] += current_batch_cycles
            current_cycles_stats += current_batch_cycles

        return current_cycles_stats

