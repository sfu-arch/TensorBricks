from dnn_schedules.schedule import Schedule
from dnn_schedules.cross_layer.cross_layer_utils import second_layer_dataflow, init_cc_stats
from dnn_schedules.per_layer.hwfc_schedule import conv2d_pw as hwfc_conv2d_pw
from dnn_schedules.per_layer.hwcf_schedule2 import conv2d_pw as hwcf_conv2d_pw
from dnn_schedules.per_layer.fchw_schedule import conv2d_pw as fchw_conv2d_pw
from dnn_schedules.per_layer.cfhw_schedule import conv2d_pw as cfhw_conv2d_pw
from attrdict import AttrDict
from dnn_schedules.per_layer.hwc_schedule import conv2d_dw  as hwc_conv2d_dw

from  dnn_schedules.cross_layer.two_layer.hwfc_cc import conv_conv, dw_conv

class FuseLayer(Schedule):

    def __init__(self,hw_type ,second_pw_dataflow, net, model_name, result_dir, verbose,
                 hardware_yaml=None, hardware_dict=None):
        assert hw_type == 'tangram', 'Wrong hardware type {} != tangram'.format(hw_type)
        super().__init__(hw_type,net, model_name, result_dir, verbose, hardware_yaml, hardware_dict)
        self.second_pw_dataflow = second_pw_dataflow
        assert self.second_pw_dataflow in second_layer_dataflow, 'dataflow not present for last layer'
        self.conv2d_dw = hwc_conv2d_dw

    def __str__(self):
        return 'fuselayer_fchw_{}_schedule_cc_{}'.format(self.second_pw_dataflow, self.hw_type)

    def assert_checks(self, hw_params):
        assert hw_params.hx == 1, 'fuse layer has hx=wx=1'
        assert hw_params.hxx == 1, 'fuse layer has hx=wx=1'
        assert hw_params.wx == 1, 'fuse layer has hx=wx=1'
        assert hw_params.wxx == 1, 'fuse layer has hx=wx=1'
        return

    # def run_model(self):
    #     items = list(self.net.layers.items())
    #     idx = 0
    #     while idx < len(items):
    #         current_layer = items[idx][1]
    #         if idx + 1 < len(items):
    #             next_layer = items[idx+1][1]
    #
    #             if (current_layer.attr_type == 'PW' and next_layer.attr_type == 'PW') or \
    #                 (current_layer.attr_type == '3d' and next_layer.attr_type == 'PW') or \
    #                 (current_layer.attr_type == 'PW' and next_layer.attr_type == '3d') or \
    #                 (current_layer.attr_type == '3d' and next_layer.attr_type == '3d'):
    #                 self.onchip_mem.clear()
    #                 conv_conv(self, items[idx][1], items[idx+1][1])
    #                 self.layer_names.append(current_layer.name)
    #                 idx += 2
    #                 continue
    #
    #             if (current_layer.attr_type == 'DW' and next_layer.attr_type == 'PW') or \
    #                 (current_layer.attr_type == 'DW' and next_layer.attr_type == '3d'):
    #                 self.onchip_mem.clear()
    #                 dw_conv(self, items[idx][1], items[idx+1][1])
    #                 self.layer_names.append(current_layer.name)
    #                 idx += 2
    #                 continue
    #
    #
    #         if current_layer.attr_type == 'DW':
    #             self.onchip_mem.clear()
    #             self.layer_names.append(current_layer.name)
    #             dw_layer_hw_params = self.load_hw_params_depthwise()
    #             self.assert_checks(dw_layer_hw_params)
    #
    #             self.conv2d_dw(self, current_layer, dw_layer_hw_params)
    #         if current_layer.attr_type == 'PW':
    #             self.onchip_mem.clear()
    #             self.layer_names.append(current_layer.name)
    #             pw_layer_hw_params = self.load_hw_params_pointwise(True), True
    #             # self.conv2d_pw(current_layer, pw_layer_hw_params)
    #             self.assert_checks(pw_layer_hw_params)
    #             eval_layer = '{}_conv2d_pw(self, current_layer, pw_layer_hw_params)'.format(self.second_pw_dataflow)
    #             _ = eval(eval_layer)
    #         if current_layer.attr_type == '3d':
    #             self.onchip_mem.clear()
    #             per_layer_hw_params = self.load_hw_params_conv(True, True)
    #             self.assert_checks(per_layer_hw_params)
    #             # self.conv2d_pw(current_layer, per_layer_hw_params)
    #             eval_layer = '{}_conv2d_pw(self, current_layer, per_layer_hw_params)'.format(self.second_pw_dataflow)
    #             _ = eval(eval_layer)
    #             self.layer_names.append(current_layer.name)
    #
    #         idx += 1
    #     return

    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0
        while idx < len(items):
            current_layer = items[idx][1]
            self.onchip_mem.clear()
            if idx + 1 < len(items):
                next_layer = items[idx+1][1]

                # a) In case of a PDP, DP is pipelined, b) C-C is pipelined, c) C-P-D-P: C,P,D-P is run.
                if (current_layer.attr_type == 'DW' and next_layer.attr_type == 'PW') or \
                        (current_layer.attr_type == '3d' and next_layer.attr_type == '3d'):
                    self.conv_conv(items[idx][1], items[idx+1][1])
                    idx += 2
                    continue

            # if current_layer.type == 'Conv2d' and (current_layer.attr_type == 'DW' or current_layer.attr_type == 'PW'):
            if current_layer.type == 'Conv2d':
                per_layer_hw_params = self.load_hw_params_conv(True, False, config=1)
                self.run_fuse_layer(current_layer, per_layer_hw_params)
                self.layer_names.append(current_layer.name)
                self.debug_message('===================================================')
                self.debug_message(' LAYER NAME: {} LAYER IDX: {}'.format(current_layer.name, current_layer.layer_idx))

            idx += 1

    # pointwise only stores partial products of output  in memory.
    # This implies input activations will be streamed multiple times. Hence, high DMA operations
    # Note this is a bad design, better idea is to store entire input/output activation
    # TODO: what to do with vector adds -- need to have enough to not stall pipeline
    def conv_conv(self, first_layer, second_layer):

        self.debug_message('{} {}'.format(second_layer.Cin, first_layer.Cout))
        # --- Layer 1 stats
        first_layer_hw_params = self.load_hw_params_conv(True, False, config=1)
        # self.assert_checks(first_layer_hw_params)

        self.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
        if first_layer.attr_type == 'DW':
            NUM_F1 = first_layer.Depth_multiplier
        else:
            NUM_F1 = first_layer.Cout
        first_wgt_volume = NUM_F1* first_layer.Cin * first_layer.Kx * first_layer.Ky
        self.debug_message('l1: inDMA wgts (f,c) [0:{}][0:{}]'.format(first_layer.Cout - 1, first_layer.Cin - 1))
        self.stats['in_dma_wgt'][first_layer.layer_idx] = first_wgt_volume
        self.insert_max_stats('mem_wgt', first_layer.layer_idx, first_wgt_volume)

        first_layer_mac_units = first_layer_hw_params.cxx * first_layer_hw_params.fx
        self.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
        self.stats['cycles_total'][first_layer.layer_idx] += first_layer_hw_params.dma_cycles * first_wgt_volume
        self.stats['is_dma_cycle_selected'][first_layer.layer_idx] += 1


        # -- Layer 2 stats --
        second_layer_hw_params = self.load_hw_params_conv(False, False, config=1)
        # self.assert_checks(second_layer_hw_params)
        self.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
        # Bring in all weights and store it on chip
        if second_layer.attr_type == 'DW':
            NUM_F2 = second_layer.Depth_multiplier
        else:
            NUM_F2 = second_layer.Cout
        second_total_weights = NUM_F2* second_layer.Cin * second_layer.Kx * second_layer.Ky
        self.debug_message('l2: inDMA wgts (f,c) [{}:{}]'.format(0, second_total_weights-1))
        self.stats['in_dma_wgt'][second_layer.layer_idx] = second_total_weights
        self.insert_max_stats('mem_wgt', second_layer.layer_idx, second_total_weights)
        second_layer_mac_units = second_layer_hw_params.cxx * second_layer_hw_params.fx
        self.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)
        self.stats['cycles_total'][second_layer.layer_idx] += second_layer_hw_params.dma_cycles * second_total_weights
        self.stats['is_dma_cycle_selected'][second_layer.layer_idx] += 1
        # adding mac units
        total_mac_units = first_layer_mac_units + second_layer_mac_units
        self.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
        self.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)
        # -- Layer 1: schedule loop --
        batch_cycles_1 = {}
        batch_cycles_2 = {}
        time_idx_1 = 0
        time_idx_2 = 0
        for hin in range(0, first_layer.Hin):
            for win in range(0, first_layer.Win):
                init_start_f_idx= 0

                if first_layer.attr_type == 'DW':
                    NUM_F = first_layer.Depth_multiplier
                    num_cout1 = first_layer.Cout
                else:
                    NUM_F = first_layer.Cout
                    num_cout1 = NUM_F

                for f in range(init_start_f_idx, NUM_F, first_layer_hw_params.fx):
                    # calculate filter indices
                    end_f_idx = min(f + first_layer_hw_params.fx, NUM_F) - 1
                    num_f = end_f_idx - f + 1
                    for cin in range(0, first_layer.Cin, first_layer_hw_params.cxx):
                        cycles_1 = self.add_fuse_fc_stats(hin, win, cin, f, end_f_idx, num_f, num_cout1,
                                  first_layer, first_layer_hw_params, hin, win, cin,
                                  is_cross_layer=True, is_first_layer=True, is_last_layer=False)
                        if time_idx_1 in batch_cycles_1:
                            batch_cycles_1[time_idx_1] += cycles_1
                        else:
                            batch_cycles_1[time_idx_1] = cycles_1
                    # end cin
                    time_idx_2 = time_idx_1 + 1
                    # output of 1*1*Fx is generated here, after iterating over 1*Cin
                    self.debug_message('====')
                    # # -- Layer 2: schedule loop --
                    # incoming block  i[1*1*Fx]  -->  filter[Fx2, K*K*Fx] -> o[1*1*Fx2]pp(0:Fx)
                    # Can assume a new block i[1,1,Fx]; filters[F2, K*K*Fx]; o[1*1*F2]
                    block_Cin2 = num_f
                    block_Win2 = 1
                    block_Hin2 = 1
                    # All filters in second layer
                    block_Cout2 = second_layer.Cout
                    start_cin2_idx = f

                    partial_layer2 = AttrDict({'name': second_layer.name, 'layer_idx': second_layer.layer_idx,
                                               'type': second_layer.type,
                                               'Kx': second_layer.Kx, 'Ky': second_layer.Ky, 'K': second_layer.K,
                                               'Cin': block_Cin2, 'Win': block_Win2, 'Hin': block_Hin2,
                                               'Hout': second_layer.Hout,
                                               'Cout': block_Cout2,
                                               # 'Wout': block_Wout, 'Hout': block_Hout,
                                               'attr_type': second_layer.attr_type,
                                               'Sx': second_layer.Sx, 'Sy': second_layer.Sy,
                                               'Px': second_layer.Px, 'Py': second_layer.Py
                                               # 'Bias': second_layer.Bias
                                               })
                    if second_layer.attr_type == 'DW':
                        # note: executes all filters and partial hin,win,cin
                        partial_layer2['Depth_multiplier'] = second_layer.Depth_multiplier
                    cycles_2 = self.run_fuse_layer(partial_layer2, second_layer_hw_params,init_start_hin_idx=hin,
                                        init_start_win_idx=win,
                                        init_start_cin_idx=start_cin2_idx,
                                   is_cross_layer=True, is_first_layer=False, is_last_layer=True)

                    if time_idx_2 in batch_cycles_2:
                        batch_cycles_2[time_idx_2] += cycles_2
                    else:
                        batch_cycles_2[time_idx_2] = cycles_2
                    #TODO need to add partial blocks o[1*1*Fx2] after the conv

                    time_idx_1 += 1
                # end f
            # end win
        # end hin
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

    def calc_fuse_stats(self, hw_params, num_cin, num_f, layer_attr):
        # due to reuse - K^2
        # 1*1*Fx values are generated which needs to be added to K^2*Fx values to get one conv.
        # Fuse layer caches previous computations so it is added later.

        # TODO make sure to add SRAM out_act to be (Ky-1)*Hout*Cout: Before DMA to DRAM.
        # Current vector mac + previous results vector add for each filter.
        in_prev_results_from_sram = num_f
        mac_cycles = num_cin*num_f + in_prev_results_from_sram
        mac_util = num_cin*num_f/(hw_params.cxx*hw_params.fx)
        mac_util_cycles = mac_util*mac_cycles

        # # Assumption runs in parallel.
        # padd_cycles = 1
        # # num_padds = 1x systolic_dim
        # # For 1 CONV, num_col filters, you get 1x1xnum_col-> ofmaps out.
        # # Thus, for all convs.
        # padd_util = col_util
        # padd_util_cycles = padd_util*padd_cycles
        # -------------------------------------------------
        # Calculate RF accesses and SRAM accesses
        # -------------------------------------------------

        # From in act SRAM to RF.
        in_rf_act_from_sram = num_cin

        # store to in_rf
        self.stats['mem_in_act_accesses'][layer_attr.layer_idx] += in_rf_act_from_sram + in_prev_results_from_sram

        # Act Reuse is Fx in space and  0 in time
        self.stats['in_rf_act_accesses'][layer_attr.layer_idx] += in_rf_act_from_sram + in_prev_results_from_sram
        self.insert_max_stats('in_rf_act_size', layer_attr.layer_idx, in_rf_act_from_sram)

        # From in wgt SRAM to RF
        # weights need to be fetched every time.
        in_rf_wgt_from_sram = num_f * num_cin
        self.stats['mem_wgt_accesses'][layer_attr.layer_idx] += in_rf_wgt_from_sram
        self.stats['wgt_rf_accesses'][layer_attr.layer_idx] += in_rf_wgt_from_sram

        # No reuse of weights in time
        self.insert_max_stats('wgt_rf_size', layer_attr.layer_idx, in_rf_wgt_from_sram)

        # From RF to out act SRAM
        # Need to save partial results per filter
        out_rf_act_to_sram = num_f
        self.stats['mem_out_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram
        self.stats['out_rf_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram
        self.insert_max_stats('out_rf_act_size', layer_attr.layer_idx, out_rf_act_to_sram)

        # mac_util_cycles, mac_cycles, padd_util_cycles, padd_cycles
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

        mac_cycles, mac_util_cycles = self.calc_fuse_stats(hw_params, num_cin, num_f, layer_attr)
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


