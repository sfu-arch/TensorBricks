from dnn_schedules.per_layer.fchw_eyeriss import FCHWScheduleEyeriss
# from dnn_schedules.cfhw.cfhw_eyeriss import run_tangram
import math
import dnn_schedules.per_layer.sram_traffic as sram_ws
from attrdict import AttrDict

class FCHWScheduleCFHWTangram(FCHWScheduleEyeriss):
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
                per_layer_hw_params = self.load_hw_params_conv(True, False, config=2)
                self.run_tangram(current_layer, per_layer_hw_params)
                self.layer_names.append(current_layer.name)
                self.debug_message('===================================================')
                self.debug_message(' LAYER NAME: {} LAYER IDX: {}'.format(current_layer.name, current_layer.layer_idx))

            idx += 1


    def conv_conv(self, first_layer, second_layer):

        self.debug_message('{} {}'.format(second_layer.Cin, first_layer.Cout))
        # --- Layer 1 stats
        first_layer_hw_params = self.load_hw_params_conv(True,False,config=2)
        self.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
        # assert first_layer_hw_params.fx == 1, 'Tangram first layer runs only 1 filter at a time'
        if first_layer.attr_type == 'DW':
            NUM_F1 = first_layer.Depth_multiplier
        else:
            NUM_F1 = first_layer.Cout
        first_wgt_volume = NUM_F1* first_layer.Cin * first_layer.Kx * first_layer.Ky
        self.debug_message('l1: inDMA wgts (f,c) [0:{}][0:{}]'.format(NUM_F1 - 1, first_layer.Cin - 1))
        self.stats['in_dma_wgt'][first_layer.layer_idx] = first_wgt_volume
        self.insert_max_stats('mem_wgt', first_layer.layer_idx, first_wgt_volume)

        first_layer_mac_units = first_layer_hw_params.X * first_layer_hw_params.Y
        self.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
        self.stats['cycles_total'][first_layer.layer_idx] += first_layer_hw_params.dma_cycles * first_wgt_volume
        self.stats['is_dma_cycle_selected'][first_layer.layer_idx] += 1


        # -- Layer 2 stats --
        second_layer_hw_params = self.load_hw_params_conv(False,False, config=2)
        self.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
        # Bring in all weights and store it on chip
        if second_layer.attr_type == 'DW':
            NUM_F2 = second_layer.Depth_multiplier
        else:
            NUM_F2 = second_layer.Cout
        second_total_weights = NUM_F2* second_layer.Cin * second_layer.Kx * second_layer.Ky
        self.debug_message('l2: inDMA wgts (total wgt) [{}:{}]'.format(0, second_total_weights-1))
        self.stats['in_dma_wgt'][second_layer.layer_idx] = second_total_weights
        self.insert_max_stats('mem_wgt', second_layer.layer_idx, second_total_weights)
        second_layer_mac_units = second_layer_hw_params.X * second_layer_hw_params.Y
        self.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)
        self.stats['cycles_total'][second_layer.layer_idx] += second_layer_hw_params.dma_cycles * second_total_weights
        self.stats['is_dma_cycle_selected'][second_layer.layer_idx] += 1
        # adding mac units
        total_mac_units = first_layer_mac_units + second_layer_mac_units
        self.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
        self.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)

        previous_ip_act_cached = {'cin1':0, 'end_cin_idx1': 0}
        for f1 in range(0, NUM_F1, first_layer_hw_params.fx):
            # calculate filter indices
            end_f_idx1 = min(f1 + first_layer_hw_params.fx, NUM_F1) - 1
            num_f1 = end_f_idx1 - f1 + 1
            start_f_idx1 = f1
            end_f_idx1 = end_f_idx1

            for cin1 in range(0, first_layer.Cin, first_layer_hw_params.cxx):
                # ------ c parameter calculations
                start_cin_idx1 = cin1
                end_cin_idx1 = min(cin1 + first_layer_hw_params.cxx, first_layer.Cin) - 1
                num_cin1 = end_cin_idx1 - start_cin_idx1 + 1
                num_wout1 = first_layer.Wout
                num_hout1 = first_layer.Hout
                self.debug_message(' -- ')

                #TODO: Add this optimization to all benchmarks: [Fuse layer and tensorbricks: HWCF, HWFC]
                # If input activation is previously not cached.
                if not (previous_ip_act_cached['cin1'] == cin1 and previous_ip_act_cached['end_cin_idx1'] == end_cin_idx1):
                    self.debug_message('inDMA ip_act[{}:{}][{}:{}][{}:{}]'.format(cin1, end_cin_idx1,
                                                                                  0, first_layer.Win,
                                                                                  0, first_layer.Hin
                                                                                  ))

                    dma_cycles1 = num_cin1 * first_layer.Win * first_layer.Hin * first_layer_hw_params.dma_cycles
                    self.stats['in_dma_act'][first_layer.layer_idx] += num_cin1 * first_layer.Win * first_layer.Hin
                    cur_in_act_memory = num_cin1 * first_layer.Hin * first_layer.Win
                    self.insert_max_stats('mem_in_act', first_layer.layer_idx, cur_in_act_memory)
                    self.debug_message('====')
                    previous_ip_act_cached = {'cin1':cin1, 'end_cin_idx1': end_cin_idx1}
                else:
                    dma_cycles1 = 0

                if first_layer.attr_type == 'DW':
                    num_cout1 = num_cin1 * second_layer.Depth_multiplier
                else:
                    num_cout1 = num_f1

                # iact (cwh)[cin:end_cin_idx][0:Win][0:Hin], filter (f) [f:end_f_idx][start_cout_idx:end_cout_idx]
                self.eyeriss_block(first_layer, first_layer_hw_params, f1, end_f_idx1, cin1,
                                   cin1, num_cout1, num_f1, dma_cycles1,
                               is_cross_layer=True, is_first_layer=True, is_last_layer=False)

                # output HWFx, all(Fx2) -> (HoutWoutFx2) to be updated multiple times except when c!=0
                # -- start Layer 2
                start_cin_idx2 = start_f_idx1
                end_cin_idx2 = end_f_idx1
                start_win_idx2 = 0
                end_win_idx2 = second_layer.Win -1
                start_hin_idx2 = 0
                end_hin_idx2 = second_layer.Hin - 1
                self.debug_message('  --second layer input_act(cwh) [{}:{}][{}:{}][{}:{}]'.format(
                    start_cin_idx2,end_cin_idx2,
                    start_win_idx2, end_win_idx2,
                    start_hin_idx2, end_hin_idx2))



                block_Cin2 = end_cin_idx2 - start_cin_idx2 + 1
                block_Win2 = second_layer.Win
                block_Hin2 = second_layer.Hin
                # ALl filters in second layer
                if second_layer.attr_type == 'DW':
                    block_Cout2 =  block_Cin2*second_layer.Depth_multiplier
                else:
                    block_Cout2 = second_layer.Cout
                partial_layer2 = AttrDict({'name': second_layer.name, 'layer_idx': second_layer.layer_idx,
                                          'type': second_layer.type,
                                          'Kx': second_layer.Kx, 'Ky': second_layer.Ky, 'K': second_layer.K,
                                          'Cin': block_Cin2, 'Win': block_Win2, 'Hin': block_Hin2,
                                          'Cout': block_Cout2,
                                          # 'Wout': block_Wout, 'Hout': block_Hout,
                                          'attr_type': second_layer.attr_type,
                                          'Sx': second_layer.Sx, 'Sy': second_layer.Sy,
                                          'Px': second_layer.Px, 'Py': second_layer.Py
                                          # 'Bias': second_layer.Bias
                                          })
                if second_layer.attr_type == 'DW':
                    partial_layer2['Depth_multiplier'] = second_layer.Depth_multiplier

                # run C|F|HW dataflow tangram: Only Cin is partial; complete Hin,Win,F
                self.run_tangram_cfhw(partial_layer2, second_layer_hw_params,
                                 init_start_cin_idx=start_cin_idx2,
                            is_cross_layer=True, is_first_layer=False, is_last_layer=True)

    # TODO: Use only one copy: multiple inheritance
    def run_tangram_cfhw(self, layer_attr, hw_params, init_start_cin_idx=0,
                         is_cross_layer=False, is_first_layer=True, is_last_layer=True):
        init_start_hout_idx = 0
        init_start_wout_idx = 0

        if layer_attr.attr_type == 'DW':
            NUM_F = layer_attr.Depth_multiplier
        else:
            NUM_F = layer_attr.Cout

        for cin in range(0, layer_attr.Cin, hw_params.cxx):
            # calculate cin indices
            end_cin_idx = min(cin + hw_params.cxx, layer_attr.Cin) - 1
            num_cin = end_cin_idx + 1

            for f in range(0, NUM_F, hw_params.fx):
                # calculate filter indices
                end_f_idx = min(f + hw_params.fx, NUM_F) - 1
                num_f = end_f_idx - f + 1
                if layer_attr.attr_type == 'DW':
                    num_cout = layer_attr.Cout
                else:
                    num_cout = num_f

                padd_cycles = 0
                self.debug_message(' -- ')
                tmp_wgt_h = 0
                tmp_wgt_h_end = layer_attr.Hin
                tmp_wgt_w = 0
                tmp_wgt_w_end = layer_attr.Win
                tmp_wgt_c = init_start_cin_idx + cin
                tmp_wgt_c_end = init_start_cin_idx + end_cin_idx

                if is_cross_layer == False and \
                        not self.onchip_mem.check_if_wgt_exists(layer_attr.layer_idx,
                                                                tmp_wgt_h, tmp_wgt_h_end, tmp_wgt_w, tmp_wgt_w_end,
                                                                tmp_wgt_c, tmp_wgt_c_end, f, end_f_idx):
                    wgt_volume = NUM_F * layer_attr.Cin * layer_attr.Kx * layer_attr.Ky
                    self.debug_message(
                        'inDMA wgts (f,c) [{}:{}][{}:{}]'.format(0, NUM_F - 1, 0, layer_attr.Cin - 1))
                    self.stats['in_dma_wgt'][layer_attr.layer_idx] = wgt_volume
                    self.insert_max_stats('mem_wgt', layer_attr.layer_idx, wgt_volume)
                    self.stats['cycles_total'][layer_attr.layer_idx] += hw_params.dma_cycles * wgt_volume
                    self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
                    self.onchip_mem.insert_wgt(layer_attr.layer_idx, tmp_wgt_h, tmp_wgt_h_end,
                                               tmp_wgt_w, tmp_wgt_w_end,
                                               tmp_wgt_c, tmp_wgt_c_end, f, end_f_idx)
                # if not is_cross_layer:
                #     self.debug_message('using wgts (f,c) [{}:{}][{}:{}]'.format(f, end_f_idx, cin, end_cin_idx))
                # start h
                self.debug_message(' -- ')
                tmp_ip_act_h = 0
                tmp_ip_act_h_end = layer_attr.Hin
                tmp_ip_act_w = 0
                tmp_ip_act_w_end = layer_attr.Win
                tmp_ip_act_c = init_start_cin_idx + cin
                tmp_ip_act_c_end = init_start_cin_idx + end_cin_idx
                if ((is_cross_layer and is_first_layer) or not is_cross_layer) and \
                        not self.onchip_mem.check_if_ip_act_exists(layer_attr.layer_idx,
                                                                   tmp_ip_act_h, tmp_ip_act_h,
                                                                   tmp_ip_act_w, tmp_ip_act_w,
                                                                   tmp_ip_act_c, tmp_ip_act_c_end):
                    self.debug_message('inDMA ip_act[{}:{}][{}:{}][{}:{}]'.format(cin, end_cin_idx,
                                                                                  0, layer_attr.Win - 1,
                                                                                  0, layer_attr.Hin - 1
                                                                                  ))

                    self.stats['in_dma_act'][layer_attr.layer_idx] += num_cin * layer_attr.Win * layer_attr.Hin
                    cur_in_act_memory = num_cin * layer_attr.Hin * layer_attr.Win
                    self.insert_max_stats('mem_in_act', layer_attr.layer_idx, cur_in_act_memory)
                    dma_cycles = num_cin * layer_attr.Win * layer_attr.Hin * hw_params.dma_cycles
                    self.onchip_mem.insert_ip_act(layer_attr.layer_idx,
                                                  tmp_ip_act_h, tmp_ip_act_h_end,
                                                  tmp_ip_act_w, tmp_ip_act_w_end,
                                                  tmp_ip_act_c, tmp_ip_act_c_end)
                else:
                    dma_cycles = 0

                self.debug_message('====')
                # --------------------------------------
                # For Eyeriss fx=12, cxx=14
                array_h = hw_params.X
                array_w = hw_params.Y
                # Assumes Sx=Sy
                cycles, num_mac_cycles, mac_util_cycles, num_hout, num_wout = \
                    sram_ws.sram_traffic(
                        dimension_rows=array_h,
                        dimension_cols=array_w,
                        ifmap_h=layer_attr.Hin, ifmap_w=layer_attr.Win,
                        filt_h=layer_attr.Kx, filt_w=layer_attr.Ky,
                        num_channels=num_cin,
                        strides=layer_attr.Sx, num_filt=num_f)

                assert cycles == num_mac_cycles, 'cycles: {} != num_mac_cycles: {}'.format(cycles, num_mac_cycles)
                start_hout_idx = init_start_hout_idx
                end_hout_idx = init_start_hout_idx + num_hout - 1

                start_wout_idx = init_start_wout_idx
                end_wout_idx = init_start_wout_idx + num_wout - 1

                self.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
                self.debug_message('cycles: {} mac util: {}'.format(cycles, mac_util_cycles / num_mac_cycles * 100))

                if not is_cross_layer:
                    mac_units = array_h * array_w
                    self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)
                # --------------------------------------

                # end h

                if (is_cross_layer and is_last_layer) or not is_cross_layer:
                    self.stats['out_dma_act'][layer_attr.layer_idx] += num_cout * num_wout * num_hout
                    self.debug_message('outDMA (cwh) op_act[0:{}][{}:{}][{}:{}]'.format(num_cout - 1,
                                                                                        0, end_wout_idx,
                                                                                        start_hout_idx, end_hout_idx))
                    self.insert_max_stats('mem_out_act', layer_attr.layer_idx, num_cout * num_wout * num_hout)
                if is_cross_layer and not is_last_layer:
                    self.debug_message('mem_out_act op_act[0:{}][{}:{}][{}:{}]'.format(num_cout - 1,
                                                                                       start_wout_idx, end_wout_idx,
                                                                                       start_hout_idx, end_hout_idx))
                    self.insert_max_stats('mem_out_act', layer_attr.layer_idx, num_cout * num_wout * num_hout)

                self.debug_message('====')

                # If it is DW, then padds are not done in CONV hardware.
                if layer_attr.attr_type != 'DW' and (cin != 0 or (is_cross_layer and (init_start_cin_idx != 0))):
                    # since streaming in hx only cx*wx *fx mac happens which are reduced to wx*fx
                    # which are partial products
                    self.insert_max_stats('padd_cycles_max_per_batch', layer_attr.layer_idx,
                                          num_cout * num_wout)
                    self.stats['padd_cycles'][layer_attr.layer_idx] += num_cout * num_wout * num_hout
                    padd_cycles += num_cout * num_wout * math.ceil(num_hout / (hw_params.X*hw_params.Y))
                    self.insert_max_stats('mem_partial_product', layer_attr.layer_idx,
                                          num_cout * num_wout * num_hout)

                # cycles information
                current_batch_cycles = num_mac_cycles + padd_cycles
                self.stats['mac_cycles'][layer_attr.layer_idx] += num_mac_cycles
                self.stats['mac_util_cycles'][layer_attr.layer_idx] += mac_util_cycles
                self.stats['padd_cycles'][layer_attr.layer_idx] += padd_cycles

                self.insert_max_stats('cycles_max_per_batch', layer_attr.layer_idx, current_batch_cycles)
                if not is_cross_layer or (is_cross_layer and is_first_layer):
                    # if dma cost is higher then add dma cycles
                    if dma_cycles > current_batch_cycles:
                        self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
                        self.stats['cycles_total'][layer_attr.layer_idx] += dma_cycles
                    else:
                        self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
                        self.stats['cycles_total'][layer_attr.layer_idx] += current_batch_cycles
                else:
                    self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
                    self.stats['cycles_total'][layer_attr.layer_idx] += current_batch_cycles

            # end f

        # end cin

        return