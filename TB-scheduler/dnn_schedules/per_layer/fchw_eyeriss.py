from dnn_schedules.per_layer.hwcf_schedule import HWCFSchedule
import math
import dnn_schedules.per_layer.sram_traffic as sram_ws
from tqdm import tqdm
from dnn_schedules.onchip_memory import OnchipMem


class FCHWScheduleEyeriss(HWCFSchedule):
    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        # 'tb' should be dummy in this case. VERIFY
        super().__init__('tb', net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)
        self.onchip_mem = OnchipMem()

    def __str__(self):
        return 'fchw_schedule'

    def run_model(self):
        for layer_name, layer_attr in tqdm(self.net.layers.items()):
            self.onchip_mem.clear()
            self.debug_message('===================================================')
            self.debug_message(' LAYER NAME: {} LAYER IDX: {}'.format(layer_attr.name, layer_attr.layer_idx))

            if layer_attr.type == 'Conv2d':
                per_layer_hw_params = self.load_hw_params_conv(True,False, config=2)
                self.run_tangram(layer_attr, per_layer_hw_params)
                self.layer_names.append(layer_attr.name)
        return

    def run_tangram(self, layer_attr, hw_params, init_start_cin_idx=0,
                    is_cross_layer=False, is_first_layer=True, is_last_layer=True):
        init_start_hout_idx = 0
        init_start_wout_idx = 0

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
                # ------ c parameter calculations
                # calculate cin indices
                end_cin_idx = min(cin + hw_params.cxx, layer_attr.Cin) - 1
                num_cin = end_cin_idx + 1
                self.debug_message(' -- ')
                tmp_wgt_h = 0
                tmp_wgt_h_end = layer_attr.Hin
                tmp_wgt_w = 0
                tmp_wgt_w_end = layer_attr.Win
                tmp_wgt_c = init_start_cin_idx + cin
                tmp_wgt_c_end = init_start_cin_idx + end_cin_idx

                if is_cross_layer == False and  \
                        not self.onchip_mem.check_if_wgt_exists(layer_attr.layer_idx,
                            tmp_wgt_h,tmp_wgt_h_end, tmp_wgt_w,tmp_wgt_w_end,
                            tmp_wgt_c,tmp_wgt_c_end, f, end_f_idx):

                    wgt_volume = num_f * num_cin * layer_attr.Kx * layer_attr.Ky
                    self.debug_message(
                        'inDMA wgts (f,c) [{}:{}][{}:{}]'.format(f, end_f_idx, cin, end_cin_idx))
                    self.stats['in_dma_wgt'][layer_attr.layer_idx] = wgt_volume
                    self.insert_max_stats('mem_wgt', layer_attr.layer_idx, wgt_volume)
                    self.stats['cycles_total'][layer_attr.layer_idx] += hw_params.dma_cycles * wgt_volume
                    self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
                    self.onchip_mem.insert_wgt(layer_attr.layer_idx, tmp_wgt_h, tmp_wgt_h_end,
                                               tmp_wgt_w, tmp_wgt_w_end,
                                               tmp_wgt_c, tmp_wgt_c_end, f, end_f_idx)

                tmp_ip_act_h = 0
                tmp_ip_act_h_end = layer_attr.Hin
                tmp_ip_act_w = 0
                tmp_ip_act_w_end = layer_attr.Win
                tmp_ip_act_c = init_start_cin_idx + cin
                tmp_ip_act_c_end = init_start_cin_idx + end_cin_idx
                if is_cross_layer == False and \
                        not self.onchip_mem.check_if_ip_act_exists(layer_attr.layer_idx,
                                                        tmp_ip_act_h,tmp_ip_act_h,
                                                        tmp_ip_act_w,tmp_ip_act_w,
                                                        tmp_ip_act_c,tmp_ip_act_c_end):

                    self.debug_message('inDMA ip_act[{}:{}][{}:{}][{}:{}]'.format(cin, end_cin_idx,
                                                                                  0, layer_attr.Win,
                                                                                  0, layer_attr.Hin
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
                self.eyeriss_block(layer_attr, hw_params, f, end_f_idx, cin, init_start_cin_idx,
                                   num_cout, num_f, dma_cycles,
                      is_cross_layer, is_first_layer, is_last_layer )

                # end cin

        # end f



        return

    def eyeriss_block(self, layer_attr, hw_params, f, end_f_idx, cin, init_start_cin_idx,
                      num_cout, num_f, dma_cycles,
                      is_cross_layer=False, is_first_layer=True, is_last_layer=True ):
        init_start_hout_idx = 0
        init_start_wout_idx= 0
        padd_cycles = 0
        # ------ c parameter calculations
        # calculate cin indices
        end_cin_idx = min(cin + hw_params.cxx, layer_attr.Cin) - 1
        num_cin = end_cin_idx + 1

        if not is_cross_layer:
            self.debug_message('using wgts (f,c) [{}:{}][{}:{}]'.format(f, end_f_idx, cin, end_cin_idx))
        # start h
        # --------------------------------------
        # For Eyeriss fx=12, cxx=14
        array_h = hw_params.X
        array_w = hw_params.Y
        # cycles1, mac_util_cycles1, num_hout1, num_wout1 = sram_ws.conv_cycles(layer_attr, hw_params)
        # num_mac_cycles1 = cycles1

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
            self.stats['padd_ops'][layer_attr.layer_idx] += num_cout * num_wout * num_hout
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

        return