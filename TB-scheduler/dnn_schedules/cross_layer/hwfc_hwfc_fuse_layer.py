from dnn_schedules.per_layer.hwfc_fuse_layer import HWFCScheduleFuseLayer
from attrdict import AttrDict

class HWFCScheduleHWFCFuseLayer(HWFCScheduleFuseLayer):
    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'hwfc_schedule'

    def run_model(self):
        items = list(self.net.layers.items())
        idx = 0
        while idx < len(items):
            current_layer = items[idx][1]
            self.onchip_mem.clear()
            if idx + 1 < len(items):
                next_layer = items[idx+1][1]

                # if (current_layer.attr_type == 'DW' or current_layer.attr_type == 'PW') \
                #         and (next_layer.attr_type == 'DW' or next_layer.attr_type == 'PW') :
                if current_layer.type == 'Conv2d' and next_layer.type == 'Conv2d':
                    self.conv_conv(items[idx][1], items[idx+1][1])
                    idx += 2
                    continue

            # if current_layer.type == 'Conv2d' and (current_layer.attr_type == 'DW' or current_layer.attr_type == 'PW'):
            if current_layer.type == 'Conv2d':
                per_layer_hw_params = self.load_hw_params_conv(True, False,config=1)
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
        second_layer_hw_params = self.load_hw_params_conv(False, False,config=1)
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







