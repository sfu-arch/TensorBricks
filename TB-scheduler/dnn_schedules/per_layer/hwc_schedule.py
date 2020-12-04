from dnn_schedules.schedule import Schedule
import math
from dnn_schedules.onchip_memory import OnchipMem

# class HWCSchedule(Schedule):
#     def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
#         super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)
#
#     def __str__(self):
#         return 'hwc_schedule'
#
#     def run_model(self):
#         pass

# Only works for stride =1 and padding = 'valid'
def conv2d_dw(cls, layer_attr, hw_params, dw_start_indices=None,
              num_cross_layers=1, layer_position_idx=0):
    current_cycles = 0
    if num_cross_layers ==1:
        cls.init_single_layer_dw(hw_params, layer_attr)

    cls.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))

    if dw_start_indices is None:
        start_hout_idx =0
        start_wout_idx = 0
    else:
        start_hout_idx = dw_start_indices.hout
        start_wout_idx = dw_start_indices.wout


    for hin in range(0, layer_attr.Hin, hw_params.hx):
        assert (hw_params.hx - layer_attr.Kx + 1 >0), \
            'Increase value of hx, hx ({}) - layer_attr.Kx ({}) + 1 <0'.format(hw_params.hx, layer_attr.Kx)

        # Adjust hin indices which will be used from previous convolutions
        if hin != 0:
            hin = hin - layer_attr.Kx + 1

        num_hin = min(hin + hw_params.hx, layer_attr.Hin) - hin
        if num_hin < layer_attr.Kx:
            num_h_convs = 1
        else:
            # In case of last values -- need to add padding information,
            #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
            num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

        end_hout_idx = start_hout_idx + num_h_convs - 1
        cls.debug_message('=====')

        for win in range(0, layer_attr.Win, hw_params.wx):
            assert(hw_params.wx - layer_attr.Ky +1 >0), \
                'Increase value of wx, wx ({}) - layer_attr.Ky ({}) + 1 <0'.format(hw_params.wx, layer_attr.Ky)

            if win != 0:
                win = win - layer_attr.Ky + 1

            num_win = min(win + hw_params.wx, layer_attr.Win) - win
            # note: # macs connections will differ for stride = 2
            if num_win < layer_attr.Ky:
                num_w_convs = 1
            else:
                num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

            end_wout_idx = start_wout_idx + num_w_convs - 1
            for cin in range(0, layer_attr.Cin, hw_params.cx):
                current_cycles += conv2d_dw_block(cls, layer_attr, hw_params, cin, win, hin, start_hout_idx, start_wout_idx,
                                     num_cross_layers, layer_position_idx)
            # end cin
            start_wout_idx = end_wout_idx + 1
            cls.debug_message(' --- ')
        # end win
        start_hout_idx = end_hout_idx + 1
    # end hin
    if num_cross_layers == 1:
        cls.insert_max_stats('global_cycles', layer_attr.layer_idx, current_cycles)
    return current_cycles

def conv2d_dw_block(cls, layer_attr, hw_params, cin, win, hin, start_hout_idx, start_wout_idx,
                    num_cross_layers=1,
                    layer_position_idx=0):
    current_cycles_block = 0
    dma_cycles = 0
    # -- calculate hin params --
    end_hin_idx = min(hin + hw_params.hx, layer_attr.Hin) - 1
    num_hin = end_hin_idx - hin + 1
    # In case of last values -- need to add padding information,
    #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
    if num_hin < layer_attr.Kx:
        num_h_convs = 1
    else:
        num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

    end_hout_idx = start_hout_idx + num_h_convs - 1
    # -- calculate win params --
    end_win_idx = min(win + hw_params.wx, layer_attr.Win) - 1
    num_win = end_win_idx - win + 1
    # note: # macs connections will differ for stride = 2
    if num_win < layer_attr.Ky:
        num_w_convs = 1
    else:
        num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

    end_wout_idx = start_wout_idx + num_w_convs - 1
    # -- calculate cin params
    start_cout_idx = int(cin * layer_attr.Depth_multiplier)
    end_cout_idx = min(int(start_cout_idx + hw_params.cx * layer_attr.Depth_multiplier),
                       layer_attr.Cin) - 1
    num_cout = end_cout_idx - start_cout_idx + 1
    end_cin_idx = min(cin + hw_params.cx, layer_attr.Cin) - 1
    num_cin = end_cin_idx - cin + 1

    dma_cycles = cls.load_weights_onchip(dma_cycles,
                                         cin, 0,
                                         end_cin_idx, 0,
                                         num_cout, 1,
                                         hw_params, layer_attr)

    cls.load_activations_onchip(hin, win, cin,
                            num_hin, num_win, num_cin, hw_params,
                            layer_position_idx, layer_attr, dma_cycles)


    # cycles information
    # mac utilization
    mac_util_cycles, mac_cycles = cls.calculate_dw_mac_utilization(num_h_convs, num_w_convs, num_cin, num_cout, hw_params, layer_attr, num_hin, num_win)

    cls.stats['mac_util_cycles'][layer_attr.layer_idx] += mac_util_cycles
    cls.stats['mac_cycles'][layer_attr.layer_idx] += mac_cycles
    cls.stats['dma_cycles'][layer_attr.layer_idx] += dma_cycles


    if (layer_position_idx == 0 and num_cross_layers > 1) or num_cross_layers == 1:
        if dma_cycles > mac_cycles:
            cls.stats['cycles_total'][layer_attr.layer_idx] += dma_cycles
            cls.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
            current_cycles_block += dma_cycles

        else:
            cls.stats['cycles_total'][layer_attr.layer_idx] += mac_cycles
            cls.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
            current_cycles_block += mac_cycles
    else:
        cls.stats['cycles_total'][layer_attr.layer_idx] += mac_cycles
        cls.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
        current_cycles_block += mac_cycles

    cls.insert_max_stats('cycles_max_per_batch', layer_attr.layer_idx, mac_cycles)

    # cls.stats['cumm_mac_cycles'][layer_attr.layer_idx] += num_w_convs * num_h_convs * num_cin
    # cls.stats['theoretical_max_mac_cycles'][layer_attr.layer_idx] += num_macs * num_h_convs * hw_params.cx

    num_wout = end_wout_idx - start_wout_idx + 1
    num_hout = end_hout_idx - start_hout_idx + 1

    if (num_cross_layers > 1 and layer_position_idx == num_cross_layers - 1) or num_cross_layers == 1:
        isdma = True
        mem_out_act = num_cout * num_wout * num_hout
        cls.mem_out_act_stats(layer_attr, isdma, mem_out_act, start_hout_idx, end_hout_idx, start_wout_idx,
                              end_wout_idx, start_cout_idx, end_cout_idx)


    elif num_cross_layers > 1 and layer_position_idx != num_cross_layers - 1:
        isdma = False
        mem_out_act = num_cout * num_wout * num_hout
        cls.mem_out_act_stats(layer_attr, isdma, mem_out_act, start_hout_idx, end_hout_idx, start_wout_idx,
                              end_wout_idx, start_cout_idx, end_cout_idx)

    return current_cycles_block


