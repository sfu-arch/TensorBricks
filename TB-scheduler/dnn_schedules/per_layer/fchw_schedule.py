from dnn_schedules.per_layer.hwc_schedule import conv2d_dw
from dnn_schedules.schedule import Schedule
import math


class FCHWSchedule(Schedule):

    def __init__(self,hw_type, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(hw_type, net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'cfhw_pw_schedule + hwc_dw_schedule'

    def run_model(self):
        # orig_idx=0
        for layer_name, layer_attr in self.net.layers.items():
            self.onchip_mem.clear()
            # orig_idx += 1
            if layer_attr.attr_type == 'DW':
                # self.stats['orig_idx'][layer_attr.layer_idx] = orig_idx - 1
                dw_layer_hw_params = self.load_hw_params_depthwise()
                conv2d_dw(self, layer_attr, dw_layer_hw_params)
                self.layer_names.append(layer_attr.name)
            if layer_attr.attr_type == 'PW':
                # self.stats['orig_idx'][layer_attr.layer_idx] = orig_idx - 1
                pw_layer_hw_params = self.load_hw_params_pointwise(True, True)
                conv2d_pw(self, layer_attr, pw_layer_hw_params)
                self.layer_names.append(layer_attr.name)

            if layer_attr.attr_type == '3d':
                # self.stats['orig_idx'][layer_attr.layer_idx] = orig_idx - 1
                per_layer_hw_params = self.load_hw_params_conv(True, True)
                conv2d_pw(self, layer_attr, per_layer_hw_params)
                self.layer_names.append(layer_attr.name)
        return

def conv2d_pw(cls, layer_attr, hw_params, pw_start_indices=None, num_cross_layers=1,
              layer_position_idx=0):
    current_cycles = 0
    # SET PARTIAL LAYER ATTRIBUTES
    INIT_START_HIN_IDX, INIT_START_WIN_IDX, INIT_START_CIN_IDX, INIT_START_COUT_IDX, \
    INIT_START_HOUT_IDX, INIT_START_WOUT_IDX, \
    INIT_END_HIN, INIT_END_WIN, INIT_END_CIN, INIT_END_COUT = \
        cls.set_partial_layer_attributes(pw_start_indices, layer_attr)


    LAYER_HIN = INIT_END_HIN - INIT_START_HIN_IDX
    if LAYER_HIN < layer_attr.Kx:
        LAYER_HOUT = 1
    else:
        LAYER_HOUT = int(LAYER_HIN - layer_attr.Kx / layer_attr.Sx) + 1

    LAYER_WIN = INIT_END_WIN - INIT_START_WIN_IDX
    if LAYER_WIN < layer_attr.Ky:
        LAYER_WOUT = 1
    else:
        LAYER_WOUT = int(LAYER_WIN - layer_attr.Ky / layer_attr.Sy) + 1

    #TODO: calculate LAYER_WOUT also

    isconv_layer = (layer_attr.Kx != 1) or (layer_attr.Ky != 1)
    cls.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
    if num_cross_layers==1:
        cls.init_single_layer(hw_params, layer_attr)


######################################################
    for orig_f in range(INIT_START_COUT_IDX, INIT_END_COUT, hw_params.fx):
        # ------ f parameter calculations
        orig_hout = INIT_START_HOUT_IDX
        end_f_idx, num_f = cls.f_params_calculation(orig_f, hw_params, INIT_END_COUT)

        # -- All channel stat collection
        mac_cycles_all_channels = 0
        util_cycles_all_channels = 0
        padd_util_cycles_all_channels = 0
        padd_cycles_all_channels = 0
        dma_cycles_all_channels = 0
        current_cycles_all_channels = 0
        for orig_cin in range(INIT_START_CIN_IDX, INIT_END_CIN, hw_params.cxx):
            # ------ c parameter calculations
            num_cin, end_cin_idx = cls.c_params_calculation(orig_cin, hw_params, INIT_END_CIN)

            for orig_hin in range(INIT_START_HIN_IDX, INIT_END_HIN, hw_params.hxx):

                # ------  h parameter calculations
                orig_hin, end_hin_idx, end_orig_hout_idx, \
                num_hin, num_h_convs, num_hout = cls.h_params_calculation(orig_hin,layer_attr, hw_params,
                         INIT_START_HIN_IDX, INIT_END_HIN, orig_hout)

                orig_wout = INIT_START_WOUT_IDX
                for orig_win in range(INIT_START_WIN_IDX, INIT_END_WIN, hw_params.wxx):
                    dma_cycles_block = 0
                    padd_cycles_block = 0
                    padd_util_cycles_block = 0
                    # ------  w parameter calculations
                    orig_win, end_orig_wout_idx, end_win_idx, \
                    num_win, num_w_convs,  num_wout = cls.w_params_calculation(orig_win, layer_attr, hw_params,
                                          INIT_START_WIN_IDX, INIT_END_WIN, orig_wout)

                    ###########################################333
                    ##############################################

                    # cls.debug_message(' -- ')
                    # a) P, b) not DP c) PD d) P1 in PDP e) C1 in CC
                    # Global Hin, Win, Cin -- only used for onchip cache indices

                    dma_cycles_block = cls.load_activations_onchip(orig_hin, orig_win, orig_cin,
                                 num_hin, num_win, num_cin, hw_params,
                                 layer_position_idx, layer_attr, dma_cycles_block)

                    # cls.debug_message(' -- ')
                    dma_cycles_block = cls.load_weights_onchip(dma_cycles_block,
                                    orig_cin, orig_f,
                                    end_cin_idx, end_f_idx,
                                    num_f, num_cin,
                                    hw_params, layer_attr)
                    # --------------------------------------
                    # mac utilization
                    # --------------------------------------
                    mac_util_cycles_block, mac_cycles_block, \
                    padd_util_cycles_block, padd_cycles_block = cls.calculate_mac_utilization(num_h_convs, num_w_convs,
                                                            num_cin, num_f, hw_params, layer_attr, num_hin, num_win)

                    # --------------------------------------
                    # Accumulate per filter stats
                    # to be used in P2 in PDP
                    # --------------------------------------
                    util_cycles_all_channels += mac_util_cycles_block
                    mac_cycles_all_channels += mac_cycles_block
                    dma_cycles_all_channels += dma_cycles_block
                    # -- padd logic --
                    # -- for all PCONV/CONV. Since, in P2 of PDP padds can be done as soon as hx*wx*fx
                    # is available. This will be done untile hx*wx*F partial product.
                    # TODO: fix this
                    if orig_cin != 0 and layer_attr.attr_type != 'DW':
                        padd_util_cycles_all_channels += padd_util_cycles_block
                        padd_cycles_all_channels += padd_cycles_block
                    else:
                        padd_util_cycles_block = 0
                        padd_cycles_block = 0

                    cls.debug_message('====')
                    orig_wout = end_orig_wout_idx + 1
                # end w
                orig_hout = end_orig_hout_idx + 1
            # end h

            # --------------------------------------
            # cycles information for P2 in PDP
            # Executes all channels
            # --------------------------------------
            # TODO: Change the conditions to -- if INIT_END_COUT == layer_attr.Cout and INIT_END_CIN == layer_attr.Cin
            # TODO change to orig_cin != 0
            # Stage 2: p2 in PDP and C2 in CC, and P in DP
            # -- padd logic --
            if orig_cin != 0 and layer_attr.attr_type != 'DW':
                cls.insert_max_stats('padd_cycles_max_per_batch', layer_attr.layer_idx,
                                      padd_cycles_all_channels)
                cls.stats['padd_ops'][layer_attr.layer_idx] += num_f * LAYER_WOUT * LAYER_HOUT
                cls.insert_max_stats('mem_partial_product', layer_attr.layer_idx,
                                      num_f * LAYER_WOUT * LAYER_HOUT)

            # cycles information
            current_cycles_all_channels = cls.add_pw_block_stats(layer_attr, dma_cycles_all_channels,
                                                                  util_cycles_all_channels,
                                                                  mac_cycles_all_channels,
                                                                  padd_util_cycles_all_channels,
                                                                  padd_cycles_all_channels)
            current_cycles += current_cycles_all_channels
        # end cin


        #-------------------------
        #-------------------------
        if num_cross_layers == layer_position_idx + 1 and INIT_END_CIN == layer_attr.Cin and INIT_END_COUT == layer_attr.Cout:
            isdma = True
            mem_out_act = num_f * LAYER_WOUT * LAYER_HOUT
            cls.mem_out_act_stats(layer_attr, isdma, mem_out_act, 0, LAYER_HOUT-1, 0,
                                  LAYER_WOUT-1, orig_f, end_f_idx)

        if (num_cross_layers > 1 and layer_position_idx == 0):
            isdma = False
            mem_out_act = num_f * LAYER_WOUT * LAYER_HOUT
            cls.mem_out_act_stats(layer_attr, isdma, mem_out_act, 0, LAYER_HOUT - 1, 0,
                                  LAYER_WOUT - 1, orig_f, end_f_idx)

    # end f
    if num_cross_layers == 1:
        cls.insert_max_stats('global_cycles', layer_attr.layer_idx, current_cycles)
    return current_cycles


