from dnn_schedules.per_layer.hwc_schedule import conv2d_dw
from dnn_schedules.schedule import Schedule
import math
from attrdict import AttrDict

class HWFCSchedule(Schedule):

    def __init__(self, hw_type, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(hw_type, net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'hwfc_schedule'

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

    # pointwise only stores partial products of output  in memory.
    # This implies input activations will be streamed multiple times. Hence, high DMA operations
    # Note this is a bad design, better idea is to store entire input/output activation
    # TODO: what to do with vector adds -- need to have enough to not stall pipeline

def conv2d_pw(cls, layer_attr, hw_params, pw_start_indices=None, num_cross_layers=1,
              layer_position_idx=0):
    current_cycles = 0
    # SET PARTIAL LAYER ATTRIBUTES
    INIT_START_HIN_IDX, INIT_START_WIN_IDX, INIT_START_CIN_IDX, INIT_START_COUT_IDX, \
    INIT_START_HOUT_IDX, INIT_START_WOUT_IDX, \
    INIT_END_HIN, INIT_END_WIN, INIT_END_CIN, INIT_END_COUT = \
        cls.set_partial_layer_attributes(pw_start_indices, layer_attr)

    cls.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
    if num_cross_layers==1:
        cls.init_single_layer(hw_params, layer_attr)


    orig_hout = INIT_START_HOUT_IDX
    for orig_hin in range(INIT_START_HIN_IDX, INIT_END_HIN, hw_params.hxx):
        # ------  h parameter calculations
        orig_hin, end_hin_idx, end_orig_hout_idx, \
        num_hin, num_h_convs, num_hout = cls.h_params_calculation(orig_hin, layer_attr, hw_params,
                                                                   INIT_START_HIN_IDX, INIT_END_HIN, orig_hout)

        orig_wout = INIT_START_WOUT_IDX
        for orig_win in range(INIT_START_WIN_IDX, INIT_END_WIN, hw_params.wxx):
            # ------  w parameter calculations
            orig_win, end_orig_wout_idx, end_win_idx, \
            num_win, num_w_convs, num_wout = cls.w_params_calculation(orig_win, layer_attr, hw_params,
                                                                       INIT_START_WIN_IDX, INIT_END_WIN, orig_wout)

            for orig_f in range(INIT_START_COUT_IDX, INIT_END_COUT, hw_params.fx):
                # ------ f parameter calculations
                end_f_idx, num_f = cls.f_params_calculation(orig_f, hw_params, INIT_END_COUT)

                mac_cycles_all_channels = 0
                util_cycles_all_channels = 0
                padd_util_cycles_all_channels = 0
                padd_cycles_all_channels = 0
                dma_cycles_all_channels = 0

                if layer_attr.attr_type == 'DW':
                    INIT_START_CIN_IDX = orig_f
                    INIT_END_CIN = math.ceil(num_f/layer_attr.Depth_multiplier)

                for orig_cin in range(INIT_START_CIN_IDX, INIT_END_CIN, hw_params.cxx):

                    # ------  c parameter calculations
                    num_cin, end_cin_idx = cls.c_params_calculation(orig_cin, hw_params, INIT_END_CIN)

                    #------------------------------------
                    # ------------------------------------
                    dma_cycles = 0
                    padd_cycles = 0
                    padd_util_cycles = 0

                    # cls.debug_message(' -- ')
                    # a) P, b) not DP c) PD d) P1 in PDP e) C1 in CC
                    # Global Hin, Win, Cin -- only used for onchip cache indices
                    dma_cycles = cls.load_activations_onchip(orig_hin, orig_win, orig_cin,
                                                              num_hin, num_win, num_cin, hw_params,
                                                              layer_position_idx, layer_attr, dma_cycles)

                    # cls.debug_message(' -- ')

                    # -- Load weights --
                    # cls.debug_message(' -- ')
                    dma_cycles = cls.load_weights_onchip(dma_cycles,
                                                          orig_cin, orig_f,
                                                          end_cin_idx, end_f_idx,
                                                          num_f, num_cin,
                                                          hw_params, layer_attr)
                    # if mac_util_cycles/mac_cycles != 1:
                    #     print(' ')
                    # --------------------------------------
                    # mac utilization
                    # --------------------------------------
                    mac_util_cycles, mac_cycles, padd_util_cycles, padd_cycles = cls.calculate_mac_utilization(
                        num_h_convs, num_w_convs, num_cin,
                        num_f, hw_params, layer_attr, num_hin, num_win)

                    # if mac_util_cycles/mac_cycles != 1:
                    #     print(' ')

                    # --------------------------------------
                    # Accumulate per filter stats
                    # to be used in P2 in PDP
                    # --------------------------------------
                    util_cycles_all_channels += mac_util_cycles
                    mac_cycles_all_channels += mac_cycles
                    dma_cycles_all_channels += dma_cycles

                    # -- padd logic --
                    # -- for all PCONV/CONV. Since, in P2 of PDP padds can be done as soon as hx*wx*fx
                    # is available. This will be done untile hx*wx*F partial product.
                    # TODO: fix this
                    if orig_cin != 0 and layer_attr.attr_type != 'DW':
                        padd_util_cycles_all_channels += padd_util_cycles
                        padd_cycles_all_channels += padd_cycles
                    else:
                        padd_util_cycles = 0
                        padd_cycles = 0

                    # -- padd logic --
                    if orig_cin != 0 and layer_attr.attr_type != 'DW':
                        # padd_unit = hw_params.mac_wxx*hw_params.mac_fx
                        cls.insert_max_stats('padd_cycles_max_per_batch', layer_attr.layer_idx,
                                              padd_cycles)
                        cls.stats['padd_ops'][layer_attr.layer_idx] += num_f * num_wout * num_hout
                        # cls.stats['out_sram_act_access'][layer_attr.layer_idx] += num_f * num_wout * num_hout
                        cls.insert_max_stats('mem_partial_product', layer_attr.layer_idx,
                                              num_f * num_wout * num_hout)
                    # --------------------------------------
                    # cycles information for a) P, b) P in DP and c) P1 in PDP
                    # Executes Fx filters (partial output: hx*wx*fx; to be added  cx/cin times
                    # padds: hx*wx*fx; padd_units = mac_wx*mac_fx
                    # --------------------------------------
                    current_cycles_block = cls.add_pw_block_stats(layer_attr, dma_cycles, mac_util_cycles,
                                                                    mac_cycles,
                                                                    padd_util_cycles, padd_cycles)

                    current_cycles += current_cycles_block

                # end cin
                # ------------------------------------------------------------------------------------------------
                # a) P, b) DP c) not PD d) P2 in PDP e) C2 in CC
                if num_cross_layers == layer_position_idx + 1 and INIT_END_CIN == layer_attr.Cin and INIT_END_COUT == layer_attr.Cout:
                    isdma = True
                    mem_out_act = num_f * num_wout * num_hout
                    cls.mem_out_act_stats(layer_attr, isdma, mem_out_act, orig_hout, end_orig_hout_idx, orig_wout,
                                          end_orig_wout_idx, orig_f, end_f_idx)

                # a) not P, b) not DP c) PD d) P1 in PDP 3) C1 in CC
                if (num_cross_layers >1 and layer_position_idx==0):
                    isdma= False
                    mem_out_act = num_f * num_wout * num_hout
                    cls.mem_out_act_stats(layer_attr, isdma,mem_out_act, orig_hout,end_orig_hout_idx,orig_wout,end_orig_wout_idx, orig_f, end_f_idx)

                cls.debug_message('====')
            # ------------------------------------------------------------------------------------------------
            # end f
            orig_wout = end_orig_wout_idx + 1
        # end w
        orig_hout = end_orig_hout_idx + 1
    # end h
    if num_cross_layers == 1:
        cls.insert_max_stats('global_cycles', layer_attr.layer_idx, current_cycles)
    return current_cycles

