import yaml
from attrdict import AttrDict
import copy
from collections import OrderedDict
import csv
import math
from dnn_schedules.onchip_memory import OnchipMem

# hw_type = ['tb','cf_cfhw']
hw_type = ['cf_cfhw', 'tangram']
class Schedule():
    # cwh_sched_0
    def __init__(self, hw_type, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        self.hw_type = hw_type
        assert self.hw_type in hw_type, 'hw_type not supported'
        self.model_name = model_name
        self.net = net
        self.stats = OrderedDict()
        self.debug = verbose
        self.result_dir = result_dir
        self.onchip_mem = OnchipMem()

        if hardware_yaml is not None:
            self.process_hardware_config(hardware_yaml)
        else:
            self.process_hardware_config_dict(hardware_dict)

        if hardware_dict is None and hardware_yaml is None:
            raise Exception('Need to pass either yaml or dict')

        self.add_stats()

    def __str__(self):
        pass

    def debug_message(self, s):
        if self.debug:
            print(s)

    def process_hardware_config_dict(self, hardware_dict):
        self.hw_params = hardware_dict

    def process_hardware_config(self, hardware_yaml):
        self.hw_params = AttrDict(yaml.safe_load(open(hardware_yaml)))

    def run_model(self):
        pass

    def conv2d_dw(self, layer_attr, hw_params):
        pass

    def conv2d_pw(self, layer_attr, hw_params):
        pass
    # -------------------------------------------------------
    # Auxiliary function for statistics
    # -------------------------------------------------------
    def add_stats(self):
        self.stat_list = ['in_dma_act', 'in_dma_wgt', 'out_dma_act',
                          'mem_wgt', 'mem_in_act', 'mem_out_act',
                          'mem_partial_product',
                          'mem_wgt_accesses', 'mem_in_act_accesses', 'mem_out_act_accesses',
                          'mem_partial_product_acceses',
                          'in_rf_act_accesses','out_rf_act_accesses','wgt_rf_accesses',
                          'in_rf_act_size', 'out_rf_act_size', 'wgt_rf_size',
                          'padd_ops', 'padd_cycles_max_per_batch',
                          'padd_util_cycles', 'padd_cycles','padd_units_available','total_padd_units',
                          'mac_util_cycles',
                          'total_mac_units','cycles_max_per_batch',
                          'dma_cycles',
                          # For cross layer
                          'mac_units_available',
                          'is_dma_cycle_selected', 'is_mac_cycle_selected', 'mac_cycles',
                          'timing_cycles', 'cycles_total', 'global_cycles'
                          ]
                          # Note: out_sram_access is only for last layer in any pipeline.

        self.layer_names = []

        for stat in self.stat_list:
            self.stats[stat] = [0]*self.net.num_layers

    def insert_max_stats(self, key, idx, new_stat):
        prev_stat = self.stats[key][idx]
        self.stats[key][idx] = max(prev_stat, new_stat)

    def print_stats(self):
        # write params
        params = self.hw_params.HWConfig
        param_names = ''
        param_list = [0] * self.net.num_layers
        file_name_suffix = ''
        idx = 0
        for k, v in params.items():
            param_names += '-' + str(k)
            file_name_suffix += '-' + str(v)
            if idx >= self.net.num_layers:
                param_list.append(v)
            else:
                param_list[idx] = v
            idx += 1

        param_string = param_names + ',' + ','.join([str(i) for i in param_list])

        # write stats
        with open(self.result_dir + self.model_name + '_' + self.__str__() + file_name_suffix +'.csv', 'w') as f:
            for key,value in self.stats.items():
                if type(value[0]) != int:
                    value = [ int(v) for v in value]
                if self.debug:
                    print(key, value)
                val_list = ','.join([str(i) for i in value])
                row = '{},{}\n'.format(key, val_list)
                f.write(row)

            if self.debug:
                row = ','.join([str(i) for i in self.layer_names])
                f.write(row)

            f.write(param_string)
            f.close()

    # -------------------------------------------------------
    # Load Layer attributes per layer
    # -------------------------------------------------------
    def load_hw_params_depthwise(self):
        params = self.hw_params.HWConfig
        hw_params = AttrDict({'cx': params.cx, 'cxx': params.cx,
                              'wx': params.wx, 'wxx': params.wx,
                              'hx': params.hx, 'hxx': params.hx,
                              'fx': 1, # should be equal to depth_multiplier
                              'mac_wx': params.mac_wx, 'mac_wxx':params.mac_wx,
                              'mac_wx_type': params.mac_wx_type, 'mac_wxx_type': params.mac_wx_type,
                              'mac_cx': params.mac_cx, 'mac_cxx': params.mac_cxx,
                              'dma_cycles': params.dma_cycles})

        return hw_params

    def load_hw_params_pointwise(self, is_first, is_single):
        params = self.hw_params.HWConfig
        if is_single:
            hw_params = AttrDict({ 'hxx': params.hxx + params.hxx2, 'wxx': params.wxx + params.wxx2,
                                   'cxx': params.cxx + params.cxx2, 'fx': params.fx + params.fx2,
                                   'mac_wxx': params.mac_wxx + params.mac_wxx2,
                                   'mac_wxx_type': params.mac_wxx_type,
                                   'mac_cxx': params.mac_cxx + params.mac_cxx2,
                                   'mac_fx': params.mac_fx + params.mac_fx2,
                                   'dma_cycles': params.dma_cycles
                                   # 'padd_cycles': params.padd_cycles,
                                   # 'padd_unit': params.padd_unit
                                  })
        else:
            if is_first:
                hw_params = AttrDict({ 'hxx': params.hxx, 'wxx': params.wxx, 'cxx': params.cxx, 'fx': params.fx,
                                      'mac_wxx': params.mac_wxx, 'mac_wxx_type': params.mac_wxx_type,
                                      'mac_cxx': params.mac_cxx,'mac_fx': params.mac_fx,
                                      'dma_cycles': params.dma_cycles
                                      # 'padd_cycles': params.padd_cycles,
                                      # 'padd_unit': params.padd_unit
                                      })
            else:
                hw_params = AttrDict({ 'hxx': params.hxx2, 'wxx': params.wxx2, 'cxx': params.cxx2, 'fx': params.fx2,
                                       'mac_wxx': params.mac_wxx2,  'mac_wxx_type': params.mac_wxx2_type,
                                       'mac_cxx': params.mac_cxx2, 'mac_fx': params.mac_fx2,
                                       'dma_cycles': params.dma_cycles
                                        # 'padd_cycles': params.padd_cycles2 ,
                                        # 'padd_unit': params.padd_unit2
                                    })

        return hw_params

    # config == 0(Tensorbricks), 1(fuselayer), 2(eyersiss), 3(eyeriss2),
    def load_hw_params_conv(self, is_first, is_single, config=0):
        params = self.hw_params.HWConfig
        if is_first:
            if config == 0:
                # Assuming a standard kernel (3x3) => for CONV we need wxx - 3+1 = (wxx -2) mac units.
                # If wxx >=3, mac_wx = 1
                if is_single:
                    hw_params = AttrDict({'hxx': params.hxx + params.hxx2, 'wxx': params.wxx + params.wxx2,
                                          'cxx': params.cxx + params.cxx2, 'fx': params.fx + params.fx2,
                                          'mac_wxx': params.mac_wxx + params.mac_wxx2,
                                          'mac_wxx_type': params.mac_wxx_type,
                                          'mac_cxx': params.mac_cxx + params.mac_cxx2,
                                          'mac_fx': params.mac_fx + params.mac_fx2,
                                          'dma_cycles': params.dma_cycles
                                          # 'padd_cycles': params.padd_cycles,
                                          # 'padd_unit': params.padd_unit
                                          })

                else:
                    hw_params = AttrDict({'hxx': params.hxx, 'wxx': params.wxx, 'cxx': params.cxx, 'fx': params.fx,
                                          'mac_wxx': params.mac_wxx,  'mac_wxx_type': params.mac_wxx_type,
                                          'mac_cxx': params.mac_cxx, 'mac_fx': params.mac_fx,
                                          'dma_cycles': params.dma_cycles
                                          # 'padd_cycles': params.padd_cycles,
                                          # 'padd_unit': params.padd_unit
                                          })
            elif config==1:
                hw_params = AttrDict(
                    {'cxx': params.cxx, 'fx': params.fx,
                     'dma_cycles': params.dma_cycles
                     # 'padd_cycles': params.padd_cycles,
                     # 'padd_unit': params.padd_unit
                     })
            elif config==2:
                hw_params = AttrDict(
                    {'cxx': params.cxx, 'fx': params.fx,
                     'mac_wxx': params.mac_wxx,  'mac_wxx_type': params.mac_wxx_type,
                     'dma_cycles': params.dma_cycles
                     # 'padd_cycles': params.padd_cycles,
                     # 'padd_unit': params.padd_unit
                     })
                hw_params['X'] = params.X
                hw_params['Y'] = params.Y
            elif config==3:
                hw_params = AttrDict({'cxx': params.cxx, 'fx': params.fx,
                                      'mac_wxx': params.mac_wxx,  'mac_wxx_type': params.mac_wxx_type,
                                      'mac_cxx': params.mac_cxx, 'mac_fx': params.mac_fx,
                                      'dma_cycles': params.dma_cycles
                                      })
            elif config == 4:
                # This one is for fire layers
                # Assuming a standard kernel (3x3) => for CONV we need wxx - 3+1 = (wxx -2) mac units.
                # If wxx >=3, mac_wx = 1
                hw_params = AttrDict({'hxx': params.hxx3, 'wxx': params.wxx3, 'cxx': params.cxx3, 'fx': params.fx3,
                                      'mac_wxx': params.mac_wxx3,  'mac_wxx_type': params.mac_wxx3_type,
                                      'mac_cxx': params.mac_cxx3, 'mac_fx': params.mac_fx3,
                                      'dma_cycles': params.dma_cycles
                                      })
            else:
                raise  ValueError('config param is not set')

        else:
            if config == 0:
                hw_params = AttrDict({'hxx': params.hxx2, 'wxx': params.wxx2, 'cxx': params.cxx2, 'fx': params.fx2,
                                      'mac_wxx': params.mac_wxx2,  'mac_wxx_type': params.mac_wxx2_type,
                                      'mac_cxx': params.mac_cxx2, 'mac_fx': params.mac_fx2,
                                      'dma_cycles': params.dma_cycles
                                      # 'padd_cycles': params.padd_cycles2 ,
                                      # 'padd_unit': params.padd_unit2
                                      })
            elif config == 1:
                hw_params = AttrDict(
                    {'cxx': params.cxx2, 'fx': params.fx2,
                     'dma_cycles': params.dma_cycles,
                     # 'padd_unit': params.padd_unit2
                     })

            elif config==2:
                hw_params = AttrDict(
                    {'cxx': params.cxx2, 'fx': params.fx2,
                     'dma_cycles': params.dma_cycles,
                     # 'padd_cycles': params.padd_cycles,
                     # 'padd_unit': params.padd_unit2
                     })
                hw_params['X'] = params.XX
                hw_params['Y'] = params.YY
            elif config==3:
                hw_params = AttrDict({'cxx': params.cxx2, 'fx': params.fx2,
                                      'mac_wxx': params.mac_wxx2,  'mac_wxx_type': params.mac_wxx2_type,
                                      'mac_cxx': params.mac_cxx2, 'mac_fx': params.mac_fx2,
                                      'dma_cycles': params.dma_cycles
                                      })

            elif config == 4:
                # This one is for fire layers
                # Assuming a standard kernel (3x3) => for CONV we need wxx - 3+1 = (wxx -2) mac units.
                # If wxx >=3, mac_wx = 1
                hw_params = AttrDict({'hxx': params.hxx3, 'wxx': params.wxx3, 'cxx': params.cxx3, 'fx': params.fx3,
                                      'mac_wxx': params.mac_wxx3,  'mac_wxx_type': params.mac_wxx3_type,
                                      'mac_cxx': params.mac_cxx3, 'mac_fx': params.mac_fx3,
                                      'dma_cycles': params.dma_cycles
                                      })

            else:
                raise  ValueError('config param is not set')

        return hw_params

    # -------------------------------------------------------
    # -- Calculations for mac util, global cycles and params
    # -------------------------------------------------------
    def add_pw_block_stats(self, layer_attr, dma_cycles, mac_util_cycles, mac_cycles, padd_util_cycles, padd_cycles):
        batch_cycles  = 0
        current_batch_cycles = mac_cycles + padd_cycles
        self.stats['dma_cycles'][layer_attr.layer_idx] += dma_cycles
        self.stats['mac_cycles'][layer_attr.layer_idx] += mac_cycles
        self.stats['mac_util_cycles'][layer_attr.layer_idx] += mac_util_cycles
        self.stats['padd_cycles'][layer_attr.layer_idx] += padd_cycles
        self.stats['padd_util_cycles'][layer_attr.layer_idx] += padd_util_cycles

        self.insert_max_stats('cycles_max_per_batch', layer_attr.layer_idx, current_batch_cycles)

        # if dma cost is higher then add dma cycles
        if dma_cycles > current_batch_cycles:
            self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
            self.stats['cycles_total'][layer_attr.layer_idx] += dma_cycles
            batch_cycles = dma_cycles
        else:
            self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
            self.stats['cycles_total'][layer_attr.layer_idx] += current_batch_cycles
            batch_cycles = current_batch_cycles

        return batch_cycles

    def get_mac_utilization(self, num_convs, num_macs):
        fold = math.ceil(num_convs/num_macs)
        mac_utilization = num_convs / (fold * num_macs)
        return mac_utilization, fold

    def calculate_mac_utilization(self, num_h_convs, num_w_convs,
                                  num_cin, num_f, hw_params, layer_attr, num_hin, num_win, dataflow='cf_cfhw'):
        # if self.hw_type == 'cf_hwcf':
        #     return self.calculate_mac_utilization_cf_hwcf(num_h_convs, num_w_convs,
        #                           num_cin, num_f, hw_params, layer_attr, num_hin, num_win)
        if self.hw_type == 'cf_cfhw':
            return self.calculate_mac_utilization_cf_cfhw(num_h_convs, num_w_convs,
                                  num_cin, num_f, hw_params, layer_attr, num_hin, num_win)
        elif self.hw_type == 'tb':
            return self.calculate_mac_utilization_tensorbricks(num_h_convs, num_w_convs, num_cin, num_f, hw_params, layer_attr)
        elif self.hw_type == 'tangram':
            return self.calculate_mac_utilization_cf_cfhw(num_h_convs, num_w_convs,
                                  num_cin, num_f, hw_params, layer_attr, num_hin, num_win)

    # Systolic Implementation: Weight Stationary C|F dataflow with C|F|HW inner loop for pointwise and CONV.
    def calculate_mac_utilization_cf_cfhw(self, num_h_convs, num_w_convs,
                                          num_cin, num_f, hw_params, layer_attr, num_hin, num_win):
        # is_conv_layer = True if layer_attr.Ky != 1 else False

        systolic_dim = hw_params.mac_wxx_type
        available_engines = hw_params.mac_wxx
        c_fold = math.ceil(num_cin / systolic_dim)
        # -------------------------------------------------
        # New calculations: mac util and padd util
        # -------------------------------------------------
        # # Microarchitecture: C|F dataflow --inner dataflow = K|C|XY, Engines stacked on top of each other.
        # As engines increase, CXY dimension computation increases,
        # but number of filters is not affected and is same for each engine
        # # Rows = avail_engines*per_engine_row, # Cols= per_engine_column

        num_hw_rows = systolic_dim*available_engines
        num_hw_cols  = systolic_dim
        per_col_computation = layer_attr.Kx*layer_attr.Ky*num_cin

        row_fold = math.ceil(per_col_computation/num_hw_rows)
        col_fold = math.ceil(num_f/num_hw_cols)
        # Cycles = filter load + num_XY_convs + min(X,rem_KxKyCin_batch) + min(Y,rem_F_batch)
        # Assumption: Each engine starts loading simultaneosly
        filter_load_cycles = systolic_dim
        ipact_load_cycles = row_fold*systolic_dim
        conv_cycles = num_h_convs*num_w_convs
        opact_store_cycles = col_fold*num_hw_cols
        mac_cycles =  filter_load_cycles + ipact_load_cycles + conv_cycles  + opact_store_cycles

        # ky_h_conv_cycles = 1+ layer_attr.Ky * layer_attr.Kx + (num_h_convs-1)*layer_attr.Ky + systolic_dim
        # w_conv_h_conv_cycles = num_w_convs * ky_h_conv_cycles
        # mac_cycles = math.ceil(num_cin/systolic_dim)*math.ceil(num_f/systolic_dim) * w_conv_h_conv_cycles


        # For 1 Conv all filters:
        # util = ((fold-1)*1.0 + 1*util_last_iteration)/fold
        row_util = ((row_fold-1)*1.0 + 1.0*(per_col_computation%num_hw_rows)/num_hw_rows)/row_fold
        col_util = ((col_fold-1)*1.0 +  1.0*(num_f%num_hw_cols)/num_hw_cols)/col_fold
        mac_util = round(row_util*col_util,2)

        # mac utilization for num_h_convs*num_w_convs
        mac_util_cycles = mac_util * mac_cycles
        # Assumption runs in parallel.
        padd_cycles = 1
        # num_padds = 1x systolic_dim
        # For 1 CONV, num_col filters, you get 1x1xnum_col-> ofmaps out.
        # Thus, for all convs.
        padd_util = col_util
        padd_util_cycles = padd_util*padd_cycles
        # --------------------

        # -------------------------------------------------
        # OLD Calculation: mac util and padd util
        # -------------------------------------------------
        #  HW cycles
        # a) broadcast Kh*Kw act to all filters = 1 cycles,
        # b) partial adds along the filters to collect the result = mapped_cx cycles; Considering upper limit.
        # c) num_h_convs -> First batch takes Kx*Ky, afterwards = Ky cycles
        # ky_h_conv_cycles = 1+ layer_attr.Ky * layer_attr.Kx + (num_h_convs-1)*layer_attr.Ky + systolic_dim
        # w_conv_h_conv_cycles = num_w_convs * ky_h_conv_cycles
        # mac_util = (num_cin * num_f) / (engine_fold * systolic_dim * systolic_dim)
        # mac_cycles = engine_fold * w_conv_h_conv_cycles
        # mac_util_cycles = mac_util * mac_cycles
        # padd_cycles = 1  # Assumption, it runs in parallel with MACs.
        # padd_util = w_conv_h_conv_cycles / mac_cycles
        # padd_util_cycles = padd_util * padd_cycles
        # -------------------------------------------------


        # -------------------------------------------------
        # Calculate RF accesses and SRAM accesses
        # -------------------------------------------------
        w_accesses = num_win + (num_w_convs - 1) * (layer_attr.Ky - layer_attr.Sy)
        h_accesses = num_hin

        # From in act SRAM to RF.
        in_rf_act_from_sram = h_accesses * w_accesses * num_cin
        self.stats['mem_in_act_accesses'][layer_attr.layer_idx] += in_rf_act_from_sram  # store to in_rf

        # Kx*Ky*Cx act
        in_rf_act_tile_size = layer_attr.Kx * layer_attr.Ky * available_engines * systolic_dim
        # Reuse is Fx in space and in time = (num_h_convs - 1) * (layer_attr.Kx - layer_attr.Sx)
        in_rf_act_total_reuse_time = (num_h_convs - 1) * (layer_attr.Kx - layer_attr.Sx)
        in_rf_act_reuse_size = num_w_convs*num_cin*num_f

        self.stats['in_rf_act_accesses'][layer_attr.layer_idx] += in_rf_act_total_reuse_time*in_rf_act_reuse_size + in_rf_act_from_sram
        in_rf_tile_size =  available_engines * systolic_dim
        self.insert_max_stats('in_rf_act_size', layer_attr.layer_idx, in_rf_tile_size)

        # From in wgt SRAM to RF
        # Since, 3*3*Cx activations iterate in Cx dimensions. (Fx,3*3*Cx)
        # weights need to be read (num_h_convs*num_w_conv) times.
        in_rf_wgt_from_sram = num_f * layer_attr.Kx * layer_attr.Ky * num_cin * num_w_convs
        mem_wgt_total_reuse_time = num_h_convs
        mem_wgt_total_reuse_size = num_f*layer_attr.Kx*layer_attr.Ky*num_cin

        self.stats['wgt_rf_accesses'][layer_attr.layer_idx] += in_rf_wgt_from_sram + mem_wgt_total_reuse_time*mem_wgt_total_reuse_size

        # For each 3*3*Cx activations, Fx*3*3*Cx wgts are required.
        mem_wgt_tile_size = available_engines * systolic_dim * systolic_dim * layer_attr.Kx * layer_attr.Ky
        self.insert_max_stats('wgt_rf_size', layer_attr.layer_idx, mem_wgt_tile_size)

        # From RF to out act SRAM
        out_rf_act_to_sram = num_h_convs * num_w_convs * num_f
        self.stats['mem_out_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram


        out_rf_act_total_reuse_time =  (c_fold-1)
        out_rf_act_total_reuse_size =  num_h_convs * num_w_convs * num_f
        out_rf_tile_size = systolic_dim * available_engines  # (1xFx)
        self.stats['out_rf_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram + out_rf_act_total_reuse_size*out_rf_act_total_reuse_time
        self.insert_max_stats('out_rf_act_size', layer_attr.layer_idx, out_rf_tile_size)

        return mac_util_cycles, mac_cycles, padd_util_cycles, padd_cycles

    # # Systolic Implementation: output stationary C|F dataflow with HW|C|F inner loop for pointwise and CONV
    # # 3*3*Cx act -- iterates in C dimension, wgts Fx,3*3*Cx; are streaming from SRAM to RF.
    # # Activations 3*3*Cx are reused across filters and
    # # padds [1*1*Fx] is stationary.
    # # wgts and activations are streaming. Fx, K*K*Cx weights are sitting in SRAM.
    # def calculate_mac_utilization_cf_hwcf(self, num_h_convs, num_w_convs,
    #                               num_cin, num_f, hw_params, layer_attr, num_hin, num_win):
    #     is_conv_layer = True if layer_attr.Ky != 1 else False
    #     # NN_engines*Hx*Cx,
    #     # For Hx -- can have 3 --7 can be fixed depending on a) the latency to read from SRAM,
    #     # and b) the number of bytes in each row of SRAM.
    #     # Can safely assume 4? As 16*4 = 64 bytes = 1 row of SRAM.
    #     ONE_SRAM_ROW = 4 # Assuming 4*16=64 bytes.
    #
    #     systolic_dim = hw_params.mac_wxx_type
    #     available_engines = hw_params.mac_wxx
    #     # num_padd_engines = available_engines*systolic_dim # NN_engines*(Fx) padds in parallel.
    #     c_fold = math.ceil(num_cin/systolic_dim)
    #     f_fold = math.ceil(num_f/systolic_dim)
    #     required_engines = c_fold*f_fold
    #
    #
    #     # -------------------------------------------------
    #     # Calculate Mac cycles and padd cycles
    #     #-------------------------------------------------
    #     #  HW cycles
    #     w_conv_h_conv_cycles = num_w_convs * num_h_convs*layer_attr.Ky*layer_attr.Kx
    #     if available_engines >= required_engines:
    #         mac_util = systolic_dim * systolic_dim * available_engines / (num_cin * num_f)
    #         mac_cycles = available_engines * systolic_dim * systolic_dim * w_conv_h_conv_cycles
    #
    #     else:
    #         engine_fold = math.ceil(required_engines / available_engines)
    #         mac_util = (layer_attr.cxx * layer_attr.fx) / (engine_fold * systolic_dim * systolic_dim)
    #         mac_cycles = engine_fold * available_engines * systolic_dim * systolic_dim * w_conv_h_conv_cycles
    #
    #     mac_util_cycles = mac_util * mac_cycles
    #     padd_cycles = 1 # Since, it is output stationary, No overhead is incurred.
    #     padd_util = w_conv_h_conv_cycles / mac_cycles
    #     padd_util_cycles = padd_util * padd_cycles
    #
    #
    #     # -------------------------------------------------
    #     # Calculate RF accesses and SRAM accesses
    #     #-------------------------------------------------
    #     w_accesses =  num_win + (num_w_convs-1)*(layer_attr.Ky - layer_attr.Sy)
    #     h_accesses = num_hin + (num_h_convs-1)*(layer_attr.Kx - layer_attr.Sx)
    #
    #     # From in act SRAM to RF.
    #     in_rf_act_from_sram = h_accesses*w_accesses*num_cin
    #     self.stats['mem_in_act_accesses'][layer_attr.layer_idx] += in_rf_act_from_sram # store to in_rf
    #
    #     # Kx*Ky*Cx act
    #     in_rf_act_tile_size = layer_attr.Kx*layer_attr.Ky*available_engines*systolic_dim
    #     # Reuse is Fx but the value is broadcasted to Fx filters and read only once.
    #     # in_rf_act_reuse_per_tile = 1
    #     # Since, reuse is once in time, whatever is stored is read only once.
    #     in_rf_act_accesses = in_rf_act_from_sram
    #
    #
    #     # For every Ky*Kx*Cx in inRF --> it is read only once, but reused across Fx filters.
    #     in_rf_per_tile_accesses= layer_attr.Kx*layer_attr.Ky*available_engines*systolic_dim # Loads from in_rf
    #     self.stats['in_rf_act_accesses'][layer_attr.layer_idx] += in_rf_act_accesses + in_rf_act_from_sram
    #     self.insert_max_stats('in_rf_act_size', layer_attr.layer_idx, available_engines * systolic_dim)
    #
    #
    #
    #
    #
    #     # From in wgt SRAM to RF
    #     # Since, 3*3*Cx activations iterate in Cx dimensions. (Fx,3*3*Cx)
    #     # weights need to be read (num_h_convs*num_w_conv) times.
    #     in_rf_wgt_from_sram = num_f * layer_attr.Kx * layer_attr.Ky * num_cin * (num_h_convs * num_w_convs)
    #
    #     # For each 3*3*Cx activations, Fx*3*3*Cx wgts are required.
    #     mem_wgt_tile_size =  available_engines * systolic_dim*systolic_dim*layer_attr.Kx*layer_attr.Ky
    #
    #     # Since, reuse is once, whatever is stored is read only once.
    #     # mem_wgt_reuse_per_tile = 1
    #     # mem_wgt_num_tiles = c_fold*f_fold
    #
    #     self.stats['wgt_rf_accesses'][layer_attr.layer_idx] += in_rf_wgt_from_sram
    #     self.insert_max_stats('wgt_rf_size', layer_attr.layer_idx, mem_wgt_tile_size)
    #
    #
    #     # From RF to out act SRAM
    #     out_rf_act_to_sram = num_h_convs*num_w_convs*num_f # output stationary
    #     self.stats['mem_out_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram
    #     # 1 tile is hit (c_fold -1) times
    #     out_rf_tile_size = systolic_dim*available_engines # (1xFx)
    #     out_rf_reuse_per_tile = (c_fold-1)
    #     out_rf_num_tiles = num_w_convs*num_h_convs
    #     out_rf_accesses = out_rf_tile_size*out_rf_reuse_per_tile*out_rf_num_tiles
    #     # For every Ky*Kx*Cx in inRF --> 1*Fx is generated in outRF, which is updated (c_fold-1) times,
    #     # and then written back to SRAM
    #     self.stats['out_rf_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram + out_rf_accesses
    #     self.insert_max_stats('out_rf_act_size', layer_attr.layer_idx, out_rf_tile_size)
    #
    #     return mac_util_cycles, mac_cycles, padd_util_cycles, padd_cycles

    def calculate_mac_utilization_tensorbricks(self, num_h_convs, num_w_convs, num_cin, num_f, hw_params, layer_attr):

        # If CONV or PW
        b_W = 1.0*hw_params.mac_wxx_type/layer_attr.Ky
        assert b_W >= 1, 'raise mac_wxx_type. Since, mac_wxx_type ({}) < Ky ({}). ' \
                         'Cannot apply 2D-CONV in one cycle'.format(hw_params.mac_wxx_type, layer_attr.Ky)
        num_macs_w = hw_params.mac_wxx * math.ceil(b_W)
        w_util, w_fold = self.get_mac_utilization(num_w_convs, num_macs_w)
        # Say its a 3x3 CONV, then for PW it can do 3 CONVs in a cycle
        b_H = 1.0*hw_params.mac_wxx_type/layer_attr.Kx
        assert b_H >= 1, 'raise mac_wxx_type. Since, mac_wxx_type ({}) < Kx ({}). ' \
                         'Cannot apply 2D-CONV in one cycle'.format(hw_params.mac_wxx_type, layer_attr.Kx)
        num_macs_h = math.ceil(b_H)
        h_util, h_fold = self.get_mac_utilization(num_h_convs, num_macs_h)

        cin_util, cin_fold = self.get_mac_utilization(num_cin, hw_params.mac_cxx)
        f_util, f_fold = self.get_mac_utilization(num_f, hw_params.mac_fx)

        mac_cycles = h_fold*w_fold*cin_fold*f_fold
        mac_util_cycles = (h_util*w_util*cin_util*f_util)*mac_cycles

        # Padd calculation
        # Note: It will be different for systollic
        # TODO: Fix padd_cycles and padd_utilization
        padd_cycles = f_fold * w_fold * h_fold
        if layer_attr.attr_type == 'PW':
            padd_utilization = h_util*w_util * f_util
        elif layer_attr.attr_type == '3d':
            padd_utilization = f_util *(num_h_convs*num_w_convs)/\
                               ((w_fold * h_fold)*(hw_params.mac_wxx *hw_params.mac_wxx_type*hw_params.mac_wxx_type))
        elif layer_attr.attr_type == 'DW':
            padd_utilization = 0
            padd_cycles = 0
        else:
            raise ValueError('Unsupported layer')

        padd_util_cycles = padd_utilization*padd_cycles

        return mac_util_cycles, mac_cycles, padd_util_cycles, padd_cycles

    def calculate_dw_mac_utilization(self, num_h_convs, num_w_convs, num_cin, num_cout, hw_params, layer_attr, num_hin, num_win):
        # TODO: change this api.. Currently, Eyeriss-v2 DW used for c|f hw_type everywhere.
        if self.hw_type == 'cf_cfhw':
            return self.calculate_dw_mac_utilization_kyCW_chw(num_h_convs, num_w_convs, num_cin, num_cout, hw_params, layer_attr, num_hin, num_win)
        elif self.hw_type == 'tb':
            return self.calculate_dw_mac_utilization_tensorbricks(num_h_convs, num_w_convs, num_cin, hw_params, layer_attr)
        elif self.hw_type == 'tangram':
            return self.calculate_dw_mac_utilization_cf_cfhw(num_h_convs, num_w_convs, num_cin, num_cout, hw_params, layer_attr, num_hin, num_win)

    def calculate_dw_mac_utilization_tensorbricks(self, num_h_convs, num_w_convs, num_cin, hw_params, layer_attr):

        # Note: even if cx != 1, mac utilization is same across channels.
        # Since, the hardware is symmetrical across channels.
        # If CONV or PW
        b_W = 1.0 * hw_params.mac_wx_type / layer_attr.Ky
        assert b_W >= 1, 'raise mac_wx_type. Since, mac_wx_type ({}) < Ky ({}). ' \
                         'Cannot apply 2D-CONV in one cycle'.format(hw_params.mac_wx_type, layer_attr.Ky)
        num_macs_w = hw_params.mac_wx * math.ceil(b_W)
        # Say its a 3x3 CONV, then for PW it can do 3 CONVs in a cycle
        b_H = 1.0 * hw_params.mac_wx_type / layer_attr.Kx
        assert b_H >= 1, 'raise mac_wxx_type. Since, mac_wxx_type({}) < Kx ({}). ' \
                         'Cannot apply 2D-CONV in one cycle'.format(hw_params.mac_wx_type, layer_attr.Kx)
        num_macs_h = math.ceil(b_H)

        # -----------
        w_util, w_fold = self.get_mac_utilization(num_w_convs, num_macs_w)
        h_util, h_fold = self.get_mac_utilization(num_h_convs, num_macs_h)
        cin_util, cin_fold = self.get_mac_utilization(num_cin, hw_params.mac_cx)

        mac_cycles = h_fold * w_fold * cin_fold * layer_attr.Depth_multiplier
        mac_util_cycles = (h_util * w_util * cin_util) * mac_cycles

        return mac_util_cycles, mac_cycles

    # Systolic implementation KyC|W, C replicated similar to Eyeriss-v2 dataflow.
    # Multiple Channels to one Engine. Weight stationary, C|HW dataflow
    def calculate_dw_mac_utilization_kyCW_chw(self, num_h_convs, num_w_convs, num_cin, num_cout, hw_params, layer_attr, num_hin, num_win):

        systolic_dim = hw_params.mac_wx_type
        available_engines = hw_params.mac_wx
        depth_multiplier = 1.0*num_cout/num_cin
        # -------------------------------------------------
        # New calculations: mac util
        # -------------------------------------------------
        num_hw_rows = available_engines*systolic_dim # Y
        num_hw_cols = systolic_dim # X
        assert num_hw_cols >= layer_attr.Kx, 'systolic engine X={} is smaller than Kx:{}'.format(
            num_hw_cols,layer_attr.Kx)
        assert num_hw_rows >= layer_attr.Ky, 'systolic engine Y={} is smaller than Ky:{}'.format(
            num_hw_rows, layer_attr.Ky)

        per_col_computation = layer_attr.Ky*num_cin
        row_fold = math.ceil(per_col_computation /num_hw_rows)
        # if one Kx per engine [Inefficient, will never occur due to assertion above]
        if layer_attr.Kx > num_hw_cols:
            col_fold_per_iter = math.ceil(layer_attr.Kx/num_hw_cols)
        # else: multiple Kx per engine
        else:
            col_fold_per_iter = math.ceil(num_hw_cols/layer_attr.Kx)

        col_fold = math.ceil(num_h_convs/col_fold_per_iter)
        # For example, Kx=3 and HW_col=7, it can only fit 2*Kx at a time.
        # Thus, it is different compared to C|F dataflow.
        per_row_computation = math.floor(num_hw_cols/layer_attr.Kx)*layer_attr.Kx
        # total num_h_conv times the HW columns will be filled.
        last_iter_per_row_computation = num_h_convs - (col_fold-1)*col_fold_per_iter

        # util = ((fold-1)*1.0 + 1*util_last_iteration)/fold
        row_util = ((row_fold-1)*1.0 + 1.0*(per_col_computation%num_hw_rows)/num_hw_rows)/row_fold
        # util = ((fold-1)*per_iter_util + 1*util_last_iteration)/fold
        col_util = (1.0*(col_fold-1)*per_row_computation/num_hw_cols +  1.0*last_iter_per_row_computation/num_hw_cols)/col_fold
        # Assumption is Cin==Cout or depth_multiplier == 1; then mac_util = round(row_util*col_util,2)
        # For depth_multiplier !=1, mac_util = depth_multiplier*mac_util
        mac_util = round(depth_multiplier*row_util*col_util,2)
        # -------------------------------------------------
        # Calculate Mac cycles and padd cycles
        # -------------------------------------------------

        # Can be assumed as a single big Systolic engine.

        # c_fold = math.ceil(num_cin*layer_attr.Ky / num_hw_rows)
        # num_cin_per_tile = max(math.floor(num_hw_rows/layer_attr.Ky),num_cin)
        #  HW cycles
        # a) broadcast Kh*Kw act to all filters = 1 cycles,
        # b) partial adds along the filters to collect the result = mapped_cx cycles; Considering upper limit.
        # c) num_h_convs -> First batch takes Kx*Ky, afterwards = Ky cycles

        # ky_kx_conv_cycles = num_h_convs  # for Kx*Ky macs
        # # For Ky*X macs
        # hw_h_fold = math.floor(num_hw_cols/layer_attr.Kx)
        # # h_fold wgt copies of (Kx*Ky) for each channel in parallel.
        # ky_h_conv_cycles = math.ceil(ky_kx_conv_cycles/hw_h_fold) + hw_h_fold
        # w_h_conv_cycles = num_w_convs * ky_h_conv_cycles
        # c_w_h_conv_cycles = w_h_conv_cycles*row_fold
        # mac_cycles  = c_w_h_conv_cycles
        # y_util =  (num_cin*layer_attr.Ky)/(Y*c_fold)
        # x_util = (X - X%layer_attr.Kx)/X
        # mac_util = x_util*y_util # ( num_w_convs*num_cin * layer_attr.Ky/Y)
        # #/ (engine_fold * systolic_dim * systolic_dim)

        filter_load_cycles = row_fold* systolic_dim
        ipact_load_cycles = systolic_dim
        conv_cycles = col_fold*row_fold
        opact_store_cycles = row_fold*systolic_dim
        mac_cycles = filter_load_cycles + ipact_load_cycles + conv_cycles + opact_store_cycles
        mac_util_cycles = mac_util * mac_cycles

        # -------------------------------------------------
        # Calculate RF accesses and SRAM accesses
        # -------------------------------------------------
        w_accesses = num_win + (num_w_convs - 1) * (layer_attr.Ky - layer_attr.Sy)
        h_accesses = num_hin

        # From in act SRAM to RF.
        in_rf_act_from_sram = h_accesses * w_accesses * num_cin
        self.stats['mem_in_act_accesses'][layer_attr.layer_idx] += in_rf_act_from_sram  # store to in_rf

        # Ky*systolic_dim*available_engines act
        # Reuse is time = (num_h_convs - 1) * (layer_attr.Kx - layer_attr.Sx)
        in_rf_act_total_reuse_time = (num_h_convs - 1) * (layer_attr.Kx - layer_attr.Sx)
        in_rf_act_reuse_size = num_w_convs*num_cin*layer_attr.Depth_multiplier

        self.stats['in_rf_act_accesses'][layer_attr.layer_idx] += in_rf_act_total_reuse_time*in_rf_act_reuse_size + in_rf_act_from_sram
        in_rf_tile_size =  row_fold * layer_attr.Kx*systolic_dim
        self.insert_max_stats('in_rf_act_size', layer_attr.layer_idx, in_rf_tile_size)

        # From in wgt SRAM to RF
        # Since, 3*3*Cx activations iterate in Cx dimensions. (Fx,3*3*Cx)
        # weights need to be read (num_h_convs*num_w_conv) times.
        in_rf_wgt_from_sram = layer_attr.Depth_multiplier * layer_attr.Kx * layer_attr.Ky * num_cin * num_w_convs
        mem_wgt_total_reuse_time = num_h_convs
        mem_wgt_total_reuse_size = layer_attr.Depth_multiplier *layer_attr.Kx*layer_attr.Ky*num_cin

        self.stats['wgt_rf_accesses'][layer_attr.layer_idx] += in_rf_wgt_from_sram + mem_wgt_total_reuse_time*mem_wgt_total_reuse_size

        # For each 3*3*Cx activations, Fx*3*3*Cx wgts are required.
        mem_wgt_tile_size = row_fold * layer_attr.Ky*layer_attr.Kx
        self.insert_max_stats('wgt_rf_size', layer_attr.layer_idx, mem_wgt_tile_size)

        # From RF to out act SRAM
        out_rf_act_to_sram = num_h_convs * num_w_convs * num_cin*layer_attr.Depth_multiplier
        self.stats['mem_out_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram


        out_rf_act_total_reuse_time =  0
        out_rf_act_total_reuse_size =  0
        out_rf_tile_size = num_hw_cols*row_fold  # (X*c_fold)
        self.stats['out_rf_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram + out_rf_act_total_reuse_size*out_rf_act_total_reuse_time
        self.insert_max_stats('out_rf_act_size', layer_attr.layer_idx, out_rf_tile_size)

        return mac_util_cycles, mac_cycles

    # Systolic implementation C|K|HW dataflow for depthwise hardware
    # Weight stationary
    def calculate_dw_mac_utilization_cf_cfhw(self, num_h_convs, num_w_convs, num_cin, num_cout, hw_params,
                                          layer_attr, num_hin, num_win):
        systolic_dim = hw_params.mac_wx_type
        available_engines = hw_params.mac_wx
        depth_multiplier = 1.0*num_cout/num_cin
        num_hw_rows = available_engines*systolic_dim # Y
        num_hw_cols = systolic_dim # X
        num_f = num_cout
        # In DW only 1 cin & depth_multiplier will run at a time.
        # -------------------------------------------------
        ## FOR 1 cin; and  depth_multiplier filters calculation
        # -------------------------------------------------
        per_col_computation = layer_attr.Kx*layer_attr.Ky
        row_fold = math.ceil(per_col_computation /num_hw_rows)
        assert depth_multiplier < num_hw_cols, ' Not supported {} > {}'.format(depth_multiplier, num_hw_cols)
        col_fold = 1
        last_iter_per_row_computation = depth_multiplier
        # util = ((fold-1)*1.0 + 1*util_last_iteration)/fold
        row_util = ((row_fold-1)*1.0 + 1.0*(per_col_computation%num_hw_rows)/num_hw_rows)/row_fold
        # Since, col_fold ==1, it can be simplified to util = ((1-1)*per_iter_util + 1*util_last_iteration)/1
        col_util = 1.0*(last_iter_per_row_computation/num_hw_cols)

        # Assumption is Cin==Cout or depth_multiplier == 1; then mac_util = round(row_util*col_util,2)
        # For depth_multiplier !=1, mac_util = depth_multiplier*mac_util
        mac_util_per_cin = round(row_util*col_util,2)
        # -------------------------------------------------
        # For num_cin channels, there are depth_multiplier*num_cin filters
        # where each iteration depth_multiplier filters are processed.
        mac_util = mac_util_per_cin*num_cin*num_cout

        # -------------------------------------------------
        # Calculate Mac cycles and padd cycles
        # -------------------------------------------------
        # Cycles = filter load + num_XY_convs + min(X,rem_KxKyCin_batch) + min(Y,rem_F_batch)
        # Assumption: Each engine starts loading simultaneosly

        # Weight stationary, but for DW weights change for every Cin.
        filter_load_cycles = systolic_dim*num_cin
        ipact_load_cycles = row_fold*systolic_dim
        conv_cycles = num_h_convs*num_w_convs
        opact_store_cycles = col_fold*num_hw_cols
        mac_cycles = filter_load_cycles + ipact_load_cycles + conv_cycles + opact_store_cycles
        mac_util_cycles = mac_util * mac_cycles

        # -------------------------------------------------
        # Calculate RF accesses and SRAM accesses
        # -------------------------------------------------
        w_accesses = num_win + (num_w_convs - 1) * (layer_attr.Ky - layer_attr.Sy)
        h_accesses = num_hin

        # From in act SRAM to RF.
        in_rf_act_from_sram = h_accesses * w_accesses * num_cin
        self.stats['mem_in_act_accesses'][layer_attr.layer_idx] += in_rf_act_from_sram  # store to in_rf

        # Kx*Ky*Cx act
        in_rf_act_tile_size = layer_attr.Kx * layer_attr.Ky * available_engines * systolic_dim
        # Reuse is Fx in space and in time = (num_h_convs - 1) * (layer_attr.Kx - layer_attr.Sx)
        in_rf_act_total_reuse_time = (num_h_convs - 1) * (layer_attr.Kx - layer_attr.Sx)
        in_rf_act_reuse_size = num_w_convs*num_cin*num_f

        self.stats['in_rf_act_accesses'][layer_attr.layer_idx] += in_rf_act_total_reuse_time*in_rf_act_reuse_size + in_rf_act_from_sram
        in_rf_tile_size =  available_engines * systolic_dim
        self.insert_max_stats('in_rf_act_size', layer_attr.layer_idx, in_rf_tile_size)

        # From in wgt SRAM to RF
        # Since, 3*3*Cx activations iterate in Cx dimensions. (Fx,3*3*Cx)
        # weights need to be read (num_h_convs*num_w_conv) times.
        in_rf_wgt_from_sram = num_f * layer_attr.Kx * layer_attr.Ky * num_cin * num_w_convs
        mem_wgt_total_reuse_time = num_h_convs
        mem_wgt_total_reuse_size = num_f*layer_attr.Kx*layer_attr.Ky*num_cin

        self.stats['wgt_rf_accesses'][layer_attr.layer_idx] += in_rf_wgt_from_sram + mem_wgt_total_reuse_time*mem_wgt_total_reuse_size

        # For each 3*3*Cx activations, Fx*3*3*Cx wgts are required.
        mem_wgt_tile_size = available_engines * systolic_dim * systolic_dim * layer_attr.Kx * layer_attr.Ky
        self.insert_max_stats('wgt_rf_size', layer_attr.layer_idx, mem_wgt_tile_size)

        # From RF to out act SRAM
        out_rf_act_to_sram = num_h_convs * num_w_convs * num_f
        self.stats['mem_out_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram


        out_rf_act_total_reuse_time = 0
        out_rf_act_total_reuse_size =  num_h_convs * num_w_convs * num_f
        out_rf_tile_size = systolic_dim * available_engines  # (1xFx)
        self.stats['out_rf_act_accesses'][layer_attr.layer_idx] += out_rf_act_to_sram + out_rf_act_total_reuse_size*out_rf_act_total_reuse_time
        self.insert_max_stats('out_rf_act_size', layer_attr.layer_idx, out_rf_tile_size)


        return mac_util_cycles, mac_cycles

    def get_global_cycles_two_layer(self, batch_cycles_1, batch_cycles_2, end_time_idx):
        total_cycles = 0
        cumm_b1 = 0
        cumm_b2 = 0
        for time in range(end_time_idx + 1):
            b1 = 0
            b2 = 0
            if time in batch_cycles_1:
                b1 = batch_cycles_1[time]

            if time in batch_cycles_2:
                b2 = batch_cycles_2[time]

            total_cycles += max(b1, b2)
            cumm_b1 += b1
            cumm_b2 += b2

        return total_cycles, cumm_b1, cumm_b2

    def get_global_cycles_three_layer(self, batch_cycles_1, batch_cycles_2, batch_cycles_3, end_time_idx):
        total_cycles = 0
        cumm_b1=0
        cumm_b2=0
        cumm_b3 = 0
        for time in range(end_time_idx + 1):
            b1 = 0
            b2 = 0
            b3 = 0

            if time in batch_cycles_1:
                b1 = batch_cycles_1[time]

            if time in batch_cycles_2:
                b2 = batch_cycles_2[time]

            if time in batch_cycles_3:
                b3 = batch_cycles_3[time]

            total_cycles += max(b1, b2, b3)
            cumm_b1 += b1
            cumm_b2 += b2
            cumm_b3 += b3

        return total_cycles, cumm_b1, cumm_b2, cumm_b3

    def set_partial_layer_attributes(self, pw_start_indices, layer_attr):
        # SET PARTIAL LAYER ATTRIBUTES
        if pw_start_indices is None:
            INIT_START_HIN_IDX = 0
            INIT_START_WIN_IDX = 0
            INIT_START_CIN_IDX = 0
            INIT_START_HOUT_IDX = 0
            INIT_START_WOUT_IDX = 0
            INIT_START_COUT_IDX = 0

            # END_HIN = END_HIN_IDX + 1 => NUMLAYERS + INIT_IDX
            INIT_END_HIN = layer_attr.Hin
            INIT_END_WIN = layer_attr.Win
            INIT_END_CIN = layer_attr.Cin
            INIT_END_COUT = layer_attr.Cout
        else:
            INIT_START_HIN_IDX = pw_start_indices.hin
            INIT_START_WIN_IDX = pw_start_indices.win
            INIT_START_CIN_IDX = pw_start_indices.cin

            INIT_START_HOUT_IDX = pw_start_indices.hout
            INIT_START_WOUT_IDX = pw_start_indices.wout
            INIT_START_COUT_IDX = pw_start_indices.cout

            INIT_END_HIN = pw_start_indices.end_hin
            INIT_END_WIN = pw_start_indices.end_win
            INIT_END_CIN = pw_start_indices.end_cin
            INIT_END_COUT = pw_start_indices.end_cout

        return INIT_START_HIN_IDX, INIT_START_WIN_IDX, INIT_START_CIN_IDX, INIT_START_COUT_IDX, \
               INIT_START_HOUT_IDX, INIT_START_WOUT_IDX, \
               INIT_END_HIN, INIT_END_WIN, INIT_END_CIN, INIT_END_COUT

    def h_params_calculation(self, orig_hin, layer_attr, hw_params,
                             INIT_START_HIN_IDX, INIT_END_HIN, orig_hout):
        if INIT_START_HIN_IDX is not None and  orig_hin != INIT_START_HIN_IDX:
                orig_hin = orig_hin - layer_attr.Kx + 1

        end_hin_idx = min(orig_hin + hw_params.hxx, INIT_END_HIN) - 1
        num_hin = end_hin_idx - orig_hin + 1
        # In case of last values -- need to add padding information,
        #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
        if num_hin < layer_attr.Kx:
            num_h_convs = 1
        else:
            num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

        end_orig_hout_idx = orig_hout + num_h_convs - 1
        num_hout = end_orig_hout_idx - orig_hout + 1

        # Adjust hin indices which will be used from previous convolutions
        # Note: no such assumption is made for 'w' dimension
        assert (hw_params.hxx - layer_attr.Kx + 1 > 0), \
            'Increase value of hxx, hxx ({}) - layer_attr.Kx ({}) + 1 <0'.format(hw_params.hxx,
                                                                                 layer_attr.Kx)

        return orig_hin, end_hin_idx, end_orig_hout_idx, num_hin, num_h_convs, num_hout

    def w_params_calculation(self, orig_win, layer_attr, hw_params,
                             INIT_START_WIN_IDX, INIT_END_WIN, orig_wout):
        num_win = min(orig_win + hw_params.wxx, INIT_END_WIN) - orig_win
        if num_win < layer_attr.Ky:
            num_w_convs = 1
        else:
            num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

        end_orig_wout_idx = orig_wout + num_w_convs - 1
        num_wout = end_orig_wout_idx - orig_wout + 1

        assert (hw_params.wxx - layer_attr.Ky + 1 > 0), \
            'Increase value of wxx, wxx ({}) - layer_attr.Ky ({}) + 1 <0'.format(hw_params.wxx,
                                                                                 layer_attr.Ky)

        if INIT_START_WIN_IDX is not None and orig_win != INIT_START_WIN_IDX:
            # Retains the previous Ky-1 windows from previous iterations
            orig_win = orig_win - layer_attr.Ky + 1

        end_win_idx = min(orig_win + hw_params.wxx, INIT_END_WIN) - 1

        return orig_win, end_orig_wout_idx, end_win_idx, num_win, num_w_convs, num_wout

    def f_params_calculation(self, f, hw_params, INIT_END_COUT):
        end_f_idx = min(f + hw_params.fx, INIT_END_COUT) - 1
        num_f = end_f_idx - f + 1
        return end_f_idx, num_f

    def c_params_calculation(self, orig_cin, hw_params, INIT_END_CIN):
        num_cin = min(orig_cin + hw_params.cxx, INIT_END_CIN) - orig_cin
        end_cin_idx = orig_cin + num_cin - 1
        return num_cin, end_cin_idx
    #-------------------------------------------------------
    # DMA in activations/ weights on chip
    #-------------------------------------------------------
    def load_activations_onchip(self, orig_hin, orig_win, orig_cin,
                     num_hin, num_win, num_cin, hw_params,
                     layer_position_idx, layer_attr, dma_cycles):
        cache_hin_idx = orig_hin
        cache_win_idx = orig_win
        cache_cin_idx = orig_cin
        cache_hin_end_idx = cache_hin_idx + num_hin - 1
        cache_win_end_idx = cache_win_idx + num_win - 1
        cache_cin_end_idx = cache_cin_idx + num_cin - 1


        cur_in_act_memory = num_cin * num_hin * num_win

        if layer_position_idx == 0 and \
                not self.onchip_mem.check_if_ip_act_exists(layer_attr.layer_idx, cache_hin_idx,
                                                           cache_hin_end_idx,
                                                           cache_win_idx, cache_win_end_idx,
                                                           cache_cin_idx, cache_cin_end_idx):
            self.debug_message('inDMA (hwc) ip_act[{}:{}][{}:{}][{}:{}]'.format(
                cache_hin_idx, cache_hin_end_idx, cache_win_idx, cache_win_end_idx, cache_cin_idx,
                cache_cin_end_idx))

            self.stats['in_dma_act'][layer_attr.layer_idx] += num_cin * num_win * num_hin
            dma_cycles += num_cin * num_win * num_hin * hw_params.dma_cycles
            self.insert_max_stats('mem_in_act', layer_attr.layer_idx, cur_in_act_memory)
            # Load from DRAM to in act SRAM (store)
            self.stats['mem_in_act_accesses'][layer_attr.layer_idx] += cur_in_act_memory
            self.onchip_mem.insert_ip_act(layer_attr.layer_idx, cache_hin_idx, cache_hin_end_idx,
                                          cache_win_idx, cache_win_end_idx,
                                          cache_cin_idx, cache_cin_end_idx)

        return dma_cycles

    def load_weights_onchip(self, dma_cycles,
                            orig_cin, f,
                            end_cin_idx, end_f_idx,
                            num_f, num_cin,
                            hw_params, layer_attr):

        wgt_volume = num_f * num_cin * layer_attr.Kx * layer_attr.Ky


        # [For cross layers weights are handled separately]
        # if num_cross_layers == 1 and \
        if not self.onchip_mem.check_if_wgt_exists(layer_attr.layer_idx, 0, layer_attr.Kx - 1,
                                                   0, layer_attr.Ky - 1,
                                                   orig_cin, end_cin_idx,
                                                   f, end_f_idx):

            dma_cycles += wgt_volume * hw_params.dma_cycles
            # self.stats['cycles_total'][layer_attr.layer_idx] += hw_params.dma_cycles * wgt_volume
            # self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1

            self.debug_message('inDMA wgts (f,c) [{}:{}][{}:{}]'.format(f, end_f_idx, orig_cin, end_cin_idx))
            # Store to SRAM from DRAM
            self.stats['mem_wgt_accesses'][layer_attr.layer_idx] += wgt_volume
            self.stats['in_dma_wgt'][layer_attr.layer_idx] += wgt_volume
            self.insert_max_stats('mem_wgt', layer_attr.layer_idx, wgt_volume)
            self.onchip_mem.insert_wgt(layer_attr.layer_idx, 0, layer_attr.Kx - 1,
                                       0, layer_attr.Ky - 1, orig_cin, end_cin_idx, f, end_f_idx)
        return dma_cycles

    def mem_out_act_stats(self, layer_attr, isdma, mem_out_act, orig_hout, end_orig_hout_idx, orig_wout,
                                          end_orig_wout_idx, orig_f, end_f_idx):

        self.insert_max_stats('mem_out_act', layer_attr.layer_idx, mem_out_act)

        if isdma:
            self.stats['out_dma_act'][layer_attr.layer_idx] += mem_out_act
            self.debug_message('outDMA (hwc) op_act[{}:{}][{}:{}][{}:{}]'.format(orig_hout, end_orig_hout_idx,
                                                                                orig_wout, end_orig_wout_idx,
                                                                                orig_f,
                                                                                end_f_idx))
            # Load from SRAM to DMA
            self.stats['mem_out_act_accesses'][layer_attr.layer_idx] += mem_out_act
        else:
            self.debug_message('mem_out_act (hwc) op_act[{}:{}][{}:{}][{}:{}]'.format(orig_hout, end_orig_hout_idx,
                                                                                     orig_wout, end_orig_wout_idx,
                                                                                     orig_f,
                                                                                     end_f_idx))
        return

    def init_single_layer_dw(self, hw_params, layer_attr):
        if self.hw_type == 'tb':
            return  self.init_single_layer_tensorbricks_dw(hw_params, layer_attr)
        elif self.hw_type == 'cf_cfhw':
            return self.init_single_layer_cf_cfhw_dw(hw_params, layer_attr)

    def init_single_layer_cf_cfhw_dw(self, hw_params, layer_attr):
        num_macs_w_units = hw_params.mac_wx * hw_params.mac_wx_type * hw_params.mac_wx_type
        mac_units = num_macs_w_units
        self.insert_max_stats('mac_units_available', layer_attr.layer_idx, mac_units)
        self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)
        return

    def init_single_layer_tensorbricks_dw(self, hw_params, layer_attr):
        num_macs_w_units = hw_params.mac_wx * hw_params.mac_wx_type * hw_params.mac_wx_type
        mac_units = hw_params.mac_cx * num_macs_w_units
        self.insert_max_stats('mac_units_available', layer_attr.layer_idx, mac_units)
        self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)
        return

    def init_single_layer(self, hw_params, layer_attr):
        if self.hw_type == 'tb':
            return  self.init_single_layer_tensorbricks(hw_params, layer_attr)
        elif self.hw_type == 'cf_cfhw':
            return self.init_single_layer_cf_cfhw(hw_params, layer_attr)

    def init_single_layer_cf_cfhw(self, hw_params, layer_attr):
        num_macs_w_units = hw_params.mac_wxx* hw_params.mac_wxx_type* hw_params.mac_wxx_type
        mac_units = num_macs_w_units
        padd_units = hw_params.mac_wxx*hw_params.mac_wxx_type
        self.insert_max_stats('mac_units_available', layer_attr.layer_idx, mac_units)
        self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)
        self.insert_max_stats('padd_units_available', layer_attr.layer_idx, padd_units)
        self.insert_max_stats('total_padd_units', layer_attr.layer_idx, padd_units)
        return

    def init_single_layer_tensorbricks(self, hw_params, layer_attr):
        num_macs_w_units = hw_params.mac_wxx* hw_params.mac_wxx_type* hw_params.mac_wxx_type
        mac_units = hw_params.mac_cxx * num_macs_w_units*hw_params.mac_fx
        padd_units = hw_params.mac_wxx*hw_params.mac_wxx_type* hw_params.mac_wxx_type*hw_params.mac_fx
        self.insert_max_stats('mac_units_available', layer_attr.layer_idx, mac_units)
        self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)
        self.insert_max_stats('padd_units_available', layer_attr.layer_idx, padd_units)
        self.insert_max_stats('total_padd_units', layer_attr.layer_idx, padd_units)
        return
