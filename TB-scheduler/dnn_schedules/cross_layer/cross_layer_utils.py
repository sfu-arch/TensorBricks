second_layer_dataflow = ['hwfc','hwcf','fchw','cfhw']
first_layer_dataflow = ['HWFC_SchedulePDP','HWCF_SchedulePDP','FCHW_SchedulePDP', 'CFHW_SchedulePDP',  'Tangram', 'FuseLayer']
cc_first_layer_dataflow = ['HWFCScheduleCC','HWCFScheduleCC','FCHWScheduleCC','CFHWScheduleCC', 'Tangram']
single_layer_dataflow = ['HWFCSchedule','HWCFSchedule2','FCHWSchedule','CFHWSchedule']

# -- INIT PDP LAYER --
def init_pdp_stats(cls, first_layer, second_layer, third_layer):
    if cls.hw_type == 'tb':
        return init_pdp_stats_tensorbricks(cls, first_layer, second_layer, third_layer)
    elif cls.hw_type == 'cf_cfhw':
        return init_pdp_stats_cfh_fchw(cls, first_layer, second_layer, third_layer)



def init_pdp_stats_cfh_fchw(cls, first_layer, second_layer, third_layer):
    first_layer_hw_params = cls.load_hw_params_pointwise(True, False)
    # --- Pointwise 1 stats
    cls.debug_message('cin= {} cout= {}'.format(first_layer.Cin, first_layer.Cout))
    cls.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
    first_num_macs_w_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * \
                             first_layer_hw_params.mac_wxx_type
    first_layer_mac_units = first_num_macs_w_units
    first_layer_padd_units =  first_layer_hw_params.mac_wxx*first_layer_hw_params.mac_wxx_type
    cls.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
    cls.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)


    # -- Depthwise stats --
    second_layer_hw_params = cls.load_hw_params_depthwise()
    cls.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
    second_num_macs_w_units = second_layer_hw_params.mac_wx * second_layer_hw_params.mac_wx_type * \
                              second_layer_hw_params.mac_wx_type
    second_layer_mac_units = second_num_macs_w_units
    cls.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)

    # --- Pointwise 2 stats
    third_layer_hw_params = cls.load_hw_params_pointwise(False, False)
    cls.debug_message('{} {} {}'.format(third_layer.layer_idx, third_layer.name, third_layer.attr_type))
    third_num_macs_w_units = third_layer_hw_params.mac_wxx * third_layer_hw_params.mac_wxx_type * \
                             third_layer_hw_params.mac_wxx_type
    third_layer_mac_units = third_num_macs_w_units
    cls.insert_max_stats('mac_units_available', third_layer.layer_idx, third_layer_mac_units)
    third_layer_padd_units = third_layer_hw_params.mac_wxx *third_layer_hw_params.mac_wxx_type
    cls.insert_max_stats('padd_units_available', third_layer.layer_idx, third_layer_padd_units)

    # adding mac units
    total_mac_units = first_layer_mac_units + second_layer_mac_units + third_layer_mac_units
    cls.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', third_layer.layer_idx, total_mac_units)

    # print('total mac units: {} 1: {} 2: {} 3: {}'.format(total_mac_units, first_layer_mac_units,
    #                                                      second_layer_mac_units, third_layer_mac_units))
    cls.insert_max_stats('total_padd_units', first_layer.layer_idx, first_layer_padd_units)
    cls.insert_max_stats('total_padd_units', third_layer.layer_idx, third_layer_padd_units)

    return first_layer_hw_params, second_layer_hw_params, third_layer_hw_params

def init_pdp_stats_tensorbricks(cls, first_layer, second_layer, third_layer):
    first_layer_hw_params = cls.load_hw_params_pointwise(True, False)
    # --- Pointwise 1 stats
    cls.debug_message('cin= {} cout= {}'.format(first_layer.Cin, first_layer.Cout))
    cls.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
    first_num_macs_w_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * \
                             first_layer_hw_params.mac_wxx_type
    first_layer_mac_units = first_layer_hw_params.mac_cxx * first_num_macs_w_units * first_layer_hw_params.mac_fx
    first_layer_padd_units = first_num_macs_w_units * first_layer_hw_params.mac_fx
    cls.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
    cls.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)


    # -- Depthwise stats --
    second_layer_hw_params = cls.load_hw_params_depthwise()
    cls.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
    second_num_macs_w_units = second_layer_hw_params.mac_wx * second_layer_hw_params.mac_wx_type * \
                              second_layer_hw_params.mac_wx_type
    second_layer_mac_units = second_layer_hw_params.mac_cx * second_num_macs_w_units
    cls.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)

    # --- Pointwise 2 stats
    third_layer_hw_params = cls.load_hw_params_pointwise(False, False)
    cls.debug_message('{} {} {}'.format(third_layer.layer_idx, third_layer.name, third_layer.attr_type))
    third_num_macs_w_units = third_layer_hw_params.mac_wxx * third_layer_hw_params.mac_wxx_type * \
                             third_layer_hw_params.mac_wxx_type
    third_layer_mac_units = third_layer_hw_params.mac_cxx * third_num_macs_w_units * third_layer_hw_params.mac_fx
    cls.insert_max_stats('mac_units_available', third_layer.layer_idx, third_layer_mac_units)
    third_layer_padd_units = third_num_macs_w_units * third_layer_hw_params.mac_fx
    cls.insert_max_stats('padd_units_available', third_layer.layer_idx, third_layer_padd_units)

    # adding mac units
    total_mac_units = first_layer_mac_units + second_layer_mac_units + third_layer_mac_units
    cls.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', third_layer.layer_idx, total_mac_units)

    # print('total mac units: {} 1: {} 2: {} 3: {}'.format(total_mac_units, first_layer_mac_units,
    #                                                      second_layer_mac_units, third_layer_mac_units))
    cls.insert_max_stats('total_padd_units', first_layer.layer_idx, first_layer_padd_units)
    cls.insert_max_stats('total_padd_units', third_layer.layer_idx, third_layer_padd_units)

    return first_layer_hw_params, second_layer_hw_params, third_layer_hw_params



# -- INIT DP LAYER --
def init_dp_stats(cls, first_layer, second_layer):
    if cls.hw_type == 'tb':
        return init_dp_stats_tensorbricks(cls, first_layer, second_layer)
    elif cls.hw_type == 'cf_cfhw':
        return init_dp_stats_cfh_fchw(cls, first_layer, second_layer)

def init_dp_stats_tensorbricks(cls, first_layer, second_layer):
    cls.debug_message('{} {}'.format(first_layer.Cin, second_layer.Cout))
    # -- Depthwise stats --
    first_layer_hw_params = cls.load_hw_params_depthwise()
    cls.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
    first_num_macs_w_units = first_layer_hw_params.mac_wx * first_layer_hw_params.mac_wx_type \
                             * first_layer_hw_params.mac_wx_type
    first_layer_mac_units = first_layer_hw_params.mac_cx * first_num_macs_w_units
    cls.insert_max_stats('mac_units_available', second_layer.layer_idx, first_layer_mac_units)

    # --- Pointwise stats
    second_layer_hw_params = cls.load_hw_params_pointwise(True, False)
    cls.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
    second_num_macs_w_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type \
                              * second_layer_hw_params.mac_wxx_type
    second_layer_mac_units = second_layer_hw_params.mac_cxx * second_num_macs_w_units * second_layer_hw_params.mac_fx
    second_layer_padd_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type \
                              * second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_fx
    cls.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)
    cls.insert_max_stats('padd_units_available', second_layer.layer_idx, second_layer_padd_units)

    # adding mac units
    total_mac_units = first_layer_mac_units + second_layer_mac_units
    cls.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)

    cls.insert_max_stats('total_padd_units', first_layer.layer_idx, 0)
    cls.insert_max_stats('total_padd_units', second_layer.layer_idx, second_layer_padd_units)

    return first_layer_hw_params, second_layer_hw_params

def init_dp_stats_cfh_fchw(cls, first_layer, second_layer):

    # -- Depthwise stats --
    first_layer_hw_params = cls.load_hw_params_depthwise()
    cls.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
    first_num_macs_w_units = first_layer_hw_params.mac_wx * first_layer_hw_params.mac_wx_type * \
                             first_layer_hw_params.mac_wx_type
    first_layer_mac_units = first_num_macs_w_units
    first_layer_padd_units =  0
    cls.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
    cls.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)


    # --- Pointwise 2 stats
    second_layer_hw_params = cls.load_hw_params_pointwise(True, False)
    cls.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))
    second_num_macs_w_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * \
                             second_layer_hw_params.mac_wxx_type
    second_layer_mac_units = second_num_macs_w_units
    cls.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)
    second_layer_padd_units = second_layer_hw_params.mac_wxx *second_layer_hw_params.mac_wxx_type
    cls.insert_max_stats('padd_units_available', second_layer.layer_idx, second_layer_padd_units)

    # adding mac units
    total_mac_units = first_layer_mac_units + second_layer_mac_units
    cls.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)

    cls.insert_max_stats('total_padd_units', first_layer.layer_idx, 0)
    cls.insert_max_stats('total_padd_units', second_layer.layer_idx, second_layer_padd_units)

    return first_layer_hw_params, second_layer_hw_params


# -- INIT CC LAYER --
def init_cc_stats(cls, first_layer, second_layer):
    if cls.hw_type == 'tb':
        return init_cc_stats_tensorbricks(cls, first_layer, second_layer)
    elif cls.hw_type == 'cf_cfhw':
        return init_cc_stats_cfh_fchw(cls, first_layer, second_layer)
    elif cls.hw_type == 'tangram':
        return init_cc_stats_cfh_fchw(cls, first_layer, second_layer)


def init_cc_stats_tensorbricks(cls, first_layer, second_layer):
    # print('running conv - conv pipeline')
    first_layer_hw_params = cls.load_hw_params_pointwise(True, False)
    # --- Pointwise 1 stats
    cls.debug_message('cin= {} cout= {}'.format(first_layer.Cin, first_layer.Cout))
    cls.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
    first_num_macs_w_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_wxx_type
    first_layer_mac_units = first_layer_hw_params.mac_cxx * first_num_macs_w_units * first_layer_hw_params.mac_fx
    first_layer_padd_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_wxx_type * first_layer_hw_params.mac_fx
    cls.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
    cls.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)

    # -- Pointwise stats 2 --
    second_layer_hw_params = cls.load_hw_params_pointwise(False, False)
    cls.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))

    second_num_macs_w_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * \
                              second_layer_hw_params.mac_wxx_type
    second_layer_mac_units = second_layer_hw_params.mac_cxx * second_num_macs_w_units * second_layer_hw_params.mac_fx
    cls.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)

    second_layer_padd_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * \
                              second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_fx
    cls.insert_max_stats('padd_units_available', second_layer.layer_idx, second_layer_padd_units)

    # adding mac units
    total_mac_units = first_layer_mac_units + second_layer_mac_units
    cls.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)

    # print('total mac units: {} 1: {} 2: {}'.format(total_mac_units, first_layer_mac_units,
    #                                                      second_layer_mac_units))
    cls.insert_max_stats('total_padd_units', first_layer.layer_idx, first_layer_padd_units)
    cls.insert_max_stats('total_padd_units', second_layer.layer_idx, second_layer_padd_units)

    return first_layer_hw_params, second_layer_hw_params


def init_cc_stats_cfh_fchw(cls, first_layer, second_layer):
    # print('running conv - conv pipeline')
    first_layer_hw_params = cls.load_hw_params_pointwise(True, False)
    # --- Pointwise 1 stats
    cls.debug_message('cin= {} cout= {}'.format(first_layer.Cin, first_layer.Cout))
    cls.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
    first_num_macs_w_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * \
                             first_layer_hw_params.mac_wxx_type
    first_layer_mac_units = first_num_macs_w_units
    first_layer_padd_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type
    cls.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
    cls.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)

    # -- Pointwise stats 2 --
    second_layer_hw_params = cls.load_hw_params_pointwise(False, False)
    cls.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))

    second_num_macs_w_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * \
                              second_layer_hw_params.mac_wxx_type
    second_layer_mac_units = second_num_macs_w_units
    cls.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)

    second_layer_padd_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * \
                              second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_fx
    cls.insert_max_stats('padd_units_available', second_layer.layer_idx, second_layer_padd_units)

    # adding mac units
    total_mac_units = first_layer_mac_units + second_layer_mac_units
    cls.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)

    # print('total mac units: {} 1: {} 2: {}'.format(total_mac_units, first_layer_mac_units,
    #                                                      second_layer_mac_units))
    cls.insert_max_stats('total_padd_units', first_layer.layer_idx, first_layer_padd_units)
    cls.insert_max_stats('total_padd_units', second_layer.layer_idx, second_layer_padd_units)

    return first_layer_hw_params, second_layer_hw_params


def init_dc_stats(cls, first_layer, second_layer):
    # print('running conv - conv pipeline')
    first_layer_hw_params = cls. load_hw_params_depthwise()
    # --- Depthwise 1 stats
    cls.debug_message('cin= {} cout= {}'.format(first_layer.Cin, first_layer.Cout))
    cls.debug_message('{} {} {}'.format(first_layer.layer_idx, first_layer.name, first_layer.attr_type))
    first_num_macs_w_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type * \
                             first_layer_hw_params.mac_wxx_type
    first_layer_mac_units = first_num_macs_w_units
    first_layer_padd_units = first_layer_hw_params.mac_wxx * first_layer_hw_params.mac_wxx_type
    cls.insert_max_stats('mac_units_available', first_layer.layer_idx, first_layer_mac_units)
    cls.insert_max_stats('padd_units_available', first_layer.layer_idx, first_layer_padd_units)

    # -- Pointwise stats 2 --
    second_layer_hw_params = cls.load_hw_params_pointwise(False, False)
    cls.debug_message('{} {} {}'.format(second_layer.layer_idx, second_layer.name, second_layer.attr_type))

    second_num_macs_w_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * \
                              second_layer_hw_params.mac_wxx_type
    second_layer_mac_units = second_num_macs_w_units
    cls.insert_max_stats('mac_units_available', second_layer.layer_idx, second_layer_mac_units)

    second_layer_padd_units = second_layer_hw_params.mac_wxx * second_layer_hw_params.mac_wxx_type * \
                              second_layer_hw_params.mac_wxx_type * second_layer_hw_params.mac_fx
    cls.insert_max_stats('padd_units_available', second_layer.layer_idx, second_layer_padd_units)

    # adding mac units
    total_mac_units = first_layer_mac_units + second_layer_mac_units
    cls.insert_max_stats('total_mac_units', first_layer.layer_idx, total_mac_units)
    cls.insert_max_stats('total_mac_units', second_layer.layer_idx, total_mac_units)

    # print('total mac units: {} 1: {} 2: {}'.format(total_mac_units, first_layer_mac_units,
    #                                                      second_layer_mac_units))
    cls.insert_max_stats('total_padd_units', first_layer.layer_idx, first_layer_padd_units)
    cls.insert_max_stats('total_padd_units', second_layer.layer_idx, second_layer_padd_units)

    return first_layer_hw_params, second_layer_hw_params