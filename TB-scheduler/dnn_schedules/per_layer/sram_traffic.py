import math
from tqdm import tqdm


def calculate_mac_utilization(self, num_h_convs, num_w_convs, num_cin, num_f, num_macs_w, hw_params):
    cin_util, cin_fold = self.get_mac_utilization(num_cin, hw_params.mac_cxx)
    w_util, w_fold = self.get_mac_utilization(num_w_convs, num_macs_w)
    f_util, f_fold = self.get_mac_utilization(num_f, hw_params.mac_fx)
    mac_cycles = num_h_convs * w_fold * cin_fold * f_fold
    mac_util_cycles = (w_util * cin_util * f_util) * mac_cycles

    # Padd calculation
    padd_utilization = w_util * f_util
    padd_cycles = f_fold * w_fold * num_h_convs
    padd_util_cycles = padd_utilization * padd_cycles

    return mac_util_cycles, mac_cycles, padd_util_cycles, padd_cycles

def conv_cycles(layer_attr, hw_params, init_start_cout_idx, init_end_cout_idx):
    hin = 0
    win = 0
    cin=0
    start_hout_idx= 0
    start_wout_idx= 0

    isconv_layer = (layer_attr.Kx != 1) or (layer_attr.Ky != 1)
    # ------  h parameter calculations
    end_hin_idx = min(hin + hw_params.X, layer_attr.Hin) - 1
    num_hin = end_hin_idx - hin + 1
    if num_hin < layer_attr.Kx:
        num_h_convs = 1
    else:
        num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

    end_hout_idx = start_hout_idx + num_h_convs - 1
    num_hout = end_hout_idx - start_hout_idx + 1
    # ------  w parameter calculations
    end_win_idx = min(win + hw_params.wxx, layer_attr.Win) - 1
    num_win = end_win_idx - win + 1
    if num_win < layer_attr.Ky:
        num_w_convs = 1
    else:
        num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

    # for DW num_macs_w  == 1 and for PW num_macs_w == 3
    num_macs_w = hw_params.mac_wxx - layer_attr.Ky + 1
    end_wout_idx = start_wout_idx + num_w_convs - 1
    num_wout = end_wout_idx - start_wout_idx + 1

    # ------ c parameter calculations
    start_cin_idx = cin
    end_cin_idx = min(start_cin_idx + hw_params.cxx, layer_attr.Cin) - 1
    num_cin = end_cin_idx - start_cin_idx + 1

    mac_cycles_all_filters = 0
    util_cycles_all_filters = 0
    padd_util_cycles_all_filters = 0
    padd_cycles_all_filters = 0
    for f in range(init_start_cout_idx, init_end_cout_idx, hw_params.fx):
        padd_cycles = 0
        padd_util_cycles = 0

        # ------ f parameter calculations
        end_f_idx = min(f + hw_params.fx, init_end_cout_idx + 1) - 1
        num_f = end_f_idx - f + 1
        # --------------------------------------
        # mac utilization
        # --------------------------------------
        mac_util_cycles, mac_cycles, padd_util_cycles, padd_cycles = calculate_mac_utilization(num_h_convs,
                                                                                                    num_w_convs,
                                                                                                    num_cin,
                                                                                                    num_f, num_macs_w,
                                                                                                    hw_params)

        # --------------------------------------
        # Accumulate per filter stats
        # to be used in P2 in PDP
        # --------------------------------------
        util_cycles_all_filters += mac_util_cycles
        mac_cycles_all_filters += mac_cycles

        # -- padd logic --
        # -- for all PCONV/CONV. Since, in P2 of PDP padds can be done as soon as hx*wx*fx
        # is available. This will be done untile hx*wx*F partial product.
        if cin != 0:
            padd_util_cycles_all_filters += padd_util_cycles
            padd_cycles_all_filters += padd_cycles

    # end f

    # return util_cycles_all_filters,mac_cycles_all_filters, padd_util_cycles_all_filters, padd_cycles_all_filters
    return mac_cycles_all_filters,  util_cycles_all_filters, num_h_convs, num_w_convs


def sram_traffic(
        dimension_rows=4,
        dimension_cols=4,
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3,
        strides=1, num_filt=8,
        ofmap_base=2000000, filt_base=1000000, ifmap_base=0,
        # sram_read_trace_file="sram_read.csv",
        # sram_write_trace_file="sram_write.csv"
):
    # Dimensions of output feature map channel
    E_h = math.floor((ifmap_h - filt_h + strides) / strides)
    E_w = math.floor((ifmap_w - filt_w + strides) / strides)

    # Number of pixels in one convolution window
    px_per_conv_window = filt_h * filt_w * num_channels
    r2c = px_per_conv_window

    # Total number of ofmap px across all channels
    num_ofmap_px = E_h * E_w * num_filt
    e2 = E_h * E_w
    e2m = num_ofmap_px

    # Variables to calculate folds in runtime
    num_h_fold = 1
    num_v_fold = 1
    max_parallel_window = 1

    # Variables for utilization calculation
    util = 0
    compute_cycles = 0

    if dimension_rows < px_per_conv_window:
        num_h_fold = math.ceil(px_per_conv_window / dimension_rows)
    else:
        max_parallel_window = math.floor(dimension_rows / px_per_conv_window)

    reqd_cols = num_filt  # Total number of cols to be mapped
    max_cols_per_v_fold = max_parallel_window * dimension_cols
    num_v_folds = math.ceil(reqd_cols / max_cols_per_v_fold)

    remaining_cols = reqd_cols
    cycles = 0
    prev_cycl = 0

    # print("Vertical folds = " + str(num_v_folds))
    # print("Horizontal folds = " + str(num_h_fold))
    # These are the starting addresses of filter weights in the memory
    all_col_addr_list = []
    # for c in range(num_filt):
    #     addr = (c) * r2c + filt_base
    #     all_col_addr_list.append(addr)

    # These are the starting addresses of ifmap windows in the memory
    hc = ifmap_w * num_channels
    all_ifmap_base_addr = []
    # for px in range(int(e2)):  # number of ofmap px in a ofmap channel
    #     addr = (px / E_w) * strides * hc + (px % E_w) * strides
    #     all_ifmap_base_addr.append(addr)

    filter_load_cycles = 0
    for v in range(int(num_v_folds)):
        # print("V fold id: " + str(v))

        # Take a slice of the starting addresses that are relevant for this v_fold
        cols_this_fold = min(remaining_cols, max_parallel_window * dimension_cols)
        idx_start = v * dimension_cols
        idx_end = idx_start + cols_this_fold
        col_addr_list = all_col_addr_list[idx_start:idx_end]

        if num_h_fold > 1:

            rem_h = r2c  # Tracks the elements processed within a conv filter
            next_ifmap_addr = ifmap_base  # Starts from the top left corner of the IFMAP matrix

            for h in range(num_h_fold):
                rows_this_fold = min(rem_h, dimension_rows)
                # print("h fold id: " + str(h))

                # Values returned
                # cycles        -> Cycle count for the next operation ie. cycles elapsed + 1
                # col_addr_list -> The starting filter address for the next iteration
                cycles, col_addr_list = gen_trace_filter_partial(
                    col_addrs=col_addr_list,
                    cycle=cycles,
                    num_rows=dimension_rows,
                    remaining=rows_this_fold
                )
                # print("Weights loaded by " + str(cycles) + " cycles")
                data_out_cycles = cycles  # Store this cycle for parallel readout
                filter_load_cycles += data_out_cycles
                cycles_ifmap = gen_trace_ifmap_partial(
                    cycle=cycles,
                    num_rows=dimension_rows, num_cols=dimension_cols,
                    num_filters=num_filt,
                    remaining=rem_h,
                    remaining_filters=remaining_cols,
                    ifmap_h=ifmap_h, ifmap_w=ifmap_w,
                    filt_h=filt_h, filt_w=filt_w,
                    num_channels=num_channels
                )
                cycles_ofmap = gen_trace_ofmap(
                    cycle=data_out_cycles,
                    num_rows=dimension_rows,
                    num_cols=dimension_cols,
                    ofmap_base=ofmap_base,
                    window_size=rows_this_fold,
                    parallel_window=1,
                    num_ofmap_px=int(e2),
                    filters_done=(v * dimension_cols),
                    num_filter=num_filt
                )

                # print("IFMAPS processed by " + str(cycles) + " cycles")
                util_this_fold = (rows_this_fold * cols_this_fold) / (dimension_rows * dimension_cols)

                rem_h -= rows_this_fold
                cycles = max(cycles_ifmap, cycles_ofmap)

                del_cycl = cycles - prev_cycl
                util += util_this_fold * del_cycl
                compute_cycles += del_cycl
                prev_cycl = cycles

        else:
            # filters_this_fold = min(remaining_cols, max_cols_per_v_fold)
            filt_done = v * max_parallel_window * dimension_cols
            rem = num_filt - filt_done

            parallel_window = math.ceil(rem / dimension_cols)
            parallel_window = int(min(max_parallel_window, parallel_window))

            cycles_filter = gen_filter_trace(
                cycle=cycles,
                num_rows=dimension_rows, num_cols=dimension_cols,
                filt_h=filt_h, filt_w=filt_w, num_channels=num_channels,
                col_addr=col_addr_list,
                parallel_window=parallel_window,
                filters_this_fold=cols_this_fold
            )

            cycles_ifmap, rows_this_fold \
                = gen_ifmap_trace(
                cycle=cycles_filter,
                num_rows=dimension_rows, num_cols=dimension_cols,
                ifmap_h=ifmap_h, ifmap_w=ifmap_w,
                filt_h=filt_h, filt_w=filt_w,
                num_channels=num_channels, stride=strides,
                parallel_window=parallel_window
            )

            cycles_ofmap = gen_trace_ofmap(
                cycle=cycles_filter,
                num_rows=dimension_rows, num_cols=dimension_cols,
                ofmap_base=ofmap_base,
                parallel_window=parallel_window,
                window_size=r2c,
                num_ofmap_px=int(e2),
                filters_done=int(v * max_parallel_window * dimension_cols),
                num_filter=num_filt
            )
            cycles = max(cycles_ifmap, cycles_ofmap)
            del_cycl = cycles - prev_cycl

            # Since multiple filters are being mapped on a single col due to large number of rows
            # util calculation is a little involved,
            # cols_this_fold --> number of filters mapped this fold
            rem = cols_this_fold
            tmp_util = 0
            for _ in range(parallel_window):
                col_used = min(rem, dimension_cols)
                row_used = r2c  # Number of row used will always be in multiple of r2c,
                # parallel window calc took care of this
                tmp_util += row_used * col_used
                rem -= col_used

            # util_this_fold = (rows_this_fold * cols_this_fold) /(dimension_rows * dimension_cols)
            util_this_fold = tmp_util / (dimension_rows * dimension_cols)
            util += util_this_fold * del_cycl
            compute_cycles += del_cycl
            prev_cycl = cycles

        remaining_cols -= cols_this_fold



    # final = str(cycles)
    final_util = (util / compute_cycles) * 100
    # print('final_util: {}'.format(final_util))
    # print("Compute finished at: " + str(final) + " cycles")
    # cycles = cycles - filter_load_cycles
    # compute_cycles = compute_cycles - filter_load_cycles
    return cycles, compute_cycles, util, E_h, E_w


def gen_filter_trace(
        cycle=0,
        num_rows=4, num_cols=4,
        filt_h=3, filt_w=3, num_channels=3,
        col_addr=[],
        parallel_window=1,
        filters_this_fold=4
):

    # There is no data from the left side till the weights are fed in
    # This prefix is to mark the blanks
    prefix = ""
    for r in range(num_rows):
        prefix += ", "

    # Calculate the convolution window size
    r2c = filt_h * filt_w * num_channels

    rem = filters_this_fold  # Track the number of filters yet to process

    # For each wrap around
    for w in range(parallel_window):
        # Number of active columns in this wrap
        cols = min(num_cols, rem)
        rem -= cols

        # For each row in the window
        cycle += r2c
    return cycle


def gen_ifmap_trace(
        cycle=0,
        num_rows=4, num_cols=4,
        ifmap_h=7, ifmap_w=7,
        filt_h=3, filt_w=3,
        num_channels=3, stride=1,
        parallel_window=1
):

    postfix = ""
    for c in range(num_cols):
        postfix += ", "

    E_h = math.floor((ifmap_h - filt_h + stride) / stride)
    E_w = math.floor((ifmap_w - filt_w + stride) / stride)
    e2 = E_h * E_w
    r2c = filt_h * filt_w * num_channels
    rc = filt_w * num_channels
    hc = ifmap_w * num_channels

    idle = num_rows - (r2c * parallel_window)
    idle = max(idle, 0)
    used_rows = num_rows - idle

    # Adding entries for columns and empty rows
    # print("Idle lanes = " + str(idle))
    idle += num_cols
    for i in range(idle):
        postfix += ", "
    postfix += "\n"

    cycle += int(e2)
    return cycle, used_rows


def gen_trace_filter_partial(
        col_addrs=[],  # Ensure that this takes care of the v_folding
        cycle=0,
        num_rows=4,
        remaining=4
):
    # outfile = open(sram_read_trace_file, 'a')
    # num_cols = len(col_addrs)

    # output formatting: Add empty commas for row addresses as no element is fed from the left
    # prefix = ""
    # for r in range(num_rows):
    #     prefix += ", "

    # Entries per cycle
    cycle +=  remaining
    return cycle, col_addrs


def gen_trace_ifmap_partial(
        cycle=0,
        num_rows=4, num_cols=4,
        remaining=4,
        num_filters=8,  #
        remaining_filters=0,  # These two are used to track the reads of PS
        ifmap_h=4, ifmap_w=4,
        filt_h=3, filt_w=3,
        num_channels=3,
        stride=1
):
    # outfile = open(sram_read_trace_file, 'a')
    # postfix = ""
    # for c in range(num_cols):
    #     postfix += ", "
    # postfix += "\n"

    # r2c = filt_h * filt_w * num_channels
    # rc = filt_w * num_channels
    # hc = ifmap_w * num_channels
    E_w = (ifmap_w - filt_w + stride) / stride
    E_h = (ifmap_h - filt_h + stride) / stride

    num_ofmap_px = E_h * E_w
    # index = r2c - remaining
    # base_addr = 0

    # filter_done = num_filters - remaining_filters
    # outfile.write(str(filter_done) + ", " + str(num_filters)+", "+str(remaining_filters)+", "+ "\n")
    # ofmap_offset = filter_done * num_ofmap_px
    # ofmap_offset = filter_done
    # effective_cols = min(remaining_filters, num_cols)
    tick = 0  # Proxy for clock to track input skewing

    cycle += int(num_ofmap_px)
    return cycle


def gen_trace_ofmap(
        cycle=0,
        num_rows=4, num_cols=4,
        ofmap_base=2000000,
        parallel_window=1,
        window_size=27,
        num_ofmap_px=16,  # This is per ofmap channel
        filters_done=0,  # To track v fold
        num_filter=8  # To track if all filters have finished

):

    # cycle = num_cols + cycle     # Accounts for the time taken to reduce accross all cols

    # Corner case when parallel_window = 1, but num_filter < num_cols
    if parallel_window > 1:
        cycle += num_cols
        cycle += window_size  # window_size == r2c
    else:
        rem = (num_filter - filters_done)
        cycle += min(rem, num_cols)
        cycle += window_size

    # ofmap_add_offset = filters_done
    # remaining_filters = num_filter - filters_done

    # effective_cols = num_cols * parallel_window
    # effective_cols = min(effective_cols, remaining_filters)

    cycle += int(num_ofmap_px)
    return cycle




if __name__ == "__main__":
    h_h = 5
    h_w = 5

    r_h = 2
    r_w = 2

    c = 2
    u = 1

    m = 9

    dim_h = 16
    dim_v = 5

    (final, final_util) = sram_traffic(
        dimension_rows=dim_h,
        dimension_cols=dim_v,

        ifmap_h=h_h, ifmap_w=h_w,
        filt_h=r_h, filt_w=r_w,
        num_channels=c,
        strides=u,
        num_filt=m
    )

    print('cycles: {} utilization: {}'.format(final, final_util))
