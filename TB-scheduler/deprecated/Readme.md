# Data generator 
This project generates the data as a flat buffer file expected by uIR_ml framework

## Pre-requisites

```
pytorch >= 1.2
```

# Run command

Set the configuration in `params/params.yaml`
```
mkdir results
export PYTHONPATH=<root directory>
python main.py
```

The results will be generated in results directory


# Assumptions
* Only padding = 'valid' is supported currently, i.e in pytorch - padding = 0
* Can handle multiple strides
* hx and wx parameters in hardware\_config.yaml should be set as multiples of n
```
hx = Kx + n*Sx
wx = Ky + n*Sy 
```
* hardware parameter k should work only for square kernels i.e Kx = Ky 
* Currently, 'hx' assumes that when the next batch is fetched hardware keeps previous kernels for convolutions
* However, the same is not true with 'wx' due to our hardware design. Hence, next batch will fetch from 
```
# The number of rows fetched is still = wx = num_mac +  Ky - 1
for win in range(0, layer_attr.Win, hw_params.wx ):
    if win != 0: 
        win = win - layer_attr.Ky + 1
        # and this extra two rows need to DMA as it cannot be preserved 
        # when streaming along h direction 
```

# Schedules
* cwh_schedule_1 
    * For pointwise does not store input activations, only stores partial product in the memory
    * Schedule order for pointwise is "fcwh" and for depthwise is "cwh" 

* hwc_schedule 
    * Based on observations from the network - weight memory is small. Hence, we `indma['wgts']` 
        and stream through all filters
    * DMA all weights and keep them on chip
    * Thus, schedule order is "hwcfilter" for pointwise and for depthwise it is "hwc"
    * Along 'wx' and 'hx' dimension, rows and columns are repeated dma across batches 
    * Since we are streaming and past activations may not be preserved.
    * Since, outDMA is slow we store partial products separately, i.e. equal to double buffer
        * `mem_partial_product = mem_out_act`
        

 
##### Note: in cross layer dma['wgt'] is exclusively controlled at the topmost loop iteration   
# PW+DW+PW Schedule
| layer type | mem_in | mem_out | in_dma | out_dma | Remarks               |
|------------|--------|---------|--------|---------|-----------------------|
| PW         | Y      | Y       | Y      | -       | is_first              |
| DW         | -      | Y       | -      | -       | !is_last              |
| PW         | -      | Y       | -      | Y       | is_last               |

# DW + PW Schedule
| layer type | mem_in | mem_out | in_dma | out_dma | Remarks  |
|------------|--------|---------|--------|---------|----------|
| DW         | Y      | Y       | Y      | -       | is_first |
| PW         | -      | Y       | -      | Y       | is_last  |

Only in this schedule partial depthwise blocks are computed immediately.
 In other schedules pointwise comes first, and the next layer starts only 
 after the partial products are computed entirely across channel dimension

# PW + DW Schedule
| layer type | mem_in | mem_out | in_dma | out_dma | Remarks  |
|------------|--------|---------|--------|---------|----------|
| PW         | Y      | Y       | Y      | -       | is_first |
| DW         | -      | Y       | -      | Y       | is_last  |

# Per Layer Schedule 
| layer type | mem_in | mem_out | in_dma | out_dma | dma_wgt | Remarks       |
|------------|--------|---------|--------|---------|---------|---------------|
| PW/DW      | Y      | Y       | Y      | Y       | Y       | !cross_layer  |
