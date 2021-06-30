```
+-----------------------------+
|HW Constraints               |
|* MACs/layer for             |       +---------------------------------+      +------------------------+
|single/cross layer dataflow  +------>+Tensor-brick Compiler            |      |Chisel RTL Mapping      |
|* params/systolic_config.yaml|       |* Parse DNN graph [load_models/] |      |* Configure once per DNN|
+-----------------------------+       |* Static schedule for each       +----->+* Execute entire DNN    |
+-----------------------------+       |  layer/group of layers          |      +------------------------+
|Python Schedule              |       |* Parameters tiling and          |
|* For single/cross layer     +------>+  memory format transformation   |
|  dataflow                   |       |* Generate Configuration File    |
|* dnn_schedules/             |      >----------------------------------+
+-----------------------------+      |
+-----------------------------+      |
|Pytorch DNN                  +------+
|* custom_models/             |
+-----------------------------+

```
