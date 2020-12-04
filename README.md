# TensorBricks 
This repository contains the "TB-Scheduler" and "TB-hardware". 
* [TB-Scheduler](TB-scheduler/README.md)
* [TB-Hardware](TB-hardware/README.md)


# Tensorbricks Schedule 
* This project explores outer and inner dataflow schedules for 
a i) **single layer** dataflows, ii) **cross layer** - two layer and three layer dataflows. 
* Tensorbricks **maps non homogeneous hardware resources** for each layer 
in a cross layer dataflow. 
* Tensorbricks finds optimal designs that lower energy between **16--25\%**, 
improve performance by  **2--81\%** while requiring **3.5---52X**
less SRAM compared to prior state-of-the-art.
* The TB-Scheduler explores various single layer and cross-layer DNN schedules. 
The link to the README can be found in [TB-Scheduler](TB-scheduler/README.md)

# TensorBricks  Hardware 
* Tensor-Brick hardware is a Chisel-based generator of low-rank convolution accelerators. 
  The core of Tensor-Brick is a hardware library of building blocks (*bricks*) that provides parameterized design for:
    * creating pipelined units that execute arbitrary acyclical dataflow graphs of *linear algebra operators*.
The unit of operand in all these operators is a tensor (rank and shape can be set by designer),
    * *multi-stream interfaces* that decouple the compute unit from the underlying tensor storage and overlaps data movement with computation.
    * *layer-buffers* that capture data-reuse across layers and simplify the process of exchanging data across layers.
    * The TB-hardware  implements the chisel hardware to execute cross-layer dataflows. 
* The link to the README can be found in [TB-Hardware](TB-hardware/README.md)
  