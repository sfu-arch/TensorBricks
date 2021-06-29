# Tensor-Brick Hardware Summary

We present the results of FPGA synthesis tools. Here we choose hyper-parameters to enable the design to fit the FPGA and lead to optimal performance for end-to-end DNN. As shown in the figure below, we breakdown the resource usage into four different components, CLBs/ALMs, Registers, BRAMs, and Power.


![PieChart](https://www.dropbox.com/s/3njtn7bg5c0rgsb/PieChart.png?raw=1)

| power   | 36.3W         |
|---------|---------------|
| fps     | 128           |
| Batch=1 | 0.22 ms/inf/W |
| Titan XP| 299 fps       |
