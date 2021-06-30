
# DNN Architectures with LRCONV operations 

| DNN | Inverted Residual | Depth Separable | Skip | Bottleneck | 1xN + Nx1 Rank |
|-|-|-|-|-|-|
| EfficientNet | Yes | Yes | Yes | Yes | - |
| Mobilenet_v2 | Yes | Yes | Yes | Yes | - |
| Mnasnet1_0 | Yes | Yes | Yes | Yes | - |
| PNasNet | Depends | Yes | Yes | Yes | Yes |
| NasNet | Depends | Yes | Yes | Yes | Yes |
| AmoebaNet | Depends | Yes | Yes | Yes | Yes |
| FbNet | Depends | Yes | Yes | Yes | Yes |
| Xception | - | Yes | Yes | Yes | - |
| Mobilenet | - | Yes | Yes | Yes | - |
| DenseNet | - | - | Yes | Yes | - |
| Inception-V3 | - | - | Yes | Yes | Yes |
| ResNet | - | - | Yes | Yes | - |
| ShuffleNet | - | Yes | - | Yes | - |
| SqueezeNet | - | - | Yes | Yes | - |
| VGG | - | - | - | - | - |
| AlexNet | - | - | - | - | - |

The Table above shows different layers created by LRCONVs. The examples include 
basic building block layer types such as Depthwise separable Layers (DP + PT), 
Inverted residual layers (PT+DP+PT), skip, connections, bottleneck layers and other 
LRCONVs such as (1xN + Nx1). Emerging DNNs heavily employ LRCONVs to generate 
compact DNNs with high accuracies and low MAC operations. Emerging DNNs 
are explored by other DNN softwares called as Neural Architecture Search (NAS).
Such networks chose from the building blocks shown in the Table and can create 
non sequential and irregular DNN patterns. 


![image](dnntypes.png)


The Figure above shows an AmoebaNet and a PNASNet cell created by the NAS software.
They use irregular combinations of basic building block layers and require complex 
dataflow schedules for optimal performance. Other examples include Nasnet and FbNet.
