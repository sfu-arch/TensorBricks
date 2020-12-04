import torch
import torch.nn as nn
import utility.model_summary as model_summary
from PIL import Image
import pretrainedmodels
import pretrainedmodels.utils as utils
import pathlib

def generate_dataframe(model, data_file):
    load_img = utils.LoadImage()

    # transformations depending on the model
    # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
    tf_img = utils.TransformImage(model)

    path_img = 'load_models/dog.jpg'

    input_img = load_img(path_img)
    input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
    input = torch.autograd.Variable(input_tensor,
                                    requires_grad=False)

    output_logits = model(input)  # 1x1000


    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input.to('cuda')
        model.to('cuda')
        dummy_input =  torch.randn((1,3,224,224))
        model_list = [module for module in model.modules() if type(module) != nn.Sequential]
        i=0
        for idx, name  in enumerate(model_list):
            class_name = type(name).__name__

            if class_name == 'Conv2d':
                print(i, name )
                i+=1
            else:
                print(class_name)


        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)
    else:
        dummy_input =  torch.randn((1,3,224,224))
        model_list = [module for module in model.modules() if type(module) != nn.Sequential]
        i=0
        for idx, name  in enumerate(model_list):
            class_name = type(name).__name__

            if class_name == 'Conv2d':
                print(i, name )
                i+=1
            else:
                print(class_name)
        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)


if __name__ == '__main__':

# broken moedels
# 'nasnetalarge',
# ['fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2'

    pretrainedmodel_list = ['polynet','fbresnet152','resnet50','se_resnext50_32x4d',
     'se_resnext101_32x4d','pnasnet5large','nasnetamobile', 'xception']
    print(pretrainedmodels.model_names)
    model_name = 'xception'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000)
    data_dir = 'raw_data/benchmarks/'
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    data_file = '{}/{}.csv'.format(data_dir, model_name)
    generate_dataframe(model, data_file)