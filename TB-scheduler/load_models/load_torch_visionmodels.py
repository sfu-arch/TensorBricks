import torch

import utility.model_summary as model_summary
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import pathlib


def generate_dataframe(model, data_file):
    input_image = Image.open('load_models/dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch.to('cuda')
        model.to('cuda')
        dummy_input =  torch.randn((1,3,224,224))
        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)
    else:
        dummy_input =  torch.randn((1,3,224,224))
        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)

if __name__ == '__main__':

    # # load model
    # torchvision_model_names = ['squeezenet1_0', 'resnet18', 'alexnet', 'vgg16', 'densenet161',
    #                'inception_v3', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2',
    #                'resnext50_32x4d', 'wide_resnet50_2', 'wide_resnet50_2', 'wide_resnet50_2', 'mnasnet1_0']
    #
    # for model_name in torchvision_model_names[0]:
        model_name = 'mobilenet_v2'
        str = 'models.' + model_name + '(pretrained=True)'
        model = eval(str)
        model.eval()
        data_dir = 'raw_data/benchmarks/'
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        data_file = '{}/{}.csv'.format(data_dir, model_name)
        generate_dataframe(model,data_file)
