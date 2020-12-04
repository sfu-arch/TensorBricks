import torch
import utility.model_summary as model_summary
import pathlib
import sys
# sys.path.append('../custom_models')

from custom_models.mobilenet import MobileNet
from PIL import Image
from torchvision import transforms

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
        input_batch = input_batch.to('cuda')
        model.to('cuda')
        # x = summary(model, (3, 224, 224))
        dummy_input =  torch.randn((1,3,224,224))
        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)

    else:
        # x = summary(model, (3, 224, 224))
        dummy_input =  torch.randn((1,3,224,224))
        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)


if __name__ =='__main__':
    model = MobileNet()
    model_name = 'mobilenet'
    model.eval()
    data_dir='raw_data/test_data/'
    pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
    data_file = '{}/{}.csv'.format(data_dir,model_name)
    generate_dataframe(model, data_file)
