import torch
import utility.model_summary as model_summary
from custom_models.depthwise import Depthwise
from custom_models.pointwise import Pointwise
from custom_models.depth_separable import DepthSeparable
from custom_models.pointwise_depthwise import PWDW
from custom_models.pdp import PDP
from custom_models.conv import CONV
import pathlib

def generate_dataframe(model, data_file,cin,win,hin):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        model.to('cuda')
        dummy_input =  torch.randn((1,cin,win,hin))
        dummy_input = dummy_input.to('cuda')
        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)
    else:
        dummy_input =  torch.randn((1,cin,win,hin))
        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)


if __name__ == '__main__':

        # model_name = 'Depthwise'
        # model_name = 'Pointwise'
        # model_name = 'DepthSeparable'
        # str = '' + model_name + '()'
        # model = eval(str)

        cin=3
        cout = 6
        cout2 = 3
        stride = 1
        win= 7
        hin = 7
        # model_name = 'DepthSeparable'
        # model = DepthSeparable(cin, cout, stride)

        # model_name = 'pw_dw'
        # model = PWDW(cin, cout, stride)
        # model_name = 'pdp'
        # model = PDP(cin,cout,cout2,stride)

        model_name='conv'
        model = CONV(cin, cout, cout2, stride)
        model.eval()
        data_dir = 'raw_data/test_data/'
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        data_file = '{}/{}.csv'.format(data_dir, model_name)
        generate_dataframe(model,data_file, cin,win,hin)
