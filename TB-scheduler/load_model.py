import pandas as pd
from ast import literal_eval as make_tuple
from collections import OrderedDict
from attrdict import AttrDict

class Net():

    def __init__    (self,df):
        self.layers = OrderedDict()
        self.num_layers = len(df)
        self.extract_layer_features(df)

    def extract_layer_features(self, df):
        # layer = AttrDict()
        num_layers = 0
        for idx in range(0,len(df)):
            name = df['name'][idx]
            type = df['type'][idx]
            # TODO: Find a better way. Second condition is for efficient-net.
            if type == 'Conv2d' or 'Conv2d' in type:
                num_layers = idx
                (Kx,Ky) = make_tuple('(0,0)' if pd.isnull(df['kernel_size'][idx]) else df['kernel_size'][idx])
                if Kx == Ky:
                    K = Kx
                # else:
                #     raise Exception("ERROR KX != Ky not supported ")

                (Nin,Cin,Win,Hin) = make_tuple(df['ifm'][idx])
                (Nout, Cout, Wout, Hout) = make_tuple(df['ofm'][idx])
                mac = df['mac'][idx]
                attr_type = df['attr_type'][idx]
                # groups, stride, padding, bias
                groups = df['groups'][idx]
                (Sx, Sy) = make_tuple(df['stride'][idx])
                (Px,Py) = make_tuple(df['padding'][idx])
                Bias = df['bias'][idx]

                self.layers[name] = AttrDict({'name': name, 'layer_idx': idx, 'type': type,
                                              'Kx': Kx, 'Ky': Ky, 'K': Kx,
                                              'Nin': Nin, 'Cin': Cin, 'Win':Win, 'Hin': Hin,
                                              'Nout': Nout, 'Cout': Cout, 'Wout': Wout, 'Hout': Hout,
                                              'mac': mac, 'attr_type': attr_type,
                                              'Sx': Sx, 'Sy': Sy, 'Px': Px, 'Py': Py, 'Bias': Bias,
                                              'Depth_multiplier': int(groups/Cin)
                                              })

            elif type == 'Linear':
                continue
        num_layers +=1
        self.num_layers = num_layers




if __name__ == '__main__':
    # model_names = ['Depthwise']
    model_names = ['mobilenet']
    data_folder='./raw_data/benchmarks/'

    for model_name in model_names:

        df = pd.read_csv(data_folder + model_name + '.csv')
        net = Net(df)
        for key,value in net.layers.items():
            print('{} {}'.format(key,value.attr_type))

        print('end')







