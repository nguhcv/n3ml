import argparse

import torch.nn as nn

import n3ml.network


class SCNN(n3ml.network.Network):
    def __init__(self):
        super(SCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=24, kernel_size=3, stride=2, bias=False)
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, bias=False)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, bias=False)
        self.fc1 = nn.Linear(in_features=864, out_features=128, bias=False)
        self.fc2 = nn.Linear(in_features=128, out_features=10, bias=False)


def app(opt):
    print(opt)

    model = SCNN()

    state_dict_conv = [l for l in model.named_children() if isinstance(l[1], nn.Conv2d)]
    state_dict_linear = [l for l in model.named_children() if isinstance(l[1], nn.Linear)]

    state_dict = [l for l in reversed(state_dict_conv + [('', 0)] + state_dict_linear)]
    state_dict = {'arr_'+str(i): (state_dict[i][1] if isinstance(state_dict[i][1], int) else state_dict[i][1].weight.detach().cpu().numpy()) for i in range(len(state_dict))}

    n3ml.save(state_dict, mode='loihi', f=opt.save)

    state_dict = n3ml.load(f=opt.save, mode='loihi', allow_pickle=True)

    print(state_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 실제 실행을 위해서는 경로가 수정되어야 합니다.
    parser.add_argument('--npz', default='data/npz/mnist_loihi_test.npz')
    parser.add_argument('--save', default='data/npz/n3ml_loihi_202108241548.npz')

    app(parser.parse_args())
