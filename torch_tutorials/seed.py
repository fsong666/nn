import torch
import os

def fix_seed():
    if torch.cuda.is_available():
        print("gpu cuda is available!")
        torch.cuda.manual_seed(1000)
        print(torch.rand(1, 2, 3, device=torch.device('cuda:0')))
    else:
        print("cuda is not available! cpu is available!")
        torch.manual_seed(1000)
        print(torch.rand(1, 2, 3))


if __name__ == '__main__':
    fix_seed()
    print(__file__)
    print(os.path.realpath(__file__))
    print(os.path.dirname(os.path.realpath(__file__)))
    print(os.getcwd())