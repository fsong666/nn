import torch
import torchvision as tv
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import json
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

PATH = '/home/sf/Documents/tianchi'
PATH_json = PATH + '/dataset/amap_traffic_annotations_train.json'
PATH_imgs = PATH + '/dataset/amap_traffic_train_0712'


class Dataset(data.Dataset):
    def __init__(self, json_file, dataset_dir, transform=None, show=False):
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            dict = json.load(f)
        self.dict_list = dict['annotations']
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.show = show

    def __len__(self):
        return len(self.dict_list)

    def _read_images(self, frame_list, id):
        frames = []
        i = 0
        for img_dict in frame_list:
            i += 1
            if i > 3:
                break
            frame_name = img_dict["frame_name"]
            img_name = os.path.join(self.dataset_dir, id, frame_name)
            img = Image.open(img_name)

            if self.show:
                print('img.type= ', type(img))
                print('img.shape= ', np.array(img).shape)
                img.show()

            if self.transform is not None:
                img = self.transform(img)

            frames.append(img)

        return frames

    def __getitem__(self, index):
        dict = self.dict_list[index]

        status = dict["status"]

        id = dict["id"]

        frame_list = dict["frames"]
        frames = self._read_images(frame_list, id)
        x_3d = torch.stack(frames, dim=0)

        label = torch.tensor(status)
        # label = torch.tensor(int(id))
        return x_3d, label


class Dataset2(data.Dataset):
    def __init__(self, json_file, dataset_dir, folders, transform=None, show=False):
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            dict = json.load(f)
        self.dict_list = dict['annotations']
        self.transform = transform
        self.dataset_dir = dataset_dir
        self.show = show
        self.folders = folders

    def __len__(self):
        return len(self.folders)

    def _read_images(self, frame_list, id):
        frames = []
        i = 0
        for img_dict in frame_list:
            i += 1
            if i > 3:
                break
            frame_name = img_dict["frame_name"]
            img_name = os.path.join(self.dataset_dir, id, frame_name)
            img = Image.open(img_name)

            if self.show:
                print('img.type= ', type(img))
                print('img.shape= ', np.array(img).shape)
                img.show()

            if self.transform is not None:
                img = self.transform(img)

            frames.append(img)

        return frames

    def __getitem__(self, index):
        id = self.folders[index]
        dict = self.dict_list[int(id) - 1]
        status = dict["status"]
        # print('id = {} status= {}'.format(id, status))

        frame_list = dict["frames"]
        frames = self._read_images(frame_list, id)
        x_3d = torch.stack(frames, dim=0)

        label = torch.tensor(status)
        return x_3d, label


def get_origin_data():
    data_transforms2 = [
        tv.transforms.Resize((720, 1280)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.4451, 0.4687, 0.4713], [0.3244, 0.3277, 0.3358]),
    ]

    dataset_train = Dataset(PATH_json,
                            PATH_imgs,
                            transform=tv.transforms.Compose(data_transforms2),
                            show=False)
    return dataset_train


def get_random_split(dataset_train=get_origin_data(), val_split=0.2):
    """
    如果输入的数据是非tensor,如序列帧list[tensor1, tensor2]，
    DataLoader不能合并list构成mini_batch,所以，batch_size=1
    """

    train_size = int((1 - val_split) * len(dataset_train))
    val_size = int(val_split * len(dataset_train))
    split_train, split_val = data.random_split(dataset_train, [train_size, val_size])

    print('split_train.len= {} | split_val.len= {}'.format(len(split_train), len(split_val)))

    return split_train, split_val


def get_train_test_split(val_split=0.2, random_state=3):
    """
    train_test_split() from sklearn
    """
    data_transforms = [
        tv.transforms.Resize((224, 224)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.4451, 0.4687, 0.4713], [0.3244, 0.3277, 0.3358]),
    ]

    all_list = os.listdir(PATH_imgs)
    all_list.remove('.DS_Store')

    train_list, val_list = train_test_split(all_list, test_size=val_split, random_state=random_state)

    split_train = Dataset2(PATH_json, PATH_imgs, train_list,
                           transform=tv.transforms.Compose(data_transforms))

    split_val = Dataset2(PATH_json, PATH_imgs, val_list,
                         transform=tv.transforms.Compose(data_transforms))

    print('split_train.len= {} | split_val.len= {}'.format(len(split_train), len(split_val)))

    return split_train, split_val


def get_split(dataset_train=get_origin_data(), val_split=0.2):
    """
    not random split
    """
    val_range = list(range(int(len(dataset_train)
                               - val_split * len(dataset_train)),
                           len(dataset_train)))
    train_range = list(range(0, int(len(dataset_train)
                                    - val_split * len(dataset_train))))
    # # 必须先截取高索引的子数据，再截取低索引的数据，否则list index out of range
    # # 因为先截取低位后，原数据集就只剩从0开始索引的后半部分数据集，索引不再是从高位开始索引
    split_val = torch.utils.data.Subset(dataset_train, val_range)
    split_train = torch.utils.data.Subset(dataset_train, train_range)
    return split_train, split_val


def get_data(split_data=get_random_split()):
    split_train, split_val = split_data
    dl_train = DataLoader(split_train,
                          batch_size=4,  # x 是tensor的list,不能拼接合并
                          shuffle=True,
                          drop_last=True,
                          num_workers=4)
    dl_val = DataLoader(split_val,
                        batch_size=1,
                        shuffle=False,
                        drop_last=True,
                        num_workers=4)

    return dl_train, dl_val


def test_dataLoader(dataset):
    print('len=', len(dataset))
    for i, (inp, label) in enumerate(dataset):
        print('{} | in.shape= {} | label= {}'.format(i, inp.shape, label))


def learning():
    x = torch.range(1, 8 * 4).reshape(8, 4)
    y = torch.randint(0, 2, (8, 10))
    z = torch.randint(2, 4, (8, 10))
    print('x=\n', x)
    print('y=\n', y)
    print('z=\n', z)

    """
    data[n:m] 是一个len为n的tuple, data[n:m][0]是输入数据，　data[n:m][1]是对应的标签
    n取决于，多少种数据形成同步集合
    TensorDataset(*args)
    n = len(args)
    """
    print('\n----TensorDataset: data[n:m] '
          '是一个len为len(args)的tuple, data[n:m][0]是输入数据，　data[n:m][1]是对应的标签-----\n')
    data = TensorDataset(x, y, z)

    print('data[0:3]=\n', data[0:3])
    print('data[0:3].len=\n', len(data[0:3]))
    print('data[0:3][0]=\n', data[0:3][0])

    """
    DataLoader: 随机合并生成mini_batch数据，transformation
    #输入的 x 是tensor的list,不能拼接合并, 所以batch_size=1
    """
    print('\n----DataLoader: 随机合并生成mini_batch数据，transformation-----\n')
    dl = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)
    print('dl.len=\n', len(data))

    for x, y, z in dl:
        print('x=\n', x)
        print('y=\n', y)
        print('z=\n', z)


if __name__ == '__main__':
    # # test random_split
    test_dataLoader(get_random_split()[0])
    # test_dataLoader(get_train_test_split()[0])
    # # test no random_split
    # test_dataLoader(get_split()[0])

    # learning()
