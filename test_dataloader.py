from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from os.path import join
from torch.utils import data, model_zoo


DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
INPUT_SIZE = '512,256'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
DATA_LIST_PATH_TARGET_VALIDATION = './dataset/cityscapes_list/val.txt'

trainloader = data.DataLoader(
        GTA5DataSet(DATA_DIRECTORY_TARGET, DATA_LIST_PATH_TARGET, max_iters=25000,
                    crop_size=INPUT_SIZE),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

