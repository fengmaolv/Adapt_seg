import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import pickle

SEQ = torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]])


class cityscapesDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        
        self.lab_ids = [i_id.strip() for i_id in open("./dataset/cityscapes_list/train_lable.txt")]

        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
      #  print("fengmao1",len(self.img_ids))
        self.files = []
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        self.set = set

        # for split in ["train", "trainval", "val"]:
        with open('./dataset/cityscapes_weakLabel_19class', 'rb') as fp:
            itemlist = pickle.load(fp)
        #print("test",len(itemlist))
        itemlist = itemlist * int(np.ceil(len(self.img_ids)/len(itemlist)))

        #print("test1",len(itemlist))
        count = 0
        for name,labels_map in zip(self.img_ids,self.lab_ids):
            img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            label_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, labels_map))

            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "class_label": itemlist[count]
            })
            count +=1


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
     #   print("cao",label.shape)

        name = datafiles["name"]
        class_label = datafiles["class_label"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
#        print("cao",label.shape)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

   #     for i in range(19):
  #          print(":",np.sum(label_copy==i))
     #   print("fuck",label_copy.shape)
 #       print(class_label*np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]))
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), class_label.copy(), np.array(size), name

