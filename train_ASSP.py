import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from PIL import Image
import json
from os.path import join

from model.deeplab_multi_ASSP import Res_Deeplab   ##########
from model.discriminator_ASSP import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 8
ITER_SIZE = 1
NUM_WORKERS = 1
DATA_DIRECTORY = './data/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
INPUT_SIZE = '512,256'            ##########
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
DATA_LIST_PATH_TARGET_VALIDATION = './dataset/cityscapes_list/val.txt'
INPUT_SIZE_TARGET = '512,256'     ##########
COMPARE_SIZE = '512,256'     ##########
#LEARNING_RATE = 2.5e-4
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 250000      
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './snapshots/model_baseline_dropout/GTA5_99000.pth'       ##########
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = './snapshots/model_weakly_ASSP'   ##########
RESULTS_DIR = './result_weakly_ASSP.txt'                  ##########
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_TARGET1 = 0.2

SET = 'train'
SET_VALIDATION = 'val'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-list-target-val", type=str, default=DATA_LIST_PATH_TARGET_VALIDATION,
                        help="Path to the file listing the images in the target dataset for validation.")                       
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--set-val", type=str, default=SET_VALIDATION,
                        help="choose validation set.")           
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help="choose adaptation set.")        
    parser.add_argument("--com-size", type=str, default=COMPARE_SIZE,
                        help="choose adaptation set.")                   
    return parser.parse_args()

args = get_arguments()

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)
    
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
   # print(k.shape,np.max(n * a[k].astype(int) + b[k]))
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    h, w = map(int, args.com_size.split(','))
    com_size = (h, w)

############################
#validation data
    testloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target_val, crop_size=input_size, mean=IMG_MEAN, scale=False, mirror=False, set=args.set_val),
                                    batch_size=1, shuffle=False, pin_memory=True)
    with open('./dataset/cityscapes_list/info.json', 'r') as fp:
        info = json.load(fp)
    mapping = np.array(info['label2train'], dtype=np.int)
    label_path_list = './dataset/cityscapes_list/label.txt'
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join('./data/Cityscapes/data/gtFine/val', x) for x in gt_imgs]

    interp_val = nn.UpsamplingBilinear2d(size=(com_size[1], com_size[0]))

############################

    cudnn.enabled = True

    # Create network
    if args.model == 'DeepLab':
        model = Res_Deeplab(num_classes=args.num_classes)
     #   if args.restore_from[:4] == 'http' :
     #       saved_state_dict = model_zoo.load_url(args.restore_from)
     #   else:
        saved_state_dict = torch.load(args.restore_from)

        #new_params = model.state_dict().copy()
     #   for i in saved_state_dict:
     #       # Scale.layer5.conv2d_list.3.weight
     #       i_parts = i.split('.')
     #       # print i_parts
     #       if not args.num_classes == 19 or not i_parts[1] == 'layer5':
     #           new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(saved_state_dict)


    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)

    model_D1.train()
    model_D1.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(
        cityscapesDataSet(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size_target,
                    scale=False, mirror=args.random_mirror, mean=IMG_MEAN, set=args.set),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    
    bce_loss = torch.nn.BCEWithLogitsLoss()

    interp = nn.UpsamplingBilinear2d(size=(input_size[1], input_size[0]))
    interp_target = nn.UpsamplingBilinear2d(size=(input_size_target[1], input_size_target[0]))

    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)

        for sub_i in range(args.iter_size):

            # train G
            for param in model_D1.parameters():
                param.requires_grad = False
                
            _, batch = next(trainloader_iter)
            images_source, labels, _, _ = batch
            images_source = Variable(images_source).cuda(args.gpu)
            pred1, feature = model(images_source)
            pred1 = interp(pred1)
            loss_seg1 = loss_calc(pred1, labels, args.gpu)

            D_out1 = model_D1(feature)
            loss_D1_source = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(args.gpu))

            _, batch = next(targetloader_iter)
            images_target, _, _ = batch
            images_target = Variable(images_target).cuda(args.gpu)

            _, feature_target = model(images_target)
            D_out1 = model_D1(feature_target)
            loss_D1_target = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(args.gpu))

            loss = loss_seg1  + args.lambda_adv_target1 * (-loss_D1_source - loss_D1_target)
            loss.backward()
            loss_seg_value1 += loss_seg1.data.item()
            loss_adv_target = loss_D1_source + loss_D1_target
            loss_adv_target_value1 = loss_adv_target.data.item()

            optimizer.step()

            # train D
            for param in model_D1.parameters():
                param.requires_grad = True

            pred1, feature = model(images_source)
            feature = feature.detach()
            D_out1 = model_D1(feature)
            loss_D1_source = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(args.gpu))

            _, feature_target = model(images_target)
            feature_target = feature_target.detach()
            D_out1 = model_D1(feature_target)
            loss_D1_target = bce_loss(D_out1, Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D1 = loss_D1_source + loss_D1_target

            loss_D1.backward()

            loss_D_value1 = loss_D1.data.item()
            optimizer_D1.step()


        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_adv1 = {3:.3f}  loss_D1 = {4:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value1, loss_adv_target_value1, loss_D_value1))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))

            hist = np.zeros((19, 19))
            
            f = open(args.results_dir, 'a')
            for index, batch in enumerate(testloader):
                print(index)
                image, _, name = batch
                output1, output2 = model(Variable(image, volatile=True).cuda(args.gpu))
                pred = interp_val(output1)
                pred = pred[0].permute(1,2,0)
          #      print(pred.shape)
                pred = torch.max(pred, 2)[1].byte()
                pred = pred.data.cpu().numpy()
                label = Image.open(gt_imgs[index])
                label = np.array(label.resize(com_size, Image.NEAREST))
                label = label_mapping(label, mapping)
          #      print("fengmao,",np.max(label),np.max(pred))
                hist += fast_hist(label.flatten(), pred.flatten(), 19)
          
            mIoUs = per_class_iu(hist)
            mIoU = round(np.nanmean(mIoUs) * 100, 2)
            print(mIoU)
            f.write('i_iter:{:d},        miou:{:0.3f} \n'.format(i_iter,mIoU))
            f.close()

            
if __name__ == '__main__':
    main()
