'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
from fer import FER2013
from torch.autograd import Variable
from models import *
import matplotlib.pyplot as plt
from torchsummary import summary

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"#后加

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
parser.add_argument('--bs', default=128, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0  # best PublicTest accuracy
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 250

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
# ])
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
PublicTestset = FER2013(split = 'PublicTest', transform=transform_test)
PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=0)
PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=0)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = checkpoint['best_PublicTest_acc']
    best_PrivateTest_acc = checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = checkpoint['best_PublicTest_acc_epoch']
    best_PrivateTest_acc_epoch = checkpoint['best_PrivateTest_acc_epoch']
    start_epoch = checkpoint['best_PrivateTest_acc_epoch'] + 1
else:
    print('==> Building model..')


if use_cuda:
    net.cuda()
# if use_cuda:
#     net = torch.nn.DataParallel(net, device_ids=[0, 1]) 


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch, alpha=1, beta=1.0, cutmix_prob=0.5):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    #net.cuda() 
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        #MixCut
        {



        }
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    Train_acc = 100.*correct/total
    losses=train_loss/(batch_idx+1)
    return losses

def PublicTest(epoch):
    global PublicTest_acc
    global best_PublicTest_acc
    global best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):

        #tencrops
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    PublicTest_loss = PublicTest_loss / (batch_idx + 1)
    PublicTest_acc = 100.*correct/total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch

    return PublicTest_acc, PublicTest_loss
def rand_bbox(size, lan):
     W = size[2]
     H = size[3]
     cut_rat = np.sqrt(1. - lan)
     cut_w = int(W * cut_rat)
     cut_h = int(H * cut_rat)
     # uniform
     cx = np.random.randint(W)
     cy = np.random.randint(H)

     bbx1 = np.clip(cx - cut_w // 2, 0, W)
     bby1 = np.clip(cy - cut_h // 2, 0, H)
     bbx2 = np.clip(cx + cut_w // 2, 0, W)
     bby2 = np.clip(cy + cut_h // 2, 0, H)

     return bbx1, bby1, bbx2, bby2

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):



        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.data.item()
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    PrivateTest_acc = 100.*correct/total
    PrivateTest_loss = PrivateTest_loss / (batch_idx + 1)

    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
	        'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
    	    'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch
    return PrivateTest_acc, PrivateTest_loss

train_loss_list = []
PublicTest_acc_list = []
PublicTest_loss_list = []
PrivateTest_acc_list = []
PrivateTest_loss_list = []

for epoch in range(start_epoch, total_epoch):
    #train(epoch, alpha=1.0, beta=1.0, cutmix_prob=0.5)
    train_loss=train(epoch)
    train_loss_list.append(train_loss)
    with open('losses.txt', 'a') as f:
        f.write(str(train_loss) + '\n')
    plt.plot(train_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(np.arange(0, len(train_loss_list), 100))
    plt.savefig('train_loss.png')
    plt.close()

    #PublicTest(epoch)
    PublicTest_acc, PublicTest_loss=PublicTest(epoch)
    PublicTest_acc_list.append(PublicTest_acc)
    with open('PublicTest_acc.txt', 'a') as f:
        f.write(str(PublicTest_acc) + '\n')
    plt.plot(PublicTest_acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.title('PublicTest_acc')
    plt.xticks(np.arange(0, len(PublicTest_acc_list), 100))
    plt.savefig('PublicTest_acc.png')
    plt.close()

    PublicTest_loss_list.append(PublicTest_loss)
    with open('PublicTest_loss.txt', 'a') as f:
        f.write(str(PublicTest_loss) + '\n')
    plt.plot(PublicTest_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('PublicTest_loss')
    plt.xticks(np.arange(0, len(PublicTest_loss_list), 100))
    plt.savefig('PublicTest_loss.png')
    plt.close()



    #PrivateTest(epoch)
    PrivateTest_acc, PrivateTest_loss=PrivateTest(epoch)
    PrivateTest_acc_list.append(PrivateTest_acc)
    with open('PrivateTest_acc.txt', 'a') as f:
        f.write(str(PrivateTest_acc) + '\n')
    plt.plot(PrivateTest_acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.title('PrivateTest_acc')
    plt.xticks(np.arange(0, len(PrivateTest_acc_list), 100))
    plt.savefig('PrivateTest_acc.png')
    plt.close()

    PrivateTest_loss_list.append(PrivateTest_loss)
    with open('PrivateTest_loss.txt', 'a') as f:
        f.write(str(PrivateTest_loss) + '\n')
    plt.plot(PrivateTest_loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('PrivateTest_loss')
    plt.xticks(np.arange(0, len(PrivateTest_loss_list), 100))
    plt.savefig('PrivateTest_loss.png')
    plt.close()


print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)
summary(net, (3, 44, 44))  # 请根据你的输入图像大小调整参数
