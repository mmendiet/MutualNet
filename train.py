import importlib
import os
import time
import random

import torch
import torch.nn.functional as F
import numpy as np
import ComputePostBN
from utils.setlogger import get_logger

from utils.model_profiling import model_profiling
from utils.config import FLAGS
from utils.datasets import get_dataset
import models.slimmable_ops as slim
from utils import earlybird as eb
import csv

# set log files
saved_path = os.path.join("logs", '{}_{}'.format(FLAGS.model[7:], FLAGS.test_name))
if not os.path.exists(saved_path):
    os.makedirs(saved_path)
logger = get_logger(os.path.join(saved_path, '{}.log'.format('test' if FLAGS.test_only else 'train')))

def set_random_seed():
    """set random seed"""
    if hasattr(FLAGS, 'random_seed'):
        seed = FLAGS.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model():
    """get model"""
    model_lib = importlib.import_module(FLAGS.model)
    model = model_lib.Model(FLAGS.num_classes)
    return model

def get_optimizer(model):
    """get optimizer"""
    # all depthwise convolution (N, 1, x, x) has no weight decay
    # weight decay only on normal conv and fc
    if FLAGS.dataset == 'imagenet1k':
        model_params = []
        for params in model.parameters():
            ps = list(params.size())
            if len(ps) == 4 and ps[1] != 1:  # normal conv
                weight_decay = FLAGS.weight_decay
            elif len(ps) == 2:  # fc
                weight_decay = FLAGS.weight_decay
            else:
                weight_decay = 0
            item = {'params': params, 'weight_decay': weight_decay,
                    'lr': FLAGS.lr, 'momentum': FLAGS.momentum,
                    'nesterov': FLAGS.nesterov}
            model_params.append(item)
        optimizer = torch.optim.SGD(model_params)
    else:
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr,
                                    momentum=FLAGS.momentum, nesterov=FLAGS.nesterov,
                                    weight_decay=FLAGS.weight_decay)
    return optimizer

def profiling(model):
    """profiling on either gpu or cpu"""
    print('Start model profiling')
    overall_csv = []
    for resolution in FLAGS.resolution_list:
        acc_csv = [resolution]
        for subnet_idx, width_mult in enumerate(sorted(FLAGS.width_list, reverse=True)):
            model.apply(lambda m: setattr(m, 'subnet_idx', subnet_idx))
            model.apply(lambda m: setattr(m, 'reso_idx', subnet_idx))
            macs, params = model_profiling(model, resolution, resolution, verbose=False)
            acc_csv.append(round(macs/1e6,2))
        overall_csv.append(acc_csv)
    # Writes results to an easier to read csv
    with open(os.path.join(saved_path,'flops.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(['MFLOPS']+['{:.2f}'.format(width_mult) for width_mult in sorted(FLAGS.width_list, reverse=True)])
        for row in overall_csv:
            w.writerow(row)
    print('Finished model profiling')

def train(epoch, loader, model, criterion, optimizer, lr_scheduler):
    t_start = time.time()
    model.train()
    num_width = len(FLAGS.width_list)
    num_subnets = len(FLAGS.subnet_chnls)
    logger.info('Num of subnets: {}'.format(num_subnets))
    for batch_idx, (input_list, target) in enumerate(loader):
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        # do max width
        model.apply(lambda m: setattr(m, 'subnet_idx', 0))
        max_output = model(input_list[0])
        loss = criterion(max_output, target)
        loss.backward()
        max_output_detach = max_output.detach()
        # do other widths and resolution
        subnet_list = [num_subnets-1]
        if num_subnets > 1:
            sampled_width = random.sample(range(num_subnets), 2)
        else:
            sampled_width = [0,0]
        subnet_list.extend(sampled_width)
        for subnet_idx in sorted(subnet_list, reverse=True):
            reso_idx = random.randint(0, 3)
            model.apply(lambda m: setattr(m, 'subnet_idx', subnet_idx))
            model.apply(lambda m: setattr(m, 'reso_idx', subnet_idx))
            output = model(input_list[reso_idx])
            loss = torch.nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output, dim=1), F.softmax(max_output_detach, dim=1))
            loss.backward()
        if num_subnets != num_width:
            updateBN(model, epoch)
        optimizer.step()
        lr_scheduler.step()
        # print training log
        if batch_idx % FLAGS.print_freq == 0 or batch_idx == len(loader)-1:
            with torch.no_grad():
                loss = criterion(max_output_detach, target).cpu().numpy()
                indices = torch.max(max_output_detach, dim=1)[1]
                acc = (indices == target).sum().cpu().numpy() / indices.size()[0]
                logger.info('TRAIN {:.1f}s LR:{:.4f} {}x Epoch:{}/{} Iter:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                    time.time() - t_start, optimizer.param_groups[0]['lr'], str(FLAGS.width_list[0]), epoch,
                    FLAGS.num_epochs, batch_idx, len(loader), loss, acc))


def validate(epoch, loader, model, criterion, postloader):
    t_start = time.time()
    model.eval()
    resolution = FLAGS.image_size
    with torch.no_grad():
        for subnet_idx in [0]:
            model.apply(lambda m: setattr(m, 'subnet_idx', subnet_idx))
            model.apply(lambda m: setattr(m, 'reso_idx', subnet_idx))
            # model = ComputePostBN.ComputeBN(model, postloader, resolution)
            loss, acc, cnt = 0, 0, 0
            for batch_idx, (input, target) in enumerate(loader):
                input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
                output = model(input)
                loss += criterion(output, target).cpu().numpy() * target.size()[0]
                indices = torch.max(output, dim=1)[1]
                acc += (indices == target).sum().cpu().numpy()
                cnt += target.size()[0]
            logger.info('VAL {:.1f}s {}x Epoch:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                time.time() - t_start, str(1-0.05*subnet_idx), epoch,
                FLAGS.num_epochs, loss/cnt, acc/cnt))

def test(epoch, loader, model, criterion, postloader):
    overall_csv = []
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        for reso_idx, resolution in enumerate(FLAGS.resolution_list):
            print(reso_idx, resolution)
            acc_csv = [resolution]
            for subnet_idx, width_mult in enumerate(sorted(FLAGS.width_list, reverse=True)):
                model.apply(lambda m: setattr(m, 'subnet_idx', subnet_idx))
                model.apply(lambda m: setattr(m, 'reso_idx', subnet_idx))
                model = ComputePostBN.ComputeBN(model, postloader, resolution)
                loss, acc, cnt = 0, 0, 0
                for batch_idx, (input, target) in enumerate(loader):
                    input, target =input.cuda(non_blocking=True), target.cuda(non_blocking=True)
                    output = model(F.interpolate(input, (resolution, resolution), mode='bilinear', align_corners=True))
                    loss += criterion(output, target).cpu().numpy() * target.size()[0]
                    indices = torch.max(output, dim=1)[1]
                    acc += (indices==target).sum().cpu().numpy()
                    cnt += target.size()[0]
                logger.info('VAL {:.1f}s {}x-{} Epoch:{}/{} Loss:{:.4f} Acc:{:.3f}'.format(
                    time.time() - t_start, str(width_mult), str(resolution), epoch,
                    FLAGS.num_epochs, loss/cnt, acc/cnt))
                acc_csv.append(round(acc/cnt*100,2))
            overall_csv.append(acc_csv)
    # Writes results to an easier to read csv
    with open(os.path.join(saved_path,'accuracy.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(['Acc']+['{:.2f}'.format(width_mult) for width_mult in sorted(FLAGS.width_list, reverse=True)])
        for row in overall_csv:
            w.writerow(row)

def check_subnets(model, e):
    cfg_list = e.early_bird_emerge(model, logger, mode=3)
    FLAGS.subnet_chnls = [FLAGS.subnet_chnls[0]] + cfg_list
    slim.cfgs = slim.gen_subnet_cfgs()
    logger.info('CFGs')
    logger.info(str(FLAGS.subnet_chnls))
    with open(saved_path+"/cfgs.txt", "w") as output:
        output.write(str(FLAGS.subnet_chnls))
# additional subgradient descent on the sparsity-induced penalty term
def updateBN(model, epoch):
    # s = args.sparsity * ((math.cos(math.pi * (epoch/FLAGS.num_epochs)) + 1)/2)
    s = FLAGS.sparsity
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d) and m.weight.grad is not None and 'resobn.0' in n:
            m.weight.grad.data.add_(s*torch.sign(m.weight.data))  # L1

def train_val_test():
    """train and val"""
    # seed
    set_random_seed()
    e = eb.EarlyBird([round(1-x,2) for x in sorted(FLAGS.width_list[1:])], epoch_keep=FLAGS.epoch_keep, threshold=FLAGS.threshold)

    # model
    model = get_model()
    model_wrapper = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    train_loader, val_loader = get_dataset()

    # check pretrained
    if FLAGS.pretrained:
        checkpoint = torch.load(FLAGS.pretrained)
        # update keys from external models
        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        new_keys = list(model_wrapper.state_dict().keys())
        old_keys = list(checkpoint.keys())
        new_keys = [key for key in new_keys if 'running' not in key]
        new_keys = [key for key in new_keys if 'tracked' not in key]
        old_keys = [key for key in old_keys if 'running' not in key]
        old_keys = [key for key in old_keys if 'tracked' not in key]
        if not FLAGS.test_only:
            old_keys = old_keys[:-2]
            new_keys = new_keys[:-2]

        new_checkpoint = {}
        for key_new, key_old in zip(new_keys, old_keys):
            new_checkpoint[key_new] = checkpoint[key_old]
        model_wrapper.load_state_dict(new_checkpoint, strict=False)
        print('Loaded model {}.'.format(FLAGS.pretrained))
    optimizer = get_optimizer(model_wrapper)
    # check resume training
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume)
        model_wrapper.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        last_epoch = checkpoint['last_epoch']
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*FLAGS.num_epochs)
        lr_scheduler.last_epoch = last_epoch
        print('Loaded checkpoint {} at epoch {}.'.format(
            FLAGS.resume, last_epoch))
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader)*FLAGS.num_epochs)
        last_epoch = lr_scheduler.last_epoch
        # print model and do profiling
        print(model_wrapper)

    if FLAGS.test_only:
        logger.info('Start testing.')
        profiling(model)
        test(last_epoch, val_loader, model_wrapper, criterion, train_loader)
        return

    logger.info('Start training.')
    for epoch in range(last_epoch + 1, FLAGS.num_epochs):
        if not e.complete:
            check_subnets(model_wrapper, e)
        # train
        train(epoch, train_loader, model_wrapper, criterion, optimizer, lr_scheduler)

        # val
        validate(epoch, val_loader, model_wrapper, criterion, train_loader)

        # lr_scheduler.step()
        torch.save(
            {
                'model': model_wrapper.state_dict(),
                'optimizer': optimizer.state_dict(),
                'last_epoch': epoch,
            },
            os.path.join(saved_path, 'checkpoint.pt'))
    # Test after training
    test(last_epoch, val_loader, model_wrapper, criterion, train_loader)
    profiling(model)
    return


def main():
    """train and eval model"""
    train_val_test()


if __name__ == "__main__":
    main()