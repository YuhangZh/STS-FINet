import os
import time
import torch.nn as nn
import torch.autograd
from torch import optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity, MulticlassCELoss
from utils.utils import accuracy, SCDD_eval_all, AverageMeter

###############################################
from dataset_process import WUSU as SP
from model_v2.network import MCDNet as Net

NET_NAME = 'STS-FINet'
DATA_NAME = 'WUSU'
###############################################

###############################################
args = {
    'train_batch_size': 32,
    'val_batch_size': 64,
    'lr': 0.1,
    'epochs': 50,
    'gpu': True,
    'gpu_id': '0',
    'lr_decay_power': 2.0,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 20,
    'predict_step': 5,
    'trained_model_dir': os.path.join(working_path, 'trained_model', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'trained_model', DATA_NAME, 'pretrained.pth')
}
###############################################
os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['trained_model_dir']): os.makedirs(args['trained_model_dir'])
writer = SummaryWriter(args['log_dir'])


def main():
    net = Net(num_classes=SP.num_classes, img_size=SP.size).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    
    num_params = sum(p.numel() for p in net.parameters())
    num_params_m = num_params / 1e6
    print("parameters：{:.2f} M".format(num_params_m))

    with open('../WUSU/train/train_list.txt', "r") as f:
        data_name_list = [data_name.strip() for data_name in f]
    train_set = SP.Data(data_list=data_name_list, type='train')
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

    with open('../WUSU/test/val_list.txt', "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    val_set = SP.Data(data_list=test_data_name_list, type='test')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)

    # criterion = CrossEntropyLoss2d().to(device)
    criterion = MulticlassCELoss().to(device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    train(train_loader, net, criterion, optimizer, scheduler, val_loader)
    writer.close()
    print('Training finished.')


def train(train_loader, net, criterion, optimizer, scheduler, val_loader):
    bestaccT = 0
    bestmIoU = 0.0
    bestloss = 1.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sc = ChangeSimilarity().to(device)
    curr_epoch = 0
    while True:
        torch.cuda.empty_cache()
        net.train()
        # freeze_model(net.FCN)
        start = time.time()
        acc_meter = AverageMeter()
        train_seg_loss = AverageMeter()
        train_bn_loss = AverageMeter()
        train_sc_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters)
            imgs_A, imgs_B, labels_A, labels_B = data
            if args['gpu']:
                imgs_A = imgs_A.to(device).float()
                imgs_B = imgs_B.to(device).float()
                labels_bn = torch.eq(labels_A, labels_B).float().to(device).float()
                labels_bn = 1.0 - labels_bn
                labels_A = labels_A.to(device).long()
                labels_B = labels_B.to(device).long()

            optimizer.zero_grad()
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)

            assert outputs_A.size()[1] == SP.num_classes

            loss_seg = criterion(outputs_A, labels_A) * 0.5 + criterion(outputs_B, labels_B) * 0.5
            loss_bn = weighted_BCE_logits(out_change, labels_bn)
            loss_sc = criterion_sc(outputs_A[:, 1:], outputs_B[:, 1:], labels_bn)
            loss = loss_seg + loss_bn + loss_sc
            loss.backward()
            optimizer.step()

            labels_A = labels_A.cpu().detach().numpy()
            labels_B = labels_B.cpu().detach().numpy()
            labels_bn = labels_bn.cpu().detach().numpy()
            labels_A = (labels_A + 1) * labels_bn
            labels_B = (labels_B + 1) * labels_bn

            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = ((preds_A + 1) * change_mask.squeeze().long()).numpy()
            preds_B = ((preds_B + 1) * change_mask.squeeze().long()).numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                acc = (acc_A + acc_B) * 0.5
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_seg_loss.update(loss_seg.cpu().detach().numpy())
            train_bn_loss.update(loss_bn.cpu().detach().numpy())
            train_sc_loss.update(loss_sc.cpu().detach().numpy())

            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train seg_loss %.4f bn_loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_seg_loss.val, train_bn_loss.val, acc_meter.val * 100))  # sc_loss %.4f, train_sc_loss.val,
                writer.add_scalar('train seg_loss', train_seg_loss.val, running_iter)
                writer.add_scalar('train sc_loss', train_sc_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        Fscd_v, mIoU_v, Sek_v, acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if mIoU_v > bestmIoU:
            bestmIoU = mIoU_v
            bestaccV = acc_v
            bestloss = loss_v
            torch.save(net.state_dict(),
                       os.path.join(args['trained_model_dir'], NET_NAME + '_%de_mIoU%.2f_Sek%.2f_Fscd%.2f_OA%.2f.pth' \
                                    % (curr_epoch, mIoU_v * 100, Sek_v * 100, Fscd_v * 100, acc_v * 100)))
        print('Total time: %.1fs Best rec: Train acc %.2f, Val mIoU %.2f acc %.2f loss %.4f' % (
        time.time() - begin_time, bestaccT * 100, bestmIoU * 100, bestaccV * 100, bestloss))
        curr_epoch += 1
        # scheduler.step()
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []
    for vi, data in enumerate(val_loader):
        imgs_A, imgs_B, labels_A, labels_B = data
        if args['gpu']:
            imgs_A = imgs_A.to(device).float()
            imgs_B = imgs_B.to(device).float()
            labels_bn = torch.eq(labels_A, labels_B).float().to(device).long()
            labels_bn = 1.0 - labels_bn
            labels_A = labels_A.to(device).long()
            labels_B = labels_B.to(device).long()

        with torch.no_grad():
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            loss_A = criterion(outputs_A, labels_A)
            loss_B = criterion(outputs_B, labels_B)
            loss = loss_A * 0.5 + loss_B * 0.5
        val_loss.update(loss.cpu().detach().numpy())

        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()
        labels_bn = labels_bn.cpu().detach().numpy()
        labels_A = (labels_A + 1) * labels_bn
        labels_B = (labels_B + 1) * labels_bn
        
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = F.sigmoid(out_change).cpu().detach() > 0.5
        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = ((preds_A + 1) * change_mask.squeeze().long()).numpy()
        preds_B = ((preds_B + 1) * change_mask.squeeze().long()).numpy()
        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, SP.num_classes + 1)

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f' \
          % (curr_time, val_loss.average(), Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Fscd', Fscd, curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return Fscd, IoU_mean, Sek, acc_meter.avg, val_loss.avg


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
