import os
import time
import argparse
import torch.autograd
from skimage import io, exposure
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.utils import accuracy, SCDD_eval_all, AverageMeter
#################################
from dataset_process import WUSU as RS
from model_v2.network import MCDNet as Net
DATA_NAME = 'WUSU'
#################################
args = {'gpu_id': '0'}
os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_id']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PredOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        working_path = os.path.dirname(os.path.abspath(__file__))
        parser.add_argument('--pred_batch_size', required=False, default=32, help='prediction batch size')
        parser.add_argument('--test_dir', required=False, default='/TEST_DIR/', help='directory to test images')
        parser.add_argument('--pred_dir', required=False, default=working_path + '/trained_model/' + DATA_NAME+ '_PRED_DIR',
                            help='directory to output masks')
        parser.add_argument('--chkpt_path', required=False, default=working_path + '/trained_model/' + DATA_NAME +
                            '/.pth')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()
        return self.opt


def main():
    begin_time = time.time()
    opt = PredOptions().parse()
    net = Net( num_classes=RS.num_classes, img_size=256).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.load_state_dict(torch.load(opt.chkpt_path))
    net.eval()

    with open('../WUSU/test/test_list.txt', "r") as f:
        test_data_name_list = [data_name.strip() for data_name in f]
    test_set = RS.Data(data_list=test_data_name_list, type='test')

    test_loader = DataLoader(test_set, batch_size=opt.pred_batch_size)
    predict(net, test_set, test_loader, opt.pred_dir, test_data_name_list)
    time_use = time.time() - begin_time
    print('Total time: %.2fs' % time_use)


#  For models with 3 outputs: 1 change map + 2 semantic maps.
#  Parameters: flip->test time
#  augmentation index_map->"False" means rgb results
#  intermediate->whether to output the intermediate maps


def predict(net, pred_set, pred_loader, pred_dir, data_list):
    opt = PredOptions().parse()
    pred_A_dir_rgb = os.path.join(pred_dir, 'im1')
    pred_B_dir_rgb = os.path.join(pred_dir, 'im2')
    pred_A_dir_label = os.path.join(pred_dir, 'label1')
    pred_B_dir_label = os.path.join(pred_dir, 'label2')
    if not os.path.exists(pred_A_dir_rgb): os.makedirs(pred_A_dir_rgb)
    if not os.path.exists(pred_B_dir_rgb): os.makedirs(pred_B_dir_rgb)
    if not os.path.exists(pred_A_dir_label): os.makedirs(pred_A_dir_label)
    if not os.path.exists(pred_B_dir_label): os.makedirs(pred_B_dir_label)

    acc_meter = AverageMeter()
    preds_all = []
    labels_all = []
    inter = 0
    for vi, data in enumerate(pred_loader):
        imgs_A, imgs_B, labels_A, labels_B = data

        labels_bn = torch.eq(labels_A, labels_B).float().to(device).long()
        labels_bn = 1.0 - labels_bn

        imgs_A = imgs_A.to(device).float()
        imgs_B = imgs_B.to(device).float()
        
        labels_A = labels_A.cpu().detach().numpy()
        labels_B = labels_B.cpu().detach().numpy()

        labels_bn = labels_bn.cpu().detach().numpy()
        labels_A = (labels_A + 1) * labels_bn
        labels_B = (labels_B + 1) * labels_bn

        with torch.no_grad(): 
            out_change, outputs_A, outputs_B = net(imgs_A, imgs_B)
            out_change = F.sigmoid(out_change)
                        
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()
        change_mask = out_change.cpu().detach() > 0.5
        change_mask = change_mask.squeeze()
        preds_A = torch.argmax(outputs_A, dim=1).squeeze()
        preds_B = torch.argmax(outputs_B, dim=1).squeeze()

        preds_A = ((preds_A+1)*change_mask.long()).numpy()
        preds_B = ((preds_B+1)*change_mask.long()).numpy()
        for i in range(inter, inter + imgs_A.shape[0]):
            j = i % opt.pred_batch_size
            mask_name = data_list[i]
            pred_A_path = os.path.join(pred_A_dir_rgb, mask_name)
            pred_B_path = os.path.join(pred_B_dir_rgb, mask_name)
            label_A_path = os.path.join(pred_A_dir_label, mask_name)
            label_B_path = os.path.join(pred_B_dir_label, mask_name)

            io.imsave(pred_A_path, RS.Index2Color(pred=preds_A[j]))
            io.imsave(pred_B_path, RS.Index2Color(pred=preds_B[j]))
            io.imsave(label_A_path, RS.Index2Color(pred=labels_A[j]))
            io.imsave(label_B_path, RS.Index2Color(pred=labels_B[j]))
        inter = inter + imgs_A.shape[0]

        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            acc_A, valid_sum_A = accuracy(pred_A, label_A)
            acc_B, valid_sum_B = accuracy(pred_B, label_B)
            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, RS.num_classes + 1)

    print('Fscd: %.2f IoU: %.2f Sek: %.2f Accuracy: %.2f'\
          % (Fscd * 100, IoU_mean * 100, Sek * 100, acc_meter.average() * 100))


if __name__ == '__main__':
    main()
