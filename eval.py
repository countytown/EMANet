import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from dataset import ValDataset 
from metric import fast_hist, cal_scores
from network import EMANet 
import settings
from dataset import scale
from datasets import voc

from torchvision import datasets, transforms
from torchvision.utils import save_image
logger = settings.logger
count = 0

class Session:
    def __init__(self, dt_split):
        torch.cuda.set_device(settings.DEVICE)

        self.log_dir = settings.LOG_DIR
        self.model_dir = settings.MODEL_DIR

        self.net = EMANet(settings.N_CLASSES, settings.N_LAYERS).cuda()
        self.net = DataParallel(self.net, device_ids=[settings.DEVICE])
        dataset = ValDataset(split=dt_split)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                     num_workers=2, drop_last=False)
        self.hist = 0

        self.error_count = 0
    def load_checkpoints(self, name):
        ckp_path = osp.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path,
                             map_location=lambda storage, loc: storage.cuda())
            logger.info('Load checkpoint %s.' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!' % ckp_path)
            return

        self.net.module.load_state_dict(obj['net'])

    def inf_batch(self, image, label):
        image = image.cuda()
        label = label.cuda()

        with torch.no_grad():
            logit = self.net(image)

        pred = logit.max(dim=1)[1] # [1, 377, 505]
        # print(pred.shape,'p1')
        if label.shape != pred.shape:
            label_shape = label.shape[1:]

            float_pred = pred.unsqueeze(0).float()
            pred = F.interpolate(float_pred, label_shape, mode='bilinear',
                                  align_corners=True)
            pred = pred.squeeze(0)
            # print(pred.shape, 'p2')

        self.hist += fast_hist(label, pred)

        # if image.shape != pred.shape:
        #     img_shape = image.shape[2:]
        #
        #     float_pred = pred.unsqueeze(0).float()
        #     pred = F.interpolate(float_pred, img_shape, mode='bilinear',
        #                          align_corners=True)
        #     pred = pred.squeeze(0)
        # fused = image+pred[0]
        # fused = fused.cpu().detach().numpy().transpose(0,2,3,1)
        # print(fused.shape)
        # # cv2.imwrite(r'fused.jpg',fused)
        # plt.imsave('./fuse.png', 0.01*fused[0]+0.5)

        return pred

# def save_images(image, mask, image_file='0'):
#     # Saves the image, the model output and the results after the post processing
#     image_file = str(image_file)
#     image = image.cpu().detach().numpy()[0].transpose(1,2,0)  # 3xhxw
#     image = Image.fromarray(np.uint8(image))
#     mask = mask.cpu().detach().numpy()
#
#     c, h, w = mask.shape
#     mask = mask.flatten()
#     palette = get_voc_palette(num_classes=21)
#     print(image.size,'image')
#     colorized_mask = colorize_mask(mask, palette)
#     print(colorized_mask.size)
#
#     output_im = Image.new('RGB', (w, h))
#     output_im.paste(image)
#     output_im.paste(colorized_mask, (w,0))
#     output_im.save(os.path.join('./outs', image_file+'_colorized.png'))
#     mask_img = Image.fromarray(mask, 'L')
#     mask_img.save(os.path.join('./outs', image_file+'.png'))

def save_images(image, mask, image_file='0'):
    prediction = mask.squeeze_(1).squeeze_(0).cpu().numpy()  # 1xhxw->hxw
    prediction = voc.colorize_mask(prediction)
    prediction.save(os.path.join('./outs_my', str(image_file)[2:-3] + '_mask.png'))
    print(prediction.size,'prediction')


    # np_img = image.cpu().numpy()[0]
    # cv2.imwrite(os.path.join('./outs', str(image_file) + '_ori.png'),np_img)
    # save_image(image,os.path.join('./outs', str(image_file)[2:-3] + '_ori.png'))




def main(ckp_name='final.pth'):
    sess = Session(dt_split='val')
    ckp_name = 'my_semanic_final.pth'
    # ckp_name = 'final.pth'
    sess.load_checkpoints(ckp_name)
    dt_iter = sess.dataloader
    sess.net.eval()
    # for i, [image, label] in enumerate(dt_iter):
    for i, [image, label,image_id] in enumerate(dt_iter):
        pred = sess.inf_batch(image, label)
        save_images(image,pred,image_file=image_id)
        if i % 10 == 0:
            logger.info('num-%d' % i)
            scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
            for k, v in scores.items():
                logger.info('%s-%f' % (k, v))

    scores, cls_iu = cal_scores(sess.hist.cpu().numpy())
    for k, v in scores.items():
        logger.info('%s-%f' % (k, v))
    logger.info('')
    for k, v in cls_iu.items():
        logger.info('%s-%f' % (k, v))


if __name__ == '__main__':
    main()
