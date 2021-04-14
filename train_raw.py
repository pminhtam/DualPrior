import torch
import argparse
from model.dual_prior import VDN
from torch.utils.data import DataLoader
from loss.loss import loss_fn
import os
from data.data_provider import SingleLoader_raw
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from utils.metric import calculate_psnr
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint,weight_init_kaiming
from model.Discriminator import DiscriminatorLinear
import torch.nn.functional as F
from torch import nn
def train(args):
    # torch.set_num_threads(4)
    # torch.manual_seed(args.seed)
    # checkpoint = utility.checkpoint(args)
    data_set = SingleLoader_raw(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory = True
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = loss_fn
    loss_diss = nn.BCELoss()
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model = VDN(in_channels=4).to(device)
    model_dis = DiscriminatorLinear(in_chn=4).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )
    optimizer_dis = optim.Adam(
        model_dis.parameters(),
        lr=1e-4
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 25, 30, 35, 40, 45, 50], 0.5)

    optimizer.zero_grad()
    optimizer_dis.zero_grad()
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.restart:
        model = weight_init_kaiming(model)
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    else:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', args.load_type)
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            state_dict = checkpoint['state_dict']
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = "model."+ k  # remove `module.`
            #     new_state_dict[name] = v
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            model = weight_init_kaiming(model)
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    for epoch in range(start_epoch, args.epoch):
        for step, data in enumerate(data_loader):
            # print(len(data))
            im_noisy, im_gt = data
            im_noisy = im_noisy.to(device)
            im_gt = im_gt.to(device)
            # print(im_noisy)
            # print(im_gt)
            # print(sigmaMapEst),noise_estimate=Fals
            # print(sigmaMapGt)
            for _ in range(2):
                phi_Z = model(im_noisy)
                # print(pred.size())
                batch_size = len(im_noisy)
                out_noise_1 = phi_Z[:batch_size//2,:4,:,:]
                out_noise_2 = phi_Z[batch_size//2:,:4,:,:]
                im_noise_1 = im_noisy[:batch_size//2,:,:,:]
                im_noise_2 = im_noisy[batch_size//2:,:,:,:]
                im_denoise_1 = phi_Z[:batch_size//2,4:,:,:]
                im_denoise_2 = phi_Z[batch_size//2:,4:,:,:]


                im_noise_1_fake = im_denoise_1 + out_noise_2
                im_noise_2_fake = im_denoise_2 + out_noise_1
                out_dis = model_dis(torch.cat([im_noisy,im_noise_1_fake,im_noise_2_fake]))
                label_dis = torch.cat([torch.zeros(len(im_noisy)),torch.ones(len(im_noisy))]).unsqueeze(1).to(device)
                # print(out_dis)
                # print(label_dis)
                # print(loss_diss(out_dis, label_dis))
                loss_ = 1- loss_diss(out_dis, label_dis)
                # print(loss_)
                loss = 5*criterion(out_noise_1,im_denoise_1,im_noise_1,out_noise_2,im_denoise_2,im_noise_2) + loss_
                # print("loss  : ",loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for _ in range(4):
                phi_Z = model(im_noisy)
                batch_size = len(im_noisy)
                out_noise_1 = phi_Z[:batch_size//2,:4,:,:]
                out_noise_2 = phi_Z[batch_size//2:,:4,:,:]
                im_noise_1 = im_noisy[:batch_size//2,:,:,:]
                im_noise_2 = im_noisy[batch_size//2:,:,:,:]
                im_denoise_phi_1 = im_noise_1 - out_noise_1
                im_denoise_phi_2 = im_noise_2 - out_noise_2
                im_denoise_1 = phi_Z[:batch_size//2,4:,:,:]
                im_denoise_2 = phi_Z[batch_size//2:,4:,:,:]

                im_noise_1_fake = im_denoise_1 + out_noise_2
                im_noise_2_fake = im_denoise_2 + out_noise_1
                out_dis = model_dis(torch.cat([im_noisy,im_noise_1_fake,im_noise_2_fake]))
                label_dis = torch.cat([torch.zeros(len(im_noisy)),torch.ones(len(im_noisy))]).unsqueeze(1).to(device)
                # print(out_dis)
                # print(label_dis)
                loss_ = loss_diss(out_dis,label_dis)
                # print(loss_)

                optimizer_dis.zero_grad()

                loss_.backward()
                optimizer_dis.step()
                # optimizer.zero_grad()

            # average_loss.update(loss)
            if global_step % args.save_every == 0:
                print("Save : epoch ",epoch ," step : ", global_step," with avg loss : ",average_loss.get_value() , ",   best loss : ", best_loss )
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False
                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
            if global_step % args.loss_every == 0:
                # print(im_noisy)
                # print(im_gt)
                print("loss ",loss)
                print("loss_ ",loss_)
                print(global_step, "ori 1 : ", calculate_psnr(im_noisy[:batch_size//2,:,:,:], im_gt[:batch_size//2,:,:,:]))
                print("PSNR 1 : ", calculate_psnr(im_denoise_1, im_gt[:batch_size//2,:,:,:]))
                print( "PSNR phi 1 : ", calculate_psnr(im_denoise_phi_1, im_gt[:batch_size//2,:,:,:]))
                print(global_step, "ori 2 : ", calculate_psnr(im_noisy[batch_size//2:,:,:,:], im_gt[batch_size//2:,:,:,:]))
                print( "PSNR 2 : ", calculate_psnr(im_denoise_2, im_gt[batch_size//2:,:,:,:]))
                print( "PSNR phi 2 : ", calculate_psnr(im_denoise_phi_2, im_gt[batch_size//2:,:,:,:]))
                print(average_loss.get_value())
            global_step += 1
        print("Epoch : ", epoch , "end at step: ", global_step)
        scheduler.step()

    # print(model)
if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise_raw/split/', help='path to noise folder image')
    # parser.add_argument('--noise_dir','-n', default='../image/Noisy', help='path to noise folder image')
    # parser.add_argument('--gt_dir', '-g' , default='/home/dell/Downloads/gt', help='path to gt folder image')
    parser.add_argument('--gt_dir', '-g' , default='/home/dell/Downloads/gt_raw/split/', help='path to gt folder image')
    parser.add_argument('--image_size', '-sz' , default=128, type=int, help='size of image')
    parser.add_argument('--epoch', '-e' ,default=1000, type=int, help='batch size')
    parser.add_argument('--batch_size','-bs' ,  default=2, type=int, help='batch size')
    parser.add_argument('--save_every','-se' , default=200, type=int, help='save_every')
    parser.add_argument('--loss_every', '-le' , default=10, type=int, help='loss_every')
    parser.add_argument('--restart','-r' ,  action='store_true', help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=2, type=int, help='number of workers in data loader')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint/',
                        help='the checkpoint to eval')
    parser.add_argument('--load_type', "-l" ,default="best", type=str, help='Load type best_or_latest ')

    args = parser.parse_args()
    #
    train(args)
