# === auto-injected by project cleanup: make src/ importable ===
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, 'src'))
# === end auto-injection ===
import h5py
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from model import NewNet
from scipy.io import loadmat, savemat
from torch.optim import SGD, Adam
import torch.nn.functional as nF
from psutils import *
from torchvision.transforms import Compose
import os
from os.path import join
import torch.nn.parameter as Para
import random
import argparse
import time
from torch.nn import functional as F
import logging
import yaml

class Param(nn.Module):
    def __init__(self, data):
        super(Param, self).__init__()
        self.X = Para.Parameter(data=data.clone())

    def forward(self):
        return self.X

def calculate_mean(numbers):
    if not numbers:
        return 0
    total = sum(numbers)
    count = len(numbers)
    mean = total / count
    return mean

def mypadRepli(x, pd):
    s1, s2, s3, s4 = x.shape
    y = torch.zeros((s1, s2, s3 + 2 * pd, s4 + 2 * pd)).to(x.device).type(x.dtype)
    y[:, :, :pd, :pd] = x[:, :, :1, :1]
    y[:, :, :pd, pd:pd + s4] = x[:, :, :1, :]
    y[:, :, :pd, pd + s4:] = x[:, :, :1, -1:]
    y[:, :, pd:pd + s3, :pd] = x[:, :, :, :1]
    y[:, :, pd:pd + s3, pd:pd + s4] = x
    y[:, :, pd:pd + s3, pd + s4:] = x[:, :, :, -1:]
    y[:, :, pd + s3:, :pd] = x[:, :, -1:, :1]
    y[:, :, pd + s3:, pd:pd + s4] = x[:, :, -1:, :]
    y[:, :, pd + s3:, pd + s4:] = x[:, :, -1:, -1:]
    return y

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
        print(f"create dir: {dir}")
    else:
        print(f"dir: {dir} already exists")

def main():
    parser = argparse.ArgumentParser(description='DVF Reduced Resolution Training (Simple Version)')
    parser.add_argument('--config', default=None, type=str, help='Path to YAML config file')
    parser.add_argument('--sensor', default='WV3', type=str, help='Sensor name (WV2/WV3/QB/GF2)')
    parser.add_argument('--lam1', default=0.06, type=float, help='Lambda 1 parameter')
    parser.add_argument('--lam2', default=0.94, type=float, help='Lambda 2 parameter')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--init', action='store_true', help='Initialize parameters')
    parser.add_argument('--no-init', action='store_true', help='Disable initialization')
    parser.add_argument('--init_epochs', default=6000, type=int, help='Number of initialization epochs')
    parser.add_argument('--train_epochs', default=3000, type=int, help='Number of training epochs')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--init_model', default='DCFNet', type=str, help='Initial model name')
    parser.add_argument('--optw_lr', default=5e-4, type=float, help='Learning rate for wnet optimizer')
    parser.add_argument('--optX_lr', default=0.5, type=float, help='Learning rate for X optimizer')
    parser.add_argument('--work_dir', default='./outputs/simple_reduced', type=str, help='Working directory')
    parser.add_argument('--data_dir', default='./Data/ori_data', type=str, help='Data directory')
    parser.add_argument('--xnet_dir', default='./Data/xnet_data', type=str, help='Xnet data directory')
    
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            for key, value in config.items():
                if key not in ['max_value', 'channel', 'bands'] and getattr(args, key, None) is not None:
                    setattr(args, key, value)
    
    args.init = not args.no_init
    
    device = torch.device(args.device)
    sensor = args.sensor
    rtype = 'reduced'
    
    channel = {'WV2': 8, 'WV3': 8, 'QB': 4, 'GF2': 4}
    max_value = {'WV2': 2047.0, 'WV3': 2047.0, 'QB': 2047.0, 'GF2': 1023.0}
    
    set_seed(args.seed)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    work_dir = f"{args.work_dir}_{timestamp}_lam1_{args.lam1:.4f}_lam2_{args.lam2:.4f}_{args.init_model}"
    check_makedir(work_dir)
    
    savemat_dir = join(work_dir, 'results')
    check_makedir(savemat_dir)
    
    log_file = join(work_dir, f'training_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting DVF Reduced Resolution Training")
    logger.info(f"Parameters: sensor={sensor}, lam1={args.lam1}, lam2={args.lam2}")
    logger.info(f"Init epochs: {args.init_epochs}, Train epochs: {args.train_epochs}")
    
    data_dir = join(args.data_dir, f'test_{sensor.lower()}_multiExm1.h5')
    logger.info(f"Loading data from: {data_dir}")
    data = h5py.File(data_dir)
    
    gt = np.float32(data['gt'][...] / max_value[sensor])
    ms = np.float32(data['ms'][...] / max_value[sensor])
    pan = np.float32(data['pan'][...] / max_value[sensor])
    
    Xnet_psnr_list = []
    time_list = []
    psnr_list = []
    
    for dindex in range(20):
        logger.info(f'======== Processing Sample {dindex + 1}/20 ========')
        max_X = None
        max_psnr = 0
        gt_index = gt[dindex]
        lrms_index = ms[dindex]
        pan_index = pan[dindex]

        Tlrms = torch.from_numpy(lrms_index).unsqueeze(0).to(device)
        Tpan = torch.from_numpy(pan_index).unsqueeze(0).to(device)

        ratio = 4

        lrms_up = np.float32(imresize(lrms_index.transpose(1, 2, 0), scalar_scale=ratio, method='bicubic'))
        Tlrms_up = torch.from_numpy(lrms_up).permute(2, 0, 1).unsqueeze(0).to(device)

        P3D = Tpan.repeat(1, Tlrms.shape[1], 1, 1)

        h = genMTF(ratio, sensor, channel[sensor])
        h = torch.from_numpy(np.float32(h)).permute(2, 0, 1).unsqueeze(1).to(device)

        ks = h.shape[-1]
        pd = int((ks - 1) / 2)

        X_blur = Compose([partial(mypadRepli, pd=pd), partial(nF.conv2d, weight=h, groups=channel[sensor])])

        s0 = 2
        down = lambda x: x[:, :, s0::ratio, s0::ratio]

        xnet_path = join(args.xnet_dir, f'{sensor.lower()}_{rtype}', args.init_model, 'results', f"output_mulExm_{dindex}.mat")
        logger.info(f"Loading Xnet from: {xnet_path}")
        Xnet = loadmat(xnet_path)
        Xnet = Xnet['sr'] / max_value[sensor]
        Xnet_psnr = my_psnr(Xnet, gt_index.transpose(1, 2, 0))
        Xnet = torch.from_numpy(Xnet).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
        wnet = NewNet(channel[sensor]).to(device)
        optw = Adam(wnet.parameters(), lr=args.optw_lr)

        Xopt = Param(Xnet.detach().clone()).to(device)
        optX = SGD(Xopt.parameters(), lr=args.optX_lr)

        start_time = time.time()    
        
        if args.init:
            logger.info("Starting initialization phase...")
            for i in range(args.init_epochs):
                wnet.train()
                _, W = wnet(Xnet.detach(), Tpan)
                loss = torch.norm(Tlrms_up - W * X_blur(Xnet))
                optw.zero_grad()
                loss.backward()
                optw.step()

                if (i + 1) % 1000 == 0:
                    logger.info(f"Init Iter {i + 1}/{args.init_epochs}  loss: {loss.item():.3e}")

        del optw
        optw = Adam(wnet.parameters(), lr=args.optw_lr)
        
        logger.info("Starting alternating optimization phase...")
        for i in range(args.train_epochs):
            wnet.train()

            X = Xopt()
            pred_pan, w = wnet(X.detach().clone(), Tpan.detach())
            pred_pan, w = pred_pan.detach(), w.detach()
            loss = torch.sum(torch.pow(down(X_blur(X)) - Tlrms.detach(), 2)) + \
                    args.lam1 * torch.sum(torch.pow(X - w * Xnet, 2)) + \
                    args.lam2 * F.mse_loss(pred_pan, P3D)

            optX.zero_grad()
            loss.backward()
            optX.step()

            X = Xopt().detach().clone()
            pred_pan, w = wnet(X, Tpan.detach())

            loss = torch.sum(torch.pow(down(X_blur(X)) - Tlrms.detach(), 2)) + \
                args.lam1 * torch.sum(torch.pow(X - w * Xnet, 2)) + \
                args.lam2 * F.mse_loss(pred_pan, P3D)

            optw.zero_grad()
            loss.backward()
            optw.step()

            X = Xopt().detach().squeeze(0).permute(1, 2, 0).cpu().numpy()

            psnr = my_psnr(X, gt_index.transpose(1, 2, 0))
            
            if (i + 1) % 500 == 0:
                logger.info(f"Train Iter {i + 1}/{args.train_epochs}  PSNR: {psnr:.2f} loss: {loss.item():.4e}")
            
        our_time = time.time() - start_time
        Xnet_psnr_list.append(Xnet_psnr)
        time_list.append(our_time)
        psnr_list.append(max_psnr)
        logger.info(f"Sample {dindex+1} completed - Time: {our_time:.3f}s, max_psnr: {max_psnr:.2f}, Xnet psnr: {Xnet_psnr:.2f}")

        savemat(join(savemat_dir, f'output_mulExm_{dindex}.mat'), {'sr': max_X * max_value[sensor]})
    
    logger.info(f"="*60)
    logger.info(f"Training Summary:")
    logger.info(f"Average PSNR: {calculate_mean(psnr_list):.4f}")
    logger.info(f"Average Xnet PSNR: {calculate_mean(Xnet_psnr_list):.4f}")
    logger.info(f"Average Time: {calculate_mean(time_list):.3f}s")
    logger.info(f"Results saved to: {savemat_dir}")
    logger.info(f"="*60)

if __name__ == "__main__":
    main()