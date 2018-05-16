import torch
from models import ResNet
import transforms

import h5py
import numpy as np
import dense_to_sparse
import time
import utils
from metrics import Result

oheight, owidth = 480, 640
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sparsifier = dense_to_sparse.UniformSampling(800, np.inf)
to_tensor = transforms.ToTensor()

def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])

    return rgb, depth

def val_transform(rgb, depth):
    depth_np = depth

    # perform 1st part of data augmentation
    transform = transforms.Compose([
        # transforms.Resize(240.0 / iheight),
        transforms.CenterCrop((oheight, owidth)),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np

def create_sparse_depth(rgb, depth):
    if sparsifier is None:
        return depth
    else:
        mask_keep = sparsifier.dense_to_sparse(rgb, depth)
        sparse_depth = np.zeros(depth.shape)
        sparse_depth[mask_keep] = depth[mask_keep]
        return sparse_depth

def create_rgbd(rgb, depth):
    sparse_depth = create_sparse_depth(rgb, depth)
    # rgbd = np.dstack((rgb[:,:,0], rgb[:,:,1], rgb[:,:,2], sparse_depth))
    rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
    return rgbd

def predict():
    path = "DepthPrediction/results/nyudepthv2.sparsifier=uar{ns=800,md=inf}.modality=rgbd.arch=resnet50.decoder=upproj.criterion=l1.lr=0.01.bs=8/model_best.pth.tar"
    print("=> loading checkpoint")
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    print("=> loaded checkpoint")

    file = "/home/duan/Datasets/nyudepthv2/val/official/00001.h5"
    rgb, depth = h5_loader(file)
    rgb_np, depth_np = val_transform(rgb, depth)

    input_np = create_rgbd(rgb_np, depth_np)

    target = to_tensor(depth_np).to(device)
    input_tensor = to_tensor(input_np).to(device)
    # while input_tensor.dim() < 3:
    input_tensor = input_tensor.unsqueeze(0)

    start_time = time.time()
    depth_pred = model(input_tensor)
    gpu_time = time.time() - start_time

    result = Result()
    result.evaluate(torch.squeeze(depth_pred), target)
    print('t_GPU={gpu_time:.3f}\t'
          'RMSE={result.rmse:.2f} '
          'MAE={result.mae:.2f} '
          'Delta1={result.delta1:.3f} '
          'REL={result.absrel:.3f} '
          'Lg10={result.lg10:.3f} '.format( gpu_time=gpu_time, result=result, ))

    img_merge = utils.merge_into_row_with_gt(input_tensor[:, :3, :, :], input_tensor[:, 3, :, :], target, depth_pred)
    utils.save_image(img_merge, "pics.png")

