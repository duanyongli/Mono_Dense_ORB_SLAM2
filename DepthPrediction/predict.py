import torch
import transforms
import numpy as np
import time
import os
oheight, owidth = 480, 640
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor

def imageFromArray(array):
    width = array.shape[1]
    a = array[:, 0:width:3]
    b = array[:, 1:width:3]
    c = array[:, 2:width:3]
    a = a[:, :, None]
    b = b[:, :, None]
    c = c[:, :, None]
    img_np = np.concatenate((a,b,c),axis=2)
    return img_np

def val_transform(rgb, depth):
    depth_np = depth

    # perform 1st part of data augmentation
    transform = transforms.Compose([
        # transforms.Resize(240.0 / iheight),
        transforms.CenterCrop((oheight, owidth)),
    ])
    rgb_np = transform(rgb)
    rgb_np = np.array(rgb_np).astype('float32') / 255
    depth_np = transform(depth_np)

    return rgb_np, depth_np


def load_model(inchannels):
    global model
    if inchannels == 3:
        modelpath = "RGBmodel"
    elif inchannels ==4:
        modelpath = "RGBDmodel"
    path = "../../DepthPrediction/results"
    path = os.path.join(path, modelpath)
    path = os.path.join(path, "model_best.pth.tar")
    print("=> loading checkpoint")
    checkpoint = torch.load(path)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()
    print("=> loaded checkpoint")
store = []
def predictRGB(imgarray):
    rgb_np = imageFromArray(imgarray)
    rgb_np = np.array(rgb_np).astype('float32') / 255
    rgb_tensor = torch.from_numpy(rgb_np.transpose((2, 0, 1)))
    input_tensor = rgb_tensor.to(device)
    # while input_tensor.dim() < 3:
    input_tensor = input_tensor.unsqueeze(0)
    store.append(rgb_tensor)
    start_time = time.time()
    depth_pred = model(input_tensor)
    gpu_time = time.time() - start_time
    print('t_GPU={gpu_time:.3f}'.format( gpu_time=gpu_time))

    # img_merge = utils.merge_into_row_with_gt(input_tensor[:, :3, :, :], input_tensor[:, 3, :, :], target, depth_pred)
    # utils.save_image(img_merge, "pics.png")
    depth_pred_tensor = depth_pred.cpu().detach().numpy()
    depth_pred_np = np.squeeze(depth_pred_tensor).astype('float32')
    return depth_pred_np

# def predictRGB(imgarray):
#     global rgb_tensor
#     rgb_np = imageFromArray(imgarray)
#     rgb_np = np.array(rgb_np).astype('float32') / 255
#     rgb_np_CHW = rgb_np.transpose((2, 0, 1))
#     rgb_tensor = torch.Tensor(rgb_np_CHW)
#     store.append(rgb_tensor)
#
#     a = np.random.uniform(0, 3, (480, 640))
#     return a

def predcitRGBD(img, sparsedepth):
    pass

if __name__=='__main__':
    load_model(3)
    input = np.random.uniform(0,256,(480, 1920)).astype(np.int)
    output = predictRGB(input)
    print(output.shape)
