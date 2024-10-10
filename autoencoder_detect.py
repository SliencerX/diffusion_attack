import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T, utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import random
import gym
import sys
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
from autoencoder_models import *
from autoencoder_atari import Dataset

if __name__ == "__main__":
    image_size = 84
    dataset = Dataset('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/pong_pic', image_size)
    short = range(0,5000)
    dataset = torch.utils.data.Subset(dataset, short)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 500, shuffle = False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #net = Norm_3d_15_ae(16,ResidualBlock,84,64).to(device)
    net = torch.load("/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/ae/pong_autoencoder49").to(device)
    diff = torch.nn.MSELoss(reduce= 'sum')
    # loss = 0
    # for i, image in tqdm(enumerate(train_loader)):
    #     image = image.to(device)
    #     loss += diff(image, net(image)).data
    #     print(loss)
    # loss /= len(dataset)
    # print(loss)

    threashold = 0.0002

    to_check = Dataset('/mnt/c/Users/tulan/Desktop/Xiaolin/diffusion_attack/diamond/tmp', image_size)
    loader = torch.utils.data.DataLoader(to_check, batch_size = 1, shuffle = False)


    all_loss = []

    for i, image in tqdm(enumerate(loader)):
        image = image.to(device)
        print(image.shape)
        loss = diff(image,net(image)).data.cpu()
        all_loss.append(loss)
        print(loss)
        if loss > threashold:
            print("detect!")
            cv2.imwrite('./detect_pic/'+str(i)+'.png', (image[0].detach().cpu().numpy().transpose(1,2,0))*255)

    print(np.mean(all_loss))
    print(np.std(all_loss))

