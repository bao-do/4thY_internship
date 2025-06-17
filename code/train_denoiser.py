# Libraries loading
import importlib
import model.cnn
importlib.reload(model.cnn)
from model.cnn import FFTResNet
# %load_ext autoload
# %autoreload 2

import torch 
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import functional as TF
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import time 

def get_dataloader(batch_size,data_root="./data"):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=data_root,train=True,download=True,transform=transform)
    # test_dataset = datasets.MNIST(root=data_root,train=False,download=True,transform=transform)


    # X = [train_dataset[i][0] for i in range(len(train_dataset))]
    # X = torch.stack(X)
    return DataLoader(train_dataset,batch_size = batch_size, shuffle=True)

    # translated_dataset= []
    # for shift_h in range(28):
    #     for shift_v in range(28):
    #         translated_dataset.append(torch.roll(X,shifts=(shift_h,shift_v),dims=(2,3)))
    # translated_dataset = torch.stack(translated_dataset).reshape(-1,1,28,28)

def show_images(tensor):
    tensor = tensor.detach().cpu().permute(0,2,3,1)
    num_images = tensor.size(0)
    cols = num_images
    rows = 1
    plt.figure(figsize=(2 * num_images, 2))
    for i, img in enumerate(tensor):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)  # Remove `cmap='gray'` for RGB images
        plt.axis('off')  # Hide axes
    plt.tight_layout()
    plt.show()


def train(sigma, n_epoch,batch_size,lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fftRN_model= FFTResNet(input_size=(28,28),c_out_list=[8,16,64,64,16,8])
    optimizer = torch.optim.Adam(fftRN_model.parameters(),lr = lr)
    scheduler = LinearLR(optimizer,0.6,1.,50)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    c_out_list = [8,16,64,64,16,8]
    dataloader = get_dataloader(batch_size)
    print("Training starts")
    fftRN_model.to(device)

    pb_bar = tqdm(range(n_epochs))
    for epoch in pb_bar:
        fftRN_model.train()
        running_loss = 0.0
        for X, _ in dataloader:
            X = X.to(device) 
            Y = X + sigma * torch.randn_like(X)
            optimizer.zero_grad()
            outputs = fftRN_model(Y)
            loss = loss_fn(outputs, X)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss/len(dataloader)
        last_lr = scheduler.get_last_lr()[0]
        pb_bar.set_description(f'Avg loss = {avg_loss:.3e}, lr = {last_lr:.3f}')
        
        scheduler.step()
        with torch.no_grad():
            if epoch%100==0:
                # print(f"Epoch {epoch} Loss : {avg_loss:.6f}, lr: {optimizer.param_groups[0]["lr"]:.6f}")
                show_images(X[:4])
                show_images(Y[:4])
                show_images(outputs[:4])
                torch.save(fftRN_model.state_dict(), 'fft_conv2d_fixed_sigma_whole_dataset.pt')




# python train_denoiser.py --n_epochs 100 --batch_size 32 --sigma 3 --lr 0.005 
if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser() # chatGPT 
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sigma', type=float, default=3)
    parser.add_argument('--lr', type=float, default=0.005)


    args = parser.parse_args()

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    sigma = args.sigma

    train(sigma,n_epochs,batch_size,lr)

