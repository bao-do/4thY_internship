### MODEL
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------FFT ResNet model ------------------------------
class fftConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,input_size):
        super().__init__()
        self.input_size = input_size
        # self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.weights = nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(out_channels,in_channels,*input_size)))
        
    def forward(self,x):
        B, C_in,H,W = x.shape        

        
        # assert (H, W) == self.input_size, "Input size must match the initialized input_size."

        # pad_h = H - self.kernel_size
        # pad_w = W - self.kernel_size

        # --- Pad and center the kernel like fftshift ---
        # kernel_padded = torch.zeros((self.out_channels, self.in_channels, H, W), device=x.device)
        # h_start = (H - self.kernel_size) // 2
        # w_start = (W - self.kernel_size) // 2
        # kernel_padded[:, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size] = self.weights
        x_f = torch.fft.rfft2(x)
        k_f = torch.fft.rfft2(self.weights)
        
        #out_f = torch.einsum('bcij, ocij -> boij',x_f,k_f)  # b c h w 
        out_f = torch.sum(x_f[:,None] * k_f[None,...], dim = 2) 
        out = torch.fft.irfft2(out_f,s=(H,W))
        
        return out



class FFTResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,input_size):
        super().__init__()
        self.conv1 = fftConv2d(in_channels,out_channels,input_size)
        self.conv2 = fftConv2d(out_channels, out_channels,input_size)
        self.batchNorm1 = nn.BatchNorm2d(num_features = out_channels)
        self.batchNorm2 = nn.BatchNorm2d(num_features = out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=1)

    def forward(self, x):
        x_skip = self.conv(x)
        z = self.batchNorm2(self.conv2(F.relu(self.batchNorm1(self.conv1(x)))))
        return F.relu(x_skip + z)
    

class FFTResNet(nn.Module):
    def __init__(self,input_size,c_out_list):
        super().__init__()
        layers = nn.ModuleList()
        self.input_size = input_size
        self.c_out_list = c_out_list
        layers.append(FFTResBlock(1,c_out_list[0],input_size))
        for i in range(len(c_out_list)-1):
            layers.append(FFTResBlock(c_out_list[i],c_out_list[i+1],input_size))
            # layers.append(nn.BatchNorm2d(num_features=c_out_list[i+1]))
        layers.append(FFTResBlock(c_out_list[-1],1,input_size))
        self.layers = nn.Sequential(layers)

    def forward(self,x):
        x = self.layers(x)
        return F.relu(x)
        

class FFTModel(nn.Module):
    def __init__(self,input_size,c_out_list):
        super().__init__()
        self.layers = []
        self.input_size = input_size
        self.layers.append(fftConv2d(in_channels=1,out_channels = c_out_list[0],input_size=input_size))
        self.layers.append(nn.ReLU())
        for i in range(len(c_out_list)-1):
            self.layers.append(fftConv2d(in_channels = c_out_list[i],
                                         out_channels = c_out_list[i+1],
                                         input_size = input_size))
            self.layers.append(nn.BatchNorm2d(num_features = c_out_list[i+1]))
            self.layers.append(nn.ReLU())
        
        self.layers.append(fftConv2d(in_channels = c_out_list[-1],
                                         out_channels =1,
                                         input_size = input_size))
        self.output_activation = nn.ReLU()
        
    
    def forward(self,x):
        # input_tensor = x.clone()
        for layer in self.layers:
            x = layer(x)
        x = self.output_activation(x)
        return  x
    


    # -------------------------CNN ResNet Model----------------------------


class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size = 3):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding_mode='circular', padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding_mode='circular', padding=kernel_size // 2)

        self.batchNorm1 = nn.BatchNorm2d(num_features = out_channels)
        self.batchNorm2 = nn.BatchNorm2d(num_features = out_channels)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode='circular')

    def forward(self, x):
        x_skip = self.conv(x.clone())
        z = self.batchNorm2(self.conv2(F.relu(self.batchNorm1(self.conv1(x)))))
        return F.relu(x_skip + z)

class ResNet(nn.Module):
    def __init__(self, input_size, c_out_list):
        super().__init__()
        layers = []
        self.input_size = input_size
        self.c_out_list = c_out_list
        layers.append(ResBlock(1,c_out_list[0]))
        self.layers = nn.Sequential(*layers)
        
        for i in range(len(c_out_list)-1):
            self.layers.append(ResBlock(c_out_list[i],c_out_list[i+1]))
        self.layers.append(ResBlock(c_out_list[-1],1))

    def forward(self,x):
        x = self.layers(x)
        return F.relu(x)
    

    