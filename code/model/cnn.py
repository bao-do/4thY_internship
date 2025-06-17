### MODEL
import torch
import torch.nn.functional as F
import e2cnn
from e2cnn import gspaces



# ----------------FFT ResNet model ------------------------------
class fftConv2d(torch.nn.Module):
    def __init__(self,in_channels,out_channels,input_size):
        super().__init__()
        self.input_size = input_size
        # self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.weights = torch.nn.Parameter(torch.torch.nn.init.xavier_uniform_(torch.empty(out_channels,in_channels,*input_size)))
        
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



class FFTResBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,input_size):
        super().__init__()
        self.conv1 = fftConv2d(in_channels,out_channels,input_size)
        self.conv2 = fftConv2d(out_channels, out_channels,input_size)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_features = out_channels)
        self.batchNorm2 = torch.nn.BatchNorm2d(num_features = out_channels)
        self.conv = torch.nn.Conv2d(in_channels, out_channels,kernel_size=1)

    def forward(self, x):
        x_skip = self.conv(x)
        z = self.batchNorm2(self.conv2(F.relu(self.batchNorm1(self.conv1(x)))))
        return F.relu(x_skip + z)
    

class FFTResNet(torch.nn.Module):
    def __init__(self,input_size,c_out_list):
        super().__init__()
        layers = []
        self.input_size = input_size
        self.c_out_list = c_out_list
        layers.append(FFTResBlock(1,c_out_list[0],input_size))
        for i in range(len(c_out_list)-1):
            layers.append(FFTResBlock(c_out_list[i],c_out_list[i+1],input_size))
            # layers.append(torch.nn.BatchNorm2d(num_features=c_out_list[i+1]))
        layers.append(FFTResBlock(c_out_list[-1],1,input_size))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self,x):  # forward(self, x, sigma)
        # x = torch.cat([x, sigma_mask], dim = 1)
        x = self.layers(x)
        return F.relu(x)


class FFTResNet_varied_noise(torch.nn.Module):
    def __init__(self,input_size,c_out_list):
        super().__init__()
        layers = []
        self.input_size = input_size
        self.c_out_list = c_out_list
        layers.append(FFTResBlock(2,c_out_list[0],input_size))
        for i in range(len(c_out_list)-1):
            layers.append(FFTResBlock(c_out_list[i],c_out_list[i+1],input_size))
            # layers.append(torch.nn.BatchNorm2d(num_features=c_out_list[i+1]))
        layers.append(FFTResBlock(c_out_list[-1],1,input_size))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, sigma):
        # print("xin chao",x.device,sigma.device)
        sigma_mask = torch.ones(x.size(0),1,28,28,device=x.device)*sigma
        # print(sigma_mask.device)
        x = torch.cat([x, sigma_mask], dim = 1)
        x = self.layers(x)
        return F.relu(x)
        

class FFTModel(torch.nn.Module):
    def __init__(self,input_size,c_out_list):
        super().__init__()
        self.layers = []
        self.input_size = input_size
        self.layers.append(fftConv2d(in_channels=1,out_channels = c_out_list[0],input_size=input_size))
        self.layers.append(torch.nn.ReLU())
        for i in range(len(c_out_list)-1):
            self.layers.append(fftConv2d(in_channels = c_out_list[i],
                                         out_channels = c_out_list[i+1],
                                         input_size = input_size))
            self.layers.append(torch.nn.BatchNorm2d(num_features = c_out_list[i+1]))
            self.layers.append(torch.nn.ReLU())
        
        self.layers.append(fftConv2d(in_channels = c_out_list[-1],
                                         out_channels =1,
                                         input_size = input_size))
        self.output_activation = torch.nn.ReLU()
        
    
    def forward(self,x):
        # input_tensor = x.clone()
        for layer in self.layers:
            x = layer(x)
        x = self.output_activation(x)
        return  x
    


    # -------------------------CNN ResNet Model----------------------------


class ResBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels, kernel_size = 3):
        super().__init__()


        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding_mode='circular', padding=kernel_size // 2)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding_mode='circular', padding=kernel_size // 2)

        self.batchNorm1 = torch.nn.BatchNorm2d(num_features = out_channels)
        self.batchNorm2 = torch.nn.BatchNorm2d(num_features = out_channels)

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, padding_mode='circular')

    def forward(self, x):
        x_skip = self.conv(x.clone())
        z = self.batchNorm2(self.conv2(F.relu(self.batchNorm1(self.conv1(x)))))
        return F.relu(x_skip + z)

class ResNet(torch.nn.Module):
    def __init__(self, input_size, c_out_list):
        super().__init__()
        layers = []
        self.input_size = input_size
        self.c_out_list = c_out_list
        layers.append(ResBlock(1,c_out_list[0]))
        self.layers = torch.nn.Sequential(*layers)
        
        for i in range(len(c_out_list)-1):
            self.layers.append(ResBlock(c_out_list[i],c_out_list[i+1]))
        self.layers.append(ResBlock(c_out_list[-1],1))

    def forward(self,x):
        x = self.layers(x)
        return F.relu(x)
    

# --------------------------Steerable CNN-----------------------------
    

    
class E2ResBlock(torch.nn.Module):
    def __init__(self,feat_type_in,feat_type_out,kernel_size = 3):
        # print(len(feat_type_in),len(feat_type_out))

        super().__init__()
        self.conv1 = e2cnn.nn.R2Conv(feat_type_in, feat_type_out, kernel_size=kernel_size, padding_mode='circular', padding=kernel_size // 2)
        self.conv2 = e2cnn.nn.R2Conv(feat_type_out, feat_type_out, kernel_size=kernel_size, padding_mode='circular', padding=kernel_size // 2)

        self.batchNorm1 = e2cnn.nn.InnerBatchNorm(feat_type_out)
        self.batchNorm2 = e2cnn.nn.InnerBatchNorm(feat_type_out)
        self.feat_type_out = feat_type_out
        self.relu = e2cnn.nn.ReLU(feat_type_out)
        self.conv = torch.nn.Conv2d(len(feat_type_in), len(feat_type_out), kernel_size=1,padding_mode='circular')

        self.reset_parameters()

    def forward(self, x):
        # x_skip = F.pad(x_skip,pad=(1,1,1,1),mode='circular')
        x_skip = self.conv(x.tensor.clone())
        x_skip = e2cnn.nn.GeometricTensor(x_skip,self.feat_type_out)
        z = self.batchNorm2(self.conv2(self.relu(self.batchNorm1(self.conv1(x)))))
        # return self.relu(x_skip + z)
        return x_skip + z

    def reset_parameters(self):
        for p in self.parameters():
            if p.ndim >= 2:
                torch.nn.init.kaiming_normal_(p)

            # torch.nn.init.constant_(p, val = 123)

class E2ResNet(torch.nn.Module):
    def __init__(self, input_size, type_out_list):
        super().__init__()
        self.r2_act = e2cnn.gspaces.Rot2dOnR2(N=8)
        self.feat_type_in_list = [e2cnn.nn.FieldType(self.r2_act,1*[self.r2_act.trivial_repr])]
        self.feat_type_out_list = []
        for type_out in type_out_list:
            self.feat_type_in_list.append(e2cnn.nn.FieldType(self.r2_act,type_out*[self.r2_act.trivial_repr]))
            self.feat_type_out_list.append(e2cnn.nn.FieldType(self.r2_act,type_out*[self.r2_act.trivial_repr]))
        self.feat_type_out_list.append(e2cnn.nn.FieldType(self.r2_act,1*[self.r2_act.trivial_repr]))
        self.input_size = input_size
        layers = []
        
        for i in range(len(type_out_list)+1):
            # print(len(self.feat_type_in_list[i]),len(self.feat_type_out_list[i]))
            layers.append(E2ResBlock(self.feat_type_in_list[i],self.feat_type_out_list[i]))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self,x):
        x = self.layers(x)
        return x
    

    # def reset_parameters(self):
    #     for p in self.parameters():
    #         torch.nn.init.kaiming_uniform_(p)
    #         # torch.nn.init.constant_(p, val = 123)

    #     for p in self.buffers():
    #         torch.nn.init.kaiming_uniform_(p, val = 123)