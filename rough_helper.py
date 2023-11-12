import torch
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
import math
import time
import cv2


"""dataset"""

class FileDataset(torch.utils.data.Dataset):
    def __init__(self, folder, ret=0, pix=1, img_size=None, trim=True):
        self.folder=folder
        self.files=os.listdir(folder)
        self.img_size=img_size
        self.ret=ret
        self.pix=pix
        self.trim=trim

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        torch.cuda.empty_cache()
        file=self.files[idx]
        #print(file)
        img=cv2.imread(self.folder+file, 0)
        shape=img.shape
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=resize(img, self.img_size, trim=self.trim)
        img=denoise(img, ret=self.ret, pix=self.pix)
        x=torch.tensor(img, dtype=torch.float32)
        x=torch.unsqueeze(x, 0)
        #print(img)
        return shape, x

def resize(x, size=None, trim=True):
    #print(x)
    W, H = x.shape
    if trim==True:
        dw=W-size[0]
        dh=H-size[1]
        if dw>0 and dh>0:
            dw=random.randint(0, dw)
            dh=random.randint(0, dh)
            x=x[dw:dw+size[0], dh:dh+size[1]]
        else:
            pass
    else:
        pass
        
    if W>H:
        d=W-H
        d=random.randint(0, d)
        x=x[d:d+H, :]
    elif W<H:
        d=H-W
        d=random.randint(0, d)
        x=x[:, d:d+W]
    else:
        pass
    if size != None:
        x=cv2.resize(x, size, interpolation=cv2.INTER_CUBIC)
    return x

        
def denoise(x, ret=0, pix=1):
    x=cv2.Laplacian(x, cv2.CV_8UC1, ksize=3)
    ret, x=cv2.threshold(x, ret, 1, cv2.THRESH_OTSU)
    #print("ret:{}".format(ret))
    kernel = np.ones((3,3),np.uint8)
    x = cv2.dilate(x,kernel,iterations = pix)
    #x = cv2.erode(x,kernel,iterations = pix)
    #x = cv2.dilate(x,kernel,iterations = 1)
    
    #cv2.imshow("img", x*255)
    #cv2.waitKey(0)
    
    return x
"""critarion"""
def SSIM(x, y,window_size=3):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2


    mu_x = nn.functional.avg_pool2d(x, window_size, 1, padding=1)
    mu_y = nn.functional.avg_pool2d(y, window_size, 1, padding=1)

    sigma_x = nn.functional.avg_pool2d((x  - mu_x)**2, window_size, 1, padding=1)
    sigma_y = nn.functional.avg_pool2d((y - mu_y)** 2, window_size, 1, padding=1)

    sigma_xy = (nn.functional.avg_pool2d((x- mu_x) * (y-mu_y), window_size, 1, padding=1))


    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    loss = torch.clamp((1 - SSIM) , 0, 2)

    return  torch.mean(loss)



"""poolformer"""

class PositionalEncoding(nn.Module):

    def __init__(self, dim, dropout = 0.0, max_len = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, 1, dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, length, dim = x.shape
        #cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        #x = torch.concat([cls_tokens, x], dim = 1)
        pe=torch.transpose(self.pe[:length], 0,1)
        x = x + pe.repeat(batch_size, 1,1)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_num, dropout = 0.0):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.linear_Q = nn.Linear(dim, dim, bias = False)
        self.linear_K = nn.Linear(dim, dim, bias = False)
        self.linear_V = nn.Linear(dim, dim, bias = False)
        self.linear = nn.Linear(dim, dim, bias = False)
        self.soft = nn.Softmax(dim = 3)
        self.dropout = nn.Dropout(dropout)
    def split_head(self, x):
        x = torch.tensor_split(x, self.head_num, dim = 2)
        x = torch.stack(x, dim = 1)
        return x
    def concat_head(self, x):
        x = torch.tensor_split(x, x.size()[1], dim = 1)
        x = torch.concat(x, dim = 3).squeeze(dim = 1)
        return x

    def forward(self, Q, K, V, mask = None):
        Q = self.linear_Q(Q)   #(BATCH_SIZE,word_count,dim)
        K = self.linear_K(K)
        V = self.linear_V(V)

        Q = self.split_head(Q)   #(BATCH_SIZE,head_num,word_count//head_num,dim)
        K = self.split_head(K)
        V = self.split_head(V)

        QK = torch.matmul(Q, torch.transpose(K, 3, 2))
        QK = QK/((self.dim//self.head_num)**0.5)

        if mask is not None:
        #print(f"QK:{np.shape(QK)}, mask:{np.shape(mask)}")
            QK = QK + mask

        softmax_QK = self.soft(QK)
        softmax_QK = self.dropout(softmax_QK)

        QKV = torch.matmul(softmax_QK, V)
        QKV = self.concat_head(QKV)
        QKV = self.linear(QKV)
        return QKV

class FeedForward(nn.Module):

    def __init__(self, dim, dropout = 0.0):
        super().__init__()
        hidden_dim=int(dim*2)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, dim)
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class transblock(nn.Module):

    def __init__(self, dim, head_num, dropout = 0.0):
        super().__init__()
        self.MHA = MultiHeadAttention(dim, head_num)
        self.layer_norm_1 = nn.LayerNorm([dim])
        self.layer_norm_2 = nn.LayerNorm([dim])
        self.FF = FeedForward(dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        Q = K = V = x
        x = self.layer_norm_1(x)
        x = self.MHA(Q, K, V)
        x = self.dropout_1(x)
        x = x + Q
        _x = x.clone()
        x = self.layer_norm_2(x)
        x = self.FF(x)
        x = self.dropout_2(x)
        x = x + _x
        return x


class transformer(nn.Module):
    def __init__(self, channel, layer=6):
        super().__init__()
        #self.conv=nn.Conv2d(channel,channel,2,2)
        self.pe=PositionalEncoding(channel)
        self.transformer = nn.Sequential(*[transblock(channel, channel//4) for _ in range(layer)])
    def forward(self, x):
        
        #x=self.conv(x)
        B, C, W, H = x.size()
        #print(x.size())
        x = torch.reshape(x, [B, C, W*H])
        x = torch.transpose(x, 1, 2)
        x=self.pe(x)
        x=self.transformer(x)
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, [B, C, W, H])
        return x



class ConvProj(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size=2, stride=2, reverse=False):
        super().__init__()


        self.point=nn.Conv2d(in_channel, out_channel, 1, 1, padding=0)
        #print(in_channel, out_channel)
        if reverse==False:
            self.conv2=nn.Conv2d(out_channel, out_channel, kernal_size, stride, groups=out_channel, padding=0)
            self.pool=nn.AvgPool2d(kernal_size, stride=kernal_size, padding=0)
        else:
            self.conv2=nn.ConvTranspose2d(out_channel, out_channel, kernal_size, stride, groups=out_channel, padding=0)
            self.pool=nn.ConvTranspose2d(out_channel, out_channel, kernal_size, stride, groups=out_channel, padding=0)
        self.conv1=nn.Conv2d(in_channel,out_channel, 1, 1, padding=0)
        self.conv3=nn.Conv2d(out_channel, out_channel, 1, 1,  padding=0)
        self.norm1=nn.GroupNorm(1, out_channel)
        self.norm2=nn.GroupNorm(1, out_channel)
    def shortcut(self, x):
        x=self.point(x)
        x=self.pool(x)
        return x
    def forward(self, x):
        _x=self.shortcut(x)
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.norm2(x)
        x+=_x

        return x

class ConvBlock(nn.Module):
    def __init__(self, channel, kernal_size=3, stride=1):
        super().__init__()

        self.shortcut=nn.Sequential(nn.Identity())
        if channel==1:
            self.conv2=nn.Conv2d(channel, channel, kernal_size, stride, padding=kernal_size//2)
            self.conv1=nn.Conv2d(channel, channel, 1, 1, padding=0)
            self.conv3=nn.Conv2d(channel, channel, 1, 1,  padding=0)
            self.gelu=nn.GELU()
            self.norm1=nn.GroupNorm(1, channel)
            self.norm2=nn.GroupNorm(1, channel)
        else:
            self.conv2=nn.Conv2d(channel//2, channel//2, kernal_size, stride, padding=kernal_size//2)
            self.conv1=nn.Conv2d(channel, channel//2, 1, 1, padding=0)
            self.conv3=nn.Conv2d(channel//2, channel, 1, 1,  padding=0)
            self.gelu=nn.GELU()
            self.norm1=nn.GroupNorm(1, channel//2)
            self.norm2=nn.GroupNorm(1, channel//2)


    def forward(self, x):
        _x=self.shortcut(x)
        #print(x.shape)
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.gelu(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x=self.gelu(x)
        x=self.conv3(x)

        x+=_x
        return x

class stage(nn.Module):
    def __init__(self, channel, kernal_size=2, stride=2, reverse=False):
        super().__init__()
        if reverse == False:
            out_channel=channel*2
        else:
            out_channel=channel//2
        
        self.conv1=ConvProj(channel, out_channel, kernal_size, stride=stride, reverse=reverse)
        self.conv2=nn.Sequential(*[ConvBlock(out_channel,kernal_size=3) for _ in range(3)])

    def forward(self, x):
        #print(x.shape)
        x=self.conv1(x)
        x=self.conv2(x)

        return x




class autoencoder(nn.Module):
    def __init__(self, bottleneck=64):
        super().__init__()
        down_channels=[1, 2, 4, 8, 16, 32]
        up_channels=[64, 32, 16, 8, 4, 2]
        self.down=nn.Sequential(*[stage(channel, reverse=False) for channel in down_channels])
        self.up=nn.Sequential(*[stage(channel, reverse=True) for channel in up_channels])
        self.transformer=transformer(up_channels[0], layer=6)
        self.sigmoid=nn.Sigmoid()
        self.point1=nn.Conv2d(up_channels[0], bottleneck, 1, 1,  padding=0)
        self.point2=nn.Conv2d(bottleneck, up_channels[0],  1, 1,  padding=0)

    def forward(self, x):

        x=self.down(x)
        x=self.point1(x)
        x=self.point2(x)
        x=self.transformer(x)
        x=self.up(x)
        x=self.sigmoid(x)

        return x

"""train"""

def train(device):

    batch_size = 1
    epochs = 64
    lr =  1e-5
    valid_rate = 0.25
    bottleneck=64
    img_size= (256, 256)
    trim=True
    model = torch.load("rough_helper.pth").to(device)
    #model=autoencoder(bottleneck=bottleneck).to(device)
    #criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    
    dataset = "/Users/uzuki/gochiusa_faces/gochiuza/"
    dataset = FileDataset(dataset, img_size=img_size, trim=trim)
    #print(len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset)-int(len(dataset)*valid_rate),int(len(dataset)*valid_rate)])
    train_dataloader = torch.utils.data.DataLoader(train_dataset,  batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,  batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    

    try:
        with open("train_loss_log_ae", "r") as tl:
            train_loss_list=tl.read().split()

        train_loss_list=[float(n) for n in train_loss_list]
        with open("valid_loss_log_ae", "r") as vl:
            valid_loss_list=vl.read().split()
        valid_loss_list=[float(n) for n in valid_loss_list]
    except:
        train_loss_list=[]
        valid_loss_list=[]
    print(train_loss_list)
    print(valid_loss_list)
    for epoch in range(len(train_loss_list)+1,epochs+1):

        model.train()
        total_loss_train=0.0
        n=0
        small=0
        for shape, img in tqdm(train_dataloader):
            #print(shape)
            if shape[0]<128 or shape[1]<128:
                #print("small")
                small+=1
                continue
            img=img.to(device)
            
            optimizer.zero_grad()
            output=model(img)
            B, C, Wi, Hi = img.shape
            B, C, Wo, Ho = output.shape
            #img=torch.squeeze(img, dim=1)
            #output=torch.squeeze(output, dim=1)
            #print(img, output)
            loss=-1*SSIM(img, output)
            #loss=criterion(output, img)
            loss.backward()
            total_loss_train+=loss.item()
            #print(f'running loss:{loss.item():5.7}')
            optimizer.step()
            del img, output, loss
            torch.cuda.empty_cache()
            if n%1024==1:
                torch.save(model, "rough_helper.pth")
                #print("saved")
            n+=1
            #break
        torch.save(model, "rough_helper.pth")
        


            #except Exception as e:
                #print(e)

        total_loss_train=total_loss_train/len(train_dataloader)
        scheduler.step()

        #valid
        model.eval()
        total_loss_valid=0.0
        with torch.no_grad():
            for shape, img in tqdm(val_dataloader):
                if shape[0]<128 or shape[1]<128:
                    #print("small")
                    small+=1
                    continue
                img=img.to(device)
                optimizer.zero_grad()
                output=model(img)
                B, C, Wi, Hi = img.shape
                B, C, Wo, Ho = output.shape
                #print(img)
                #img=torch.squeeze(img, dim=1)
                #output=torch.squeeze(output, dim=1)
                #print(img.shape, output.shape)
                loss=-1*SSIM(img, output)
                #loss=criterion(output, img)
                total_loss_valid+=loss.item()
                #print(f'running loss:{loss.item():5.7}')
                del img, output, loss
                torch.cuda.empty_cache()
                #break
        total_loss_valid=total_loss_valid/len(val_dataloader)



        train_loss_list.append(total_loss_train)
        valid_loss_list.append(total_loss_valid)
        with open("train_loss_log_ae", "w") as tl:
            tl.write('\n'.join([str(n) for n in train_loss_list]))
        with open("valid_loss_log_ae", "w") as vl:
            vl.write('\n'.join([str(n) for n in valid_loss_list]))
        print(f'{epoch:3d}:epoch | {total_loss_train:5.7} : train loss | {total_loss_valid:5.7} : valid loss|{small}: too_small')


    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.plot(train_loss_list, color='r', label="train")
    plt.plot(valid_loss_list, color='b', label="valid")
    plt.show()



    return model

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    train(device)