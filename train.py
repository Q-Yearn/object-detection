import  torch
import os
import random
import cv2


from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from vgg_yolo import vgg16_bn
from dataset import MyData
from yoloLoss import yoloLoss

os.environ["CUDA_VISIBLE_DEVICES"]="2"

def train():
    '''train model
    '''

    use_gpu=torch.cuda.is_available()

    #训练集存储目录
    file_root="/hy-tmp/data/train"
    video_files=os.listdir(file_root)

    #将训练接划分为训练集和验证集
    random.shuffle(video_files)
    split_index=int(len(video_files)*0.8)
    train_video=video_files[:split_index]
    vali_video=video_files[split_index:]
    
    #定义网络
    net=vgg16_bn()
    if use_gpu :
        net.cuda()

    #加载预训练模型参数
    print("--load pre-trained model--")
    vgg=models.vgg16_bn(pretrained=True)
    new_state_dict=vgg.state_dict()
    state=net.state_dict()
    for k in new_state_dict.keys():
        if k in state.keys() and k.startswith('features'):
            state[k]=new_state_dict[k]
    net.load_state_dict(state)
    print("pre-trained model loaded successfully")
    
    #初始化数据
    train_dataset=MyData(root=file_root,video_files=train_video,train=True,transform=[transforms.ToTensor()])
    train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=4)
    vali_dataset=MyData(root=file_root,video_files=vali_video,train=False,transform=[transforms.ToTensor()])
    vali_loader=DataLoader(vali_dataset,batch_size=64,shuffle=False,num_workers=4)

    #定义loss
    criterion=yoloLoss(7,2,5,0.5)

    #设置优化器和相关超参数
    learning_rate=0.001
    params=[]
    params_dict=dict(net.named_parameters())
    for key,value in params_dict.items():
        if key.startswith('features'):
            params += [{'params':[value],'lr':learning_rate*1}]
        else:
            params += [{'params':[value],'lr':learning_rate}]
    #论文中的参数设置        
    optimizer=torch.optim.SGD(params,lr=learning_rate,momentum=0.9,weight_decay=5e-4)


    #train
    net.train()
    
    num_epochs=30
    num_iter=0
    best_vali_loss=np.inf

    for epoch in range(num_epochs):

        if epoch==18:
            learning_rate=0.0001
        if epoch==24:
            learning_rate=0.00001
        
        for param_group in optimizer.param_groups:
            param_group['lr']=learning_rate

        print('\nStarting epoch %d / %d' % (epoch+1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss=0.0

        for i,(images,target) in enumerate(train_loader):
            if use_gpu:
                images,target=images.cuda(),target.cuda()
        
            pred=net(images) 

            loss=criterion(pred,target)
            total_loss+=loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%5==0:
                print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                %(epoch+1, num_epochs, i+1, len(train_loader), loss.data.item(), total_loss/(i+1)))
                num_iter+=1

        #evalution
        net.eval()
        validation_loss=0.0
        for i,(images,target) in enumerate(vali_loader):
            if use_gpu:
                images,target=images.cuda(),target.cuda()

            pred=net(images)
            loss=criterion(pred,target)
            validation_loss+=loss.data.item()
        validation_loss/=len(vali_loader)

        if best_vali_loss>validation_loss:
            best_vali_loss=validation_loss
            print('get best test loss %.5f' % best_vali_loss)
        if not os.path.exists('/hy-tmp/augment_nomodify'):
            os.makedirs('/hy-tmp/augment_nomodify')
            torch.save(net.state_dict(),'/hy-tmp/augment_nomodify/best30.pth')

        if not os.path.exists('logs'):
            os.makedirs('logs')

        logfile = open('logs/log.txt', 'w')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
        logfile.flush()      

        save_path=os.path.join('logs','epoch_%02d_valloss_%0.4f_yolo.pth'%(epoch,validation_loss))
        torch.save(net.state_dict(),save_path)


if __name__ == '__main__':
    train()
