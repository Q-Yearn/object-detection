import os
import cv2
import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self,root,video_files,train,transform):

        print('--data init--')
       
        self.root=root
        self.video_files=video_files
        #训练集做数据增强
        self.train=train
        #将数据转变为tensor类型
        self.transform=transform
        
        #######################################################
        #读取数据
        #将全部视频的图片合在一起
        self.img_files=[]
        self.json_files=[]
        for video in self.video_files:
            img_files=[]
            video_path=os.path.join(self.root,video)
            #这里每一项都是video_name/img_name的形式便于处理
            img_files=[os.path.join(video,file) for file in os.listdir(video_path) if file.endswith('.jpg')]
            print(img_files)
            #os.listdir读取的图片是乱序的，按照序号排序(这因os不同而不同)
            def extract_number(file_name):
                return int(file_name.split("/")[-1].split(".")[0])   

            sorted_img_files = sorted(img_files, key=extract_number)    
            
            #满足一般情况,可能每个json文件文件名不一样
            json_files=[file for file in os.listdir(video_path) if file.endswith('.json')]
            self.img_files.extend(sorted_img_files)
            self.json_files.extend(json_files)
        

        #每一个视频文件有一个json文件
        assert(len(self.json_files)==len(self.video_files))

        #将全部box和label(是否存在无人机)合起来，和上述img_files一一对应
        self.boxes=[]
        self.labels=[]

        num=0
        for i in range(len(self.json_files)):
            json_path=os.path.join(self.root,self.video_files[i],self.json_files[i])
            with open(json_path,'r') as f:
                json_data=json.load(f)

                labels=json_data["exist"]
                gt_rect=json_data["gt_rect"]

                assert(len(labels)==len(gt_rect))

                for j in range(len(labels)):
                    #将box为空的图片去除
                    if(len(gt_rect[j]))==0:
                        self.img_files.pop(num)
                    #每一张图片对应一个box和一个label
                    else:
                        label=[]
                        label.append(labels[j])
                        self.labels.append(torch.tensor(label, dtype=torch.long))
                        boxes=[]  
                        #bounding box的左上坐标和高度宽度                 
                        boxes.append([gt_rect[j][0],gt_rect[j][1],gt_rect[j][2],gt_rect[j][3]])
                        self.boxes.append(torch.Tensor(boxes))
                        num+=1
        #######################################################

        self.num_samples=len(self.img_files)

        #保证最终图片和label、box一一对应
        assert(len(self.img_files)==len(self.labels))

    def __getitem__(self,idx):
        '''Retrieve the information of each image along with its bounding box details

        args: 
            index in img_files

        return: 
            img (tensor)
            target (tensor) 7*7*11
        '''
        img_name=self.img_files[idx]
        #读取图像
        img=cv2.imread(os.path.join(self.root,img_name))
        boxes=self.boxes[idx].clone()
        labels=self.labels[idx].clone()

        #######################################################
        #数据预处理
        #1.数据增强
        if self.train:
            img,boxes=self.random_flip(img, boxes)
        
        #2.bounding box坐标归一化
        h,w,_=img.shape
        boxes/=torch.Tensor([w,h,w,h]).expand_as(boxes)

        #3.opencv读取的图像是BGR
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(448,448))

        #4.转变为便于网络处理的形式
        target=self.encoder(boxes,labels)

        for trans in self.transform:
            img=trans(img)
        #######################################################

        return img,target

    def __len__(self):
        '''
        return: length of img_files
        '''
        
        return self.num_samples
    
    def random_flip(self,img,boxes):
        '''Apply a horizontal flip to the input image
    
        args: 
            img (BGR) 
            boxes [[x,y,w,h],[]]

        return: 
            变换后的img boxes
        '''
        
        if random.random() < 0.5:
            img_lr=np.fliplr(img).copy()
            h,w,_=img.shape
            xmin=w-boxes[:,2]
            xmax=w-boxes[:,0]
            boxes[:,0]=xmin
            boxes[:,2]=xmax
            return img_lr,boxes
        return img,boxes
    
    def encoder(self,boxes,labels):   

        '''Encode the network's output, where the input consists of boxes and their corresponding labels, into a 7x7x11 tensor

        args: 
            boxes (tensor) [[x,y,w,h],[]]  x and y are the coordinates of the top-left corner of the bounding box.
            labels (tensor) [...]
            

        return: 
            target (tensor) 7*7*11 [:4]=[5:9]=[归一化后相对于对应网格左上角的x,y 归一化的w,h]
        '''
        grid_num=7
        target=torch.zeros((grid_num,grid_num,11))
        #每个网格的大小是占原图的一个比例
        cell_size=1./grid_num
        for box in boxes:
            cxcy=box[:2]
            #第i,j个网格
            ij=(cxcy/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]),4]=1
            target[int(ij[1]),int(ij[0]),9]=1
            target[int(ij[1]),int(ij[0]),10]=int(labels)
            xy=ij*cell_size #匹配到的网格的左上角相对坐标
            delta_xy=(cxcy-xy)/cell_size
            target[int(ij[1]),int(ij[0]),2:4]=box[2:]
            target[int(ij[1]),int(ij[0]),:2]=delta_xy
            target[int(ij[1]),int(ij[0]),7:9]=box[2:]
            target[int(ij[1]),int(ij[0]),5:7]=delta_xy
        return target    