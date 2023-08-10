import torch
import os
import cv2
import json

import torchvision.transforms as transforms
from tqdm import tqdm

from vgg_yolo import vgg16_bn

os.environ["CUDA_VISIBLE_DEVICES"]="2"
#测试集目录
test_root="/hy-tmp/data/test"
#测试集视频列表
test_video=os.listdir(test_root)
#训练集目录
train_root="/hy-tmp/data/train"
#训练集视频列表
train_video=os.listdir(train_root)


def decoder(pred):
    '''Map the network's output to the bounding boxes in the image

    args: 
        pred (tensor) 1*7*7*11

    return: 
        boxes (tensor) box[[x1,y1,x2,y2]...] Coordinates of bounding boxes within the image
        indexs [...] There's only one category here, so it's labeled as 10
        probs [...] Probability of the bounding box containing a drone
    '''

    boxes=[]
    indexs=[]
    probs=[]

    grid_num=7
    cell_size=1./grid_num

    pred=pred.data
    pred=pred.squeeze(0) #7*7*11

    c1=pred[:,:,4].unsqueeze(2)
    c2=pred[:,:,9].unsqueeze(2)
    c=torch.cat((c1,c2),2)

    mask1=c>0.1
    mask2=(c==c.max()) #选择最大置信度
    #最大值也可能小于阈值
    mask=(mask1+mask2).gt(0)

    for i in range(grid_num):
        for j in range(grid_num):
            #找到两个bouding box中c更大的那个
            b = torch.argmax(c[i, j])  
            if mask[i,j,b]==1:
           
                box=pred[i,j,b*5:b*5+4]
                c_prob=torch.FloatTensor([pred[i,j,b*5+4]])
                xy=torch.FloatTensor([j,i])*cell_size
                box[:2]=box[:2]*cell_size+xy
                box_xy=torch.FloatTensor(box.size())
                box_xy[:2]=box[:2]
                box_xy[2:]=box[:2]+box[2:]
                max_prob,index=torch.max(pred[i,j,10:],0)
                if float((c_prob*max_prob)[0])>0.1:
                    boxes.append(box_xy.view(1,4))
                    indexs.append(index)
                    probs.append(c_prob*max_prob)

    if len(boxes)==0:
        boxes=torch.zeros((1,4))
        probs=torch.zeros(1)
        indexs=torch.zeros(1)
    else:
        boxes=torch.cat(boxes,0) #(n,4)
        probs=torch.cat(probs,0) #(n,)
        indexs=[idx.unsqueeze(0) for idx in indexs]
        indexs=torch.cat(indexs,0) #(n,)
    keep = NMS(boxes,probs)
    return boxes[keep],indexs[keep],probs[keep]

def NMS(boxes,scores,threshold=0.5):
    '''Exclude the remaining bounding boxes predicting the same object

    args: 
        boxes (tensor) [N,4]
        scores (tensor) [N,4] c*prob
    
    return: 
        keep (tensor) Indexs of the remaining selected bounding boxes.
    '''

    x1=boxes[:,0]
    y1=boxes[:,1]
    x2=boxes[:,2]
    y2=boxes[:,3]
    areas=(x2-x1)*(y2-y1)

    _,order=scores.sort(0,descending=True)
    keep=[]
    #从最大置信度开始进行比较
    while order.numel()>0:
        if order.dim()==0:
            i=order.item()
        else:
            i=order[0]
        keep.append(i)

        if order.numel()==1:
            break
    
        xx1=x1[order[1:].clamp(min=x1[i])]
        yy1=y1[order[1:]].clamp(min=y1[i])
        xx2=x2[order[1:]].clamp(max=x2[i])
        yy2=y2[order[1:]].clamp(max=y2[i])

        w=(xx2-xx1).clamp(min=0)
        h=(yy2-yy1).clamp(min=0)
        inter=w*h
        
        #将交并比大于阈值的清0
        ovr=inter/(areas[i]+areas[order[1:]]-inter)
        ids=(ovr<=threshold).nonzero().squeeze()
        if ids.numel()==0:
            break
        order=order[ids+1]
    return torch.LongTensor(keep)

def predict(model,img):
    '''Utilize the model to make predictions on the input image

    args: 
        model Trained model
        img (BGR)
    
    return: 
        result [[(x1,y1),(x2,y2),"drone",prob],...]  Coordinates of the bounding box and the probability of a drone being present
    '''

    result=[]
    h,w,_=img.shape

    img=cv2.resize(img,(448,448))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    transform=transforms.ToTensor()
    img=transform(img)

    img=img[None,:,:,:]
    img=img.cuda()

    pred=model(img)
    pred=pred.cpu()
    boxes,indexs,probs=decoder(pred)

    for i,box in enumerate(boxes):
        x1=int(box[0]*w)
        x2=int(box[2]*w)
        y1=int(box[1]*h)
        y2=int(box[3]*h)
        index=indexs[i]
        index=int(index) # convert LongTensor to int
        prob=probs[i]
        prob=float(prob)
        result.append([(x1,y1),(x2,y2),"drone",prob])
    return result

def test():
    '''test
    '''

    model=vgg16_bn()
    print("--load model--")
    model.load_state_dict(torch.load("/hy-tmp/pth/best20.pth"))
    print("model loaded successfully")
    model.cuda()

    #test
    model.eval()
    print("--predict--")
    
    #预测的json文件存入该文件夹
    if not os.path.exists('test_json'):
        os.makedirs('test_json')

    for video in tqdm(test_video,desc="all video"):
        loc_list=[]
        img_files=os.listdir(os.path.join(test_root,video))
        print(img_files)
        for img_name in tqdm(img_files,desc="video "+video):
            img_path=os.path.join(test_root,video,img_name)
            img=cv2.imread(img_path)
            result=predict(model,img)

            max_index = max(range(len(result)), key=lambda i: result[i][3])
            left_u,right_d,_,_=result[max_index]
            x1,y1=left_u
            x2,y2=right_d
            w=x2-x1
            h=y2-y1
            cx=(x1+x2)/2
            cy=(y1+y2)/2
            loc=[cx,cy,w,h]
            loc_list.append(loc)

        json_filename=f"{video}.json"
        json_path=os.path.join("test_json",json_filename)
        with open(json_path,"w") as json_file:
            json.dump({"res": loc_list}, json_file)
        
    print("predict end")
            #visualize
            #img=cv2.imread(img_path)
            #max_index = max(range(len(result)), key=lambda i: result[i][3])
            #left_up,right_bottom,class_name,prob=result[max_index]
            #color=[128,0,0]
            #cv2.rectangle(img,left_up,right_bottom,color,2)
            # label=class_name+str(round(prob,2))
            # text_size,baseline=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.4,1)
            # p1 = (left_up[0],left_up[1]-text_size[1])
            # cv2.rectangle(img,(p1[0]-2//2,p1[1]-2-baseline),(p1[0]+text_size[0],p1[1]+text_size[1]),color,-1)
            # cv2.putText(img,label,(p1[0],p1[1]+baseline),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1,8)
            #cv2.imwrite('result.jpg',img)

def data_init():
    '''init train data
    '''

    img_files=[]
    json_files=[]
    for video in train_video:
        img_file=[]
        video_path=os.path.join(train_root,video)
        img_file=[os.path.join(video,file) for file in os.listdir(video_path) if file.endswith('.jpg')]
        print(img_file)
        #os.listdir读取的图片是乱序的，按照序号排序
        def extract_number(file_name):
            return int(file_name.split("/")[-1].split(".")[0])   

        sorted_img_file = sorted(img_file, key=extract_number)    
            
        #满足一般情况,可能每个json文件文件名不一样
        json_file=[file for file in os.listdir(video_path) if file.endswith('.json')]
        img_files.extend(sorted_img_file)
        json_files.extend(json_file)

    #每一个视频文件有一个json文件
    assert(len(json_files)==len(train_video))

    #将全部box和label(是否存在无人机)合起来，和上述img_files一一对应
    boxes=[]
    labels=[]

    num=0
    for i in range(len(json_files)):
        json_path=os.path.join(train_root,train_video[i],json_files[i])
        with open(json_path,'r') as f:
            json_data=json.load(f)

            label=json_data["exist"]
            gt_rect=json_data["gt_rect"]

            assert(len(label)==len(gt_rect))

            for j in range(len(label)):
                #将box为空的图片去除
                if(len(gt_rect[j]))==0:
                    img_files.pop(num)
                #每一张图片对应一个box和一个label
                else:
                    labels.append(label[j])             
                    boxes.append([gt_rect[j][0],gt_rect[j][1],gt_rect[j][0]+gt_rect[j][2],gt_rect[j][1]+gt_rect[j][3]])
                    num+=1
    assert(len(img_files)==len(labels))
    
    return img_files,labels,boxes

def compute_acc():
    '''Calculate the accuracy of the training set
    '''

    model=vgg16_bn()
    print("--load model--")
    model.load_state_dict(torch.load("/hy-tmp/pth/best20.pth"))
    print("model loaded successfully")
    model.cuda()

    #test
    model.eval()
    print("--compute acc--")
    
    #init
    img_files,labels,boxes=data_init()

    acc=0.
    #第一项和
    A=0.0
    #第二项和
    B=0.0
    
    img_num=len(img_files)
    for i in tqdm(range(img_num),desc="compute acc"):
        img_path=os.path.join(train_root,img_files[i])
        img=cv2.imread(img_path)

        result=predict(model,img) 
        #只预测一个框
        #assert(len(result)==1)
        
        #compute acc
        max_index = max(range(len(result)), key=lambda i: result[i][3])
        left_u,right_d,_,prob=result[max_index]  
        pt=1 if prob==0 else 0
        #真实标签为0 bounding box值都为0
        if(labels[i])==0:
            A+=pt
        #包含真实标签为1 bounding box都为0 和正常的数据
        else:
            x1_A,y1_A=left_u
            x2_A,y2_A=right_d
            x1_B,y1_B,x2_B,y2_B=boxes[i]
            x1=max(x1_A, x1_B)
            y1=max(y1_A, y1_B)
            x2=min(x2_A, x2_B)
            y2=min(y2_A, y2_B)

            inter=max(0,x2-x1)*max(0,y2-y1)
            areaA=(x2_A-x1_A)*(y2_A-y1_A)
            areaB=(x2_B-x1_B)*(y2_B-y1_B)

            #到这里说明图片中一定存在飞行器
            if areaA!=0 and areaB!=0:
                IoU=inter/(areaA+areaB-inter)
            #受噪声影响判断错误
            if areaA==0 and areaB==0:  
                IoU=0.0
            #不受噪声影响 判断不知道是否正确
            if areaA!=0 and areaA==0:
                continue
            #预测错误
            if areaA==0 and areaB!=0:
                IoU=0.0
            A+=IoU
            B+=pt

    # print(A)
    # print(B)
    acc=A/img_num-0.2*(B/labels.count(1))**0.3
 
    print("acc: " +str(acc))


if __name__ == '__main__':
    #predict acc of train dataset
    compute_acc()
    #predict test dataset
    #test()
