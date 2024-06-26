import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as tvtsf
import random
import cv2
import torchvision.transforms.functional as tf
from PIL import Image
from skimage.segmentation import slic,felzenszwalb
from collections import Counter
import matplotlib.pyplot as plt


def my_transform1(image1, image2, label, slic1, slic2):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = tvtsf.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image1 = image1.rotate(angle)
    image2 = image2.rotate(angle)
    label = label.rotate(angle)
    slic1 = slic1.rotate(angle)
    slic2 = slic2.rotate(angle)
    return image1, image2, label, slic1, slic2

def my_transform2(image1, image2, label, slic1, slic2):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        image1 = tf.hflip(image1)
        image2 = tf.hflip(image2)
        label = tf.hflip(label)
        slic1 = tf.hflip(slic1)
        slic2 = tf.hflip(slic2)
    if random.random() > 0.5:
        image1 = tf.vflip(image1)
        image2 = tf.vflip(image2)
        label = tf.vflip(label)
        slic1 = tf.vflip(slic1)
        slic2 = tf.vflip(slic2)
    return image1, image2, label, slic1, slic2

def my_transform3(image1, image2, label, slic1, slic2):
    # 随机裁剪
    i, j, h, w = tvtsf.RandomResizedCrop.get_params(
    image1, scale=(0.7, 1.0), ratio=(1, 1))
    size = (image1.width, image1.height)
    
    image1 = tf.resized_crop(image1, i, j, h, w, size,interpolation=Image.NEAREST)
    image2 = tf.resized_crop(image2, i, j, h, w, size,interpolation=Image.NEAREST)
    label = tf.resized_crop(label, i, j, h, w, size,interpolation=Image.NEAREST)
    slic1 = tf.resized_crop(slic1, i/2, j/2, h/2, w/2, (128,128),interpolation=Image.NEAREST)
    slic2 = tf.resized_crop(slic2, i/2, j/2, h/2, w/2, (128,128),interpolation=Image.NEAREST)
    return image1, image2, label, slic1, slic2

def tranform_sum(image1, image2, label, slic1, slic2):
    image1, image2, label, slic1, slic2 = my_transform1(image1, image2, label, slic1, slic2)
    image1, image2, label, slic1, slic2 = my_transform2(image1, image2, label, slic1, slic2)
    image1, image2, label, slic1, slic2 = my_transform3(image1, image2, label, slic1, slic2)
    return image1, image2, label, slic1, slic2


# # CDD
# def pytorch_normalzeA(img):
#     normalize = tvtsf.Normalize(mean=[0.35389233, 0.39103368, 0.3430645],
#                                 std=[0.21488518, 0.23309693, 0.20776933])
#     img = normalize(torch.from_numpy(img).float())
#     return img.numpy()
# def pytorch_normalzeB(img):
#     normalize = tvtsf.Normalize(mean=[0.47321945, 0.4985897, 0.46871135],
#                                 std=[0.24216767, 0.25931585, 0.25591645])
#     img = normalize(torch.from_numpy(img).float())
#     return img.numpy()

# SYSU
def pytorch_normalzeA(img):
    normalize = tvtsf.Normalize(mean=[0.39659575, 0.52846196, 0.46540029],
                                std=[0.20213537, 0.15811189, 0.15296703])
    img = normalize(torch.from_numpy(img).float())
    return img.numpy()

def pytorch_normalzeB(img):
    normalize = tvtsf.Normalize(mean=[0.40202364, 0.48766127, 0.39895688],
                                std=[0.18235275, 0.15682769, 0.15437150])
    img = normalize(torch.from_numpy(img).float())
    return img.numpy()

class Dataset(Dataset):

    def __init__(self,img_path,label_path,file_name_txt_path,split_flag, transform=True, seg_num=800, pixel_num_ratio=4e6, ratio=2, compactness=10,slic_path='SLIC_DATA_LEVIR/'):

        self.label_path = label_path
        self.img_path = img_path
        self.img_txt_path = file_name_txt_path
        self.slic_path = slic_path
        self.imgs_path_list = np.loadtxt(self.img_txt_path,dtype=str)
        self.flag = split_flag
        self.transform = transform
        self.img_label_path_pairs = self.get_img_label_path_pairs()
        self.seg_num = seg_num
        self.pixel_num_ratio = pixel_num_ratio
        self.ratio = ratio
        self.compactness = compactness
        self.store_matrix = torch.zeros(257,16000)
    def get_img_label_path_pairs(self):

        img_label_pair_list = {}
        if self.flag =='train':
            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name,image2_name,mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]
                img1_file = os.path.join(self.img_path , image1_name)
                img2_file = os.path.join(self.img_path , image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                slic1_file = os.path.join(self.slic_path, image1_name.split('.')[0]+'.npy')
                slic2_file = os.path.join(self.slic_path, image2_name.split('.')[0]+'.npy')
                
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file,slic1_file,slic2_file,image2_name])

        if self.flag == 'val' or self.flag == 'test':
            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name, image2_name, mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]
                img1_file = os.path.join(self.img_path , image1_name)
                img2_file = os.path.join(self.img_path , image2_name)
                lbl_file = os.path.join(self.label_path , mask_name)
                slic1_file = os.path.join(self.slic_path, image1_name.split('.')[0]+'.npy')
                slic2_file = os.path.join(self.slic_path, image2_name.split('.')[0]+'.npy')
                
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file,slic1_file,slic2_file,image2_name])

        return img_label_pair_list

    def data_transform(self, img1, img2, lbl):
        img1 = img1[:, :, ::-1]  # RGB -> BGR
        img1 = img1.astype(np.float64).transpose(2, 0, 1)
        
        img2 = img2[:, :, ::-1]  # RGB -> BGR
        img2 = img2.astype(np.float64).transpose(2, 0, 1)
        
        lbl = (torch.tensor(lbl)>100).int()
        return img1,img2,lbl

    def slic_online(self,img):
        img = img.transpose(1,2,0)
        slic_index_ = slic(img,n_segments=self.seg_num,compactness=self.compactness)
        # slic_index_ = felzenszwalb(img)
        slic_index = slic_index_.flatten()
        num = max(slic_index)+1
        max_num = max([j for i,j in Counter(list(slic_index)).items()])
        index_set = np.zeros((self.seg_num*2, int(self.pixel_num_ratio/self.seg_num)))
        for i in range(num):
            index_i = np.where(slic_index==i)[0]
            try:
                index_set[i,:len(index_i)] = index_i
            except:
                print(index_i.shape,num,max_num,index_set.shape)
        return index_set, num, max_num
    
#     def slic_offline(self,slic_index_,slic_mask):
#         slic_index_ = (slic_index_+1)*slic_mask
#         slic_index = slic_index_.flatten()
#         num = max(slic_index)
#         max_num = 0
#         index_set = self.index_set.copy()
# #         index_set = np.load('zero.npy')
#         for i in range(num):
#             i = i+1
#             index_i = np.where(slic_index==i)[0]
#             length = len(index_i)
#             index_set[i-1,:len(index_i)] = index_i
#             if length>max_num:
#                 max_num = length
# #             try:
# #                 index_set[i-1,:len(index_i)] = index_i
# #             except:
# #                 print(index_i.shape,num,max_num,index_set.shape)
#         return index_set, num, max_num

    def slic_offline(self,slic_index_,slic_mask):
        slic_index_ = torch.IntTensor(slic_index_)
        slic_index_ = (slic_index_+1)*slic_mask
        slic_index = slic_index_.flatten()
        index_set = self.store_matrix.clone()
        
        num = torch.max(slic_index)
        max_num = 0
#         index_set = torch.zeros(self.seg_num*2, int(self.pixel_num_ratio/self.seg_num))
        
        for i in range(num):
            i = i+1
            index_i = torch.where(slic_index==i)[0]

            length = len(index_i)
            index_set[i,:len(index_i)] = index_i
            if length>max_num:
                max_num = length

        return index_set, num, max_num
    
    def __getitem__(self, index):

        img1_path,img2_path,label_path,slic1_path,slic2_path,filename = self.img_label_path_pairs[index]
        ####### load images and label #############
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        label = Image.open(label_path)
        
        #process_slic
        slic1 = Image.fromarray(np.load(slic1_path))
        slic2 = Image.fromarray(np.load(slic2_path))
        
#         slic1 = Image.fromarray((slic1/np.max(slic1)*255).astype(np.uint8))
#         slic2 = Image.fromarray((slic2/np.max(slic2)*255).astype(np.uint8))

        if len(np.array(label).shape)==3:
            label = np.array(label)
            label = label[:,:,0]
            label=Image.fromarray(label)
        
        height,width = img1.height, img1.width
        
        if self.transform == True:
            img1, img2, label, slic1, slic2 = tranform_sum(img1, img2, label,slic1,slic2)

        img1 = np.array(img1)
        img2 = np.array(img2)
        label = np.array(label)
        slic1 = np.array(slic1)
        slic2 = np.array(slic2)
        
#         print(slic1[:20,:20])
#         plt.imshow(slic1)
#         plt.show()
        
        train_mask = (np.sum(img1,axis=2)!=0).astype(int)
        
        img1, img2, label = self.data_transform(img1, img2, label)
        
        online_slic = False
        if online_slic:
            
            W, H = img1.shape[1], img1.shape[2]

            img1_ = tf.resize(Image.fromarray(img1.transpose(1,2,0).astype(np.uint8)), size=(W//self.ratio, H//self.ratio))
            img2_ = tf.resize(Image.fromarray(img2.transpose(1,2,0).astype(np.uint8)), size=(W//self.ratio, H//self.ratio))

            img1_ = np.array(img1_).transpose(2,0,1)/255
            img2_ = np.array(img2_).transpose(2,0,1)/255
            
            img1_Seg, num1, max_num1 = self.slic_online(img1_)
            img2_Seg, num2, max_num2 = self.slic_online(img2_)
            
#             img1_Seg, num1, max_num1 = np.ones((256,256)),10,10
#             img2_Seg, num2, max_num2 = np.ones((256,256)),10,10
        else:
            slic_mask = Image.fromarray(train_mask.astype(np.uint8)).resize(size=(128,128),resample=Image.NEAREST)
            slic_mask = torch.IntTensor(np.array(slic_mask))
            img1_Seg, num1, max_num1 = self.slic_offline(slic1,slic_mask)
            img2_Seg, num2, max_num2 = self.slic_offline(slic2,slic_mask)

        img1 = pytorch_normalzeA(img1/255)
        img2 = pytorch_normalzeB(img2/255)
        
        return img1,img2,label,str(filename),train_mask,img1_Seg, num1, max_num1, img2_Seg, num2, max_num2 #,int(height),int(width)

    def __len__(self):

        return len(self.img_label_path_pairs)