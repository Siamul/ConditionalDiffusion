import torch
import pickle as pkl
import numpy as np
from PIL import Image, ImageFilter
import os
import torchvision
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import random

class IrisImageDataset(Dataset):
    def __init__(self, parent_dir_wsd, parent_dir_bxgrid, input_transform, image_resolution):
        super().__init__()
        self.input_size = (image_resolution, image_resolution)
        self.transform = input_transform
        
        self.image_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawData2')
        self.mask_dir_wsd = os.path.join(parent_dir_wsd, 'WarsawMasksCoarse')        
        
        self.image_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'iris')
        self.mask_dir_bxgrid = os.path.join(parent_dir_bxgrid, 'masks_coarse')       
        
        self.pupil_iris_xyrs_path_wsd = os.path.join(parent_dir_wsd, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_wsd, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_wsd = pkl.load(pupilirisxyrsFile)
        
        self.pupil_iris_xyrs_path_bxgrid = os.path.join(parent_dir_bxgrid, 'pupil_iris_xyrs_new.pkl')
        with open(self.pupil_iris_xyrs_path_bxgrid, 'rb') as pupilirisxyrsFile:
            self.pupil_iris_xyrs_bxgrid = pkl.load(pupilirisxyrsFile)
        
        self.images_info = []
        
        for imagename in self.pupil_iris_xyrs_wsd.keys():
            imagepath = os.path.join(self.image_dir_wsd, imagename + '.bmp')
            alpha = float(self.pupil_iris_xyrs_wsd[imagename]['pxyr'][2]) / float(self.pupil_iris_xyrs_wsd[imagename]['ixyr'][2])
            self.images_info.append((imagepath, self.pupil_iris_xyrs_wsd[imagename]['pxyr'], self.pupil_iris_xyrs_wsd[imagename]['ixyr']))
            
        for imagename in self.pupil_iris_xyrs_bxgrid.keys():
            imagepath = os.path.join(self.image_dir_bxgrid, imagename + '.png')
            alpha = float(self.pupil_iris_xyrs_bxgrid[imagename]['pxyr'][2]) / float(self.pupil_iris_xyrs_bxgrid[imagename]['ixyr'][2])
            self.images_info.append((imagepath, self.pupil_iris_xyrs_bxgrid[imagename]['pxyr'], self.pupil_iris_xyrs_bxgrid[imagename]['ixyr']))
    
    def load_image(self, file):
        return Image.fromarray(np.array(Image.open(file).convert('RGB'))[:,:,0], 'L')
    
    def get_class(self, alpha):
        
        if alpha < 0.2:
            return 0
        elif alpha >= 0.2 and alpha < 0.3:
            return 1
        elif alpha >= 0.3 and alpha < 0.4:
            return 2
        elif alpha >= 0.4 and alpha < 0.5:
            return 3
        elif alpha >= 0.5 and alpha < 0.6:
            return 4
        elif alpha >= 0.6 and alpha < 0.7:
            return 5
        else:
            return 6
        
    def __getitem__(self, index):
    
        image_info = self.images_info[index]
        
        inp_img_path = image_info[0]
        inp_img_pupil_xyr = image_info[1]
        inp_img_iris_xyr = image_info[2]
        
        alpha = float(inp_img_pupil_xyr[2]) / float(inp_img_iris_xyr[2])
        inp_img_class = self.get_class(alpha)
        
        inp_img = self.load_image(inp_img_path)
        inp_img = inp_img.crop((inp_img_iris_xyr[0] - inp_img_iris_xyr[2], inp_img_iris_xyr[1] - inp_img_iris_xyr[2], inp_img_iris_xyr[0] + inp_img_iris_xyr[2], inp_img_iris_xyr[1] + inp_img_iris_xyr[2]))
        inp_img = inp_img.resize(self.input_size)
                       
        if random.random() < 0.3: # random brightness or contrast change
            random_degree = np.random.choice([1,2,3,4,5,6,7,8])                
            if random_degree == 1:
                aug = iaa.GammaContrast((0.5, 2.0))
            elif random_degree == 2:
                aug = iaa.LinearContrast((0.4, 1.6))
            elif random_degree == 3:
                aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
            elif random_degree == 4:
                aug = iaa.LogContrast(gain=(0.6, 1.4)) 
            else:    
                aug = iaa.pillike.EnhanceBrightness()
            inp_img_np = aug(images = np.expand_dims(np.array(inp_img), axis=0))
            inp_img = Image.fromarray(inp_img_np[0])
                
        xform_inp_img = self.transform(inp_img)
        
        return {"images" : xform_inp_img, "classes" : inp_img_class}
        
    def __len__(self):
        return len(self.images_info)