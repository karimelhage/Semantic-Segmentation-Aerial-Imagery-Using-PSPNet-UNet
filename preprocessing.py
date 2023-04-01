import numpy as np
import cv2 #For Image Processing
from PIL import Image #For Image Processing
import glob #For reading and storing path directories
import torch
import torchvision.transforms as transforms #Leverage Torch transformation on images
from torch.utils.data import Dataset #To build a torch Dataset
import torch.nn.functional as F
import os
import shutil #to be used to move files between folders

#Generate Augmented Random Image augmentation
def augment_images(image_path, mask_path,parent_dir):
    '''Function to create augmented versions of images and their respective masks at random.
    Augmentation techniques inlcude: brightening/darkening, horizontal/vertical flips and rotation
    techniques.

    Parameters:
    -------------------
    image_path - the path of where the images to be augmented are stored
    mask_path - the path of where the masks of augmented images are to be stored
    parent_dir - The parent directory of where a new folder named 'augmented_images shall be created


    Returns a tuple with with path to the augmented images and path of the related masks

    '''
    img_aug_list = sorted(glob.glob(image_path+'/*')) #Retrieve sorted list of original images in augmnetation folder
    msk_aug_list = sorted(glob.glob(mask_path +'/*')) #retrieve sorted list of original masks in augmentation folder
    
    save_path = os.path.join(parent_dir, 'augmented_images')
        
    if os.path.exists(os.path.join(save_path,'images')) == False:        
        os.makedirs(os.path.join(save_path,'images'))
        
    if os.path.exists(os.path.join(save_path,'masks')) == False:
        os.makedirs(os.path.join(save_path,'masks'))
       
    for index in range(len(img_aug_list)):
        img_id = img_aug_list[index] #retrieve path of 1 image
        mask_id = msk_aug_list[index] #retrieve path of 1 mask
        index_img = img_id.split('/')[-1].split('.')[0] #retrieve name of image
        index_msk = mask_id.split('/')[-1].split('.')[0] #retrieve name of mask

        # Load the image (as tensor)
        shutil.copy(img_id,os.path.join(save_path,'images'))
        shutil.copy(mask_id, os.path.join(save_path,'masks'))
        img = cv2.imread(img_id, 1)  #Read each image as BGR
        mask = cv2.imread(mask_id,1)
        img = Image.fromarray(img) #read image as array
        mask = Image.fromarray(mask) #read mask as array

        #A random number is generated to decide if an image should be darkened ro brightened
        #Random numbers are also generated for whethe ran image is horizontal, vertical flipped
        #or rotated. Notice there is a region in which a random number would not initiate
        #any augmentation. Ideally, we did not want to this. However due, to training speeds
        #we needed to lower the number of newly generated images due to GPU restrictions.
        #Images could however be augmented and saved to at most 4 times with 4 different
        #augmentations

        brighten_darken_decision = np.random.rand() 

        #Darken
        if brighten_darken_decision < 0.25:
            t_darken_image = transforms.ColorJitter(brightness=[0.6, 0.8])
            img_t = t_darken_image(img)
            msk_t = mask #mask unaltered due to no colors
            img_t = np.array(img_t)
            msk_t = np.array(msk_t)
            cv2.imwrite(save_path+'/images/'+index_img+'_darken.tif',img_t)
            cv2.imwrite(save_path+'/masks/'+index_msk+'_darken.png',msk_t)

          #brighten image
        elif brighten_darken_decision > 0.75:
            t_brighten_image = transforms.ColorJitter(brightness=[1.2, 1.4])
            img_t = t_brighten_image(img)
            msk_t = mask #mask unaltered due to no colors
            img_t = np.array(img_t)
            msk_t = np.array(msk_t)
            cv2.imwrite(save_path+'/images/'+index_img+'_brighten.tif',img_t)
            cv2.imwrite( save_path+'/masks/'+index_msk+'_brighten.png',msk_t)

          #Horiontal Flip
        if np.random.rand() < 0.3:
            t_horizonal_flip = transforms.RandomHorizontalFlip(p=1)
            img_t = t_horizonal_flip(img)
            msk_t = t_horizonal_flip(mask)
            img_t = np.array(img_t)
            msk_t = np.array(msk_t)
            cv2.imwrite(save_path+'/images/'+index_img+'_hflip.tif',img_t)
            cv2.imwrite(save_path+'/masks/'+index_msk+'_hflip.png',msk_t)

          #vertical flip
        if np.random.rand() < 0.3:
            t_vertical_flip = transforms.RandomVerticalFlip(p=1)
            img_t  = t_vertical_flip(img)
            msk_t = t_vertical_flip(mask)
            img_t = np.array(img_t)
            msk_t = np.array(msk_t)
            cv2.imwrite(save_path+'/images/'+index_img+'_vflip.tif',img_t)
            cv2.imwrite( save_path+'/masks/'+index_msk+'_vflip.png',msk_t)

          # Rotate
        if np.random.rand() < 0.3:
            t_rotation = transforms.RandomRotation(degrees=(-10, 10)) #minimal degree rotation
            img_t = t_rotation(img)
            msk_t = t_rotation(mask)
            img_t = np.array(img_t)
            msk_t = np.array(msk_t)
            cv2.imwrite( save_path+'/images/'+index_img+'_rot.tif',img_t)
            cv2.imwrite( save_path+'/masks/'+index_msk+'_rot.png',msk_t)

        
    return (os.path.join(save_path,'images'), os.path.join(save_path,'masks'))
    

class patch_dataset(Dataset):
    '''
    Class based on Torch Dataset class that splits large images into a set of smaller patches
    images to caputure more detail with each image
    '''
    def __init__(self, images_path:list, masks_path:list, patch_size_x:int,patch_size_y:int):
        
        self.images_path = images_path #The paths of all the images 
        self.masks_path  = masks_path #The paths of all the masks
        self.patch_size_x = patch_size_x #Patch size in x axis
        self.patch_size_y = patch_size_y #Patch size in y axis
        

    def __getitem__(self, index):
        
        # Select a specific image's path by index
        img_id  = self.images_path[index]
        mask_id = self.masks_path[index]

        transform1 = transforms.Resize([256,256]) #resize image to 256 x 256 
        transform2 = transforms.ToTensor() #convert to tensor

        image = cv2.imread(img_id, 1)  #Read each image as BGR
        mask = cv2.imread(mask_id,1) #Read masks

        if image.shape[0] > 3000: #to ensure we get the same number of patches per image (some images are larger than 4000x3000)
            image = cv2.resize(image, (4000,3000), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(mask, (4000,3000), interpolation = cv2.INTER_AREA)

        SIZE_X = (image.shape[1]//self.patch_size_x)*self.patch_size_x #Nearest size divisible by our patch size
        SIZE_Y = (image.shape[0]//self.patch_size_y)*self.patch_size_y #Nearest size divisible by our patch size

        image = Image.fromarray(image) 
        image = image.crop((0 ,0, SIZE_X, SIZE_Y)) #imaged cropped to size to be divisible by patch sizes


        img = transform2(image) #Image converted to tensor

        msk = transform2(mask) #Mask converted to tensor
        msk = msk[0] #since mask has 1 channel tensor will recreate the same tensor 3 times. So we only take one
        
        img = img.unfold(1, self.patch_size_y, self.patch_size_y).unfold(2,self.patch_size_x,self.patch_size_x) #Retrieve patch of images
        #patches are stored in a manner such that the shape is (num_channels, num_patches_in_W,num_patches_in_H Patch_with,Patch_Height)
        img = img.permute(1, 2, 0, 3, 4).contiguous() #shape rordered so that shape is (num_patches_in_W,num_patches_in_Hnum_channels,patch_W,patch_H)
        img = torch.flatten(img,0,1) #We flatten the first two dimenstions to instead (num_total_patches,num_channels,Patch_W,Patch_H)
        msk = msk.unfold(0, self.patch_size_y, self.patch_size_y).unfold(1,self.patch_size_x,self.patch_size_x) #patches retrieved form masks
        #such that patches are stored as (num_patches_in_W,num_patches_in_H,patch_width,patch_height)

        msk = torch.flatten(msk,0,1) #mask faltten to have shape of (tot_num_patches, patch_width,patch_height)

        img = F.interpolate(img, (256,256)) #image patces resized to 256 x256 (num_patches,3,256,256)
        msk = transform1(msk) #mask patches resized to 256 x 256(num_patches,256,256)
   

        img = img.detach().clone().requires_grad_(True) #require gradients from image tensor
        msk = (msk * 255).long() #convert masks so from between 0 to 1 into integer classes between 0 and 26
        
        return img, msk #return pair of image and mask
    
    def __len__(self):
        return len(self.images_paths) #if length of full dataset called