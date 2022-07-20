from torchvision.transforms import transforms as T
from torchvision.transforms.functional import crop as crop
import torch
import os
from PIL import Image
import pandas as pd
from tqdm import tqdm
from detector import teacher
import numpy as np


# from .config import RESIZE_TO_WIDTH, RESIZE_TO_HEIGHT

class MarvelTrackClassify(torch.utils.data.Dataset):
    '''This dataset can be used to train a classifier or a tracker. 
    
    In track_mode = False, it returns an image and a target ship category class name.

    In track_mode = True, it returns an image and a target boat name. This is useful to train a tracker such as DeepSort. The Marvel Dataset includes many photos of the same boat, which is useful to train an encoder to distinguish between boats.
    
    Parameters
    ---------
    datapath: str
        The path to the directory containing the images.
    transforms: albumentations.Compose
        The transforms to apply to the images.

    Returns
    -------
    torch.utils.data.Dataset
        The dataset which turns an image tensor (720,1280) and class name as a tuple.

    Example
    -------
    In classify mode:

    >>> from pyspeir.pyspeir.datasets.marveldataset import MarvelTrackClassify
    >>> import matplotlib.pyplot as plt
    >>> datapath = '/home/davisac1/marvel_ds'
    >>> mc = MarvelClassify(datapath)
    >>> image, classname = mc[0]
    >>> image = image.permute(1,2,0).numpy()
    >>> plt.figure(dpi=1200)
    >>> plt.imshow(image)
    >>> print(classname)
    Carrier
    [1280x720] image

    In track mode:

    >>> from pyspeir.pyspeir.datasets.marveldataset import MarvelTrackClassify
    >>> import matplotlib.pyplot as plt
    >>> datapath = '/home/davisac1/marvel_ds'
    >>> mc = MarvelTrackClassify(datapath,track_mode=True)
    >>> image, boat_name = mc[0]
    >>> image = image.permute(1,2,0).numpy()
    >>> plt.figure(dpi=1200)
    >>> plt.imshow(image)
    >>> print(boat_name)
    NACC ARGONAUT
    [680x500] image

    Note that the track mode image is cropped based on the object detection from the teacher model.
    '''
    
    def __init__(self, datapath, track_mode=True, model_path=None, transforms=None, threshold=.7):
        self.dataroot = datapath
        self.track_mode = track_mode
        if model_path is not None or self.track_mode:
            self.teacher = teacher(threshold=threshold, model_path = model_path)
        self.transforms = transforms
        

        print('Loading Marvel dataset...This may take a while')
        
        # if self.dataroot is a csv, open it
        if self.dataroot.endswith('.csv'):
            self.df = pd.read_csv(self.dataroot, names=['image_path', 'class_name', 'name'])
            if track_mode:
                self.classes = self.df['name'].unique()
            else:
                self.classes = self.df['class_name'].unique()
        else:
            self.classes = os.listdir(self.dataroot)
            df = pd.DataFrame(columns=['image_path','class_name','name'])
            for class_name in tqdm(self.classes): #class level
                class_path = os.path.join(self.dataroot,class_name)
                for name in tqdm(os.listdir(class_path)): #boat name level
                    name_path = os.path.join(class_path,name)
                    for image in os.listdir(name_path): # image level
                        image_path = os.path.join(name_path,image)
                        # if track_mode == False:
                        df.loc[len(df)] = [image_path,class_name,name]
                        # else:
                        #     df.loc[len(df)] = [image_path,name]
            self.df = df
            df.to_csv('marvel_dataset.csv')


    def __getitem__(self, index):
        item = self.df.iloc[int(index)]
        image_path = item['image_path']
        class_name = item['class_name']
        name = item['name']
        img = Image.open(image_path).convert('RGB')

        if self.transforms is not None:
            #if track_mode, center crop and random crop?
            img = np.array(img)
            transformed = self.transforms(image=img)
            img = transformed
        else:
            img = T.ToTensor()(img)

        if self.track_mode:
            if self.teacher is not None:
                box, _,_ = self.teacher.detect(img)
                if box.numel() != 0:
                    top = int(box[1])
                    left = int(box[0])
                    width = int(box[2] - box[0])
                    height = int(box[3] - box[1])
                    img = crop(img, top, left, height, width)

            return img, name
        else:
            return img, class_name

    def __len__(self):
        return len(self.df)