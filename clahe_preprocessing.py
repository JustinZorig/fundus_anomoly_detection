import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
from PIL import Image

#root = '/home/miplab/data/Kaggle_Eyepacs/train/train_full'
#save_path = '/home/miplab/data/Kaggle_Eyepacs/train/train_full_CLAHE'
#annotations_path = '/home/miplab/data/Kaggle_Eyepacs/train/trainLabels.csv'

root = '/home/miplab/data/Kaggle_Eyepacs/EyeQ/EyeQ_dr/original/good_only'
save_path = '/home/miplab/data/Kaggle_Eyepacs/EyeQ/EyeQ_dr/CLAHE/good_only'
#annotations_path = '/home/miplab/data/Kaggle_Eyepacs/test/retinopathy_solution.csv'

#anno_df = pd.read_csv(annotations_path)
#print(anno_df)


#length = len(anno_df.index)

#for x,y,indx in zip(anno_df['image'], anno_df['level'], range(length)):
 #   x = x+".jpeg"

for path,dirs, files in os.walk(root):
    
    if path.endswith('original'): # skip the root directory
       
        continue
    elif path.endswith("filtered"):
       
        continue
    elif path.endswith("full"):
      
        continue
    elif path.endswith("good_only"):
        continue 

    new_path = save_path + path[64:]

    if os.path.exists(new_path):
        indx =0
        length = len(files)
        for x in files:
            file_path = os.path.join(path, x)
            if os.path.isfile(file_path):
    
                image= cv2.imread(file_path)
                image = cv2.resize(image, (600,600))

                image_bw =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#declatration of CLAHE

                clahe = cv2.createCLAHE(clipLimit = 10) 


                final_img = clahe.apply(image_bw)

                cv2.imwrite(os.path.join(new_path, x), final_img)
                indx+=1
            print("{} out of {}".format(indx, length))

    else:
        os.mkdir(new_path)
    


