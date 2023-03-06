import os
import csv
import pandas as pd
import shutil


#root= '/home/miplab/data/Kaggle_Eyepacs/EyeQ-master'
#csv_path = 'data/Label_EyeQ_train.csv'
#img_path = 'EyeQ_preprocess/original_crop/train'
#save_path = '/home/miplab/data/Kaggle_Eyepacs/preprocessed/train'



def move_to_sub_directories(root, csv, img_path, save_path):
    df = pd.read_csv(os.path.join(root, csv_path))
    for index, row in df.iterrows():
        image_name = row['image']
        image_name = image_name[:-4]+'jpeg'

   
        if os.path.isfile(os.path.join(root, img_path, image_name)):
            class_folder = str(row['quality']) +'_' +str(row['DR_grade'])


           # shutil.move( os.path.join(root, img_path, image_name),    # old_directory/test_file.txt
           #                os.path.join(save_path, class_folder, image_name))   # new_directory/test_file.txt

import random
import os
import math




def create_folder_tree(path):
    # Create files
    dr =  [0,1,2,3,4]   
    grade = [0,1,2]
    
    if os.path.exists(path):
        pass
    else: 
        os.mkdir(path)
    
    for d_rating in dr:
        for q_rating in grade:
            file_name = str(q_rating)+ '_'+str(d_rating)

            new_directory= os.path.join(path, file_name)
            os.mkdir(new_directory)

def split_train_val(root: str, percentage: float, save_path: str, action: str, nums_chosen: list =[-1,-1]):
    '''Used to split the training set into validation and training'''

    for path, dirs, files in os.walk(root):

        dirs.sort()
        for directory, num_chosen in zip(dirs, nums_chosen):
            
            if num_chosen ==-1:
                num_chosen = math.floor(percentage* len(os.listdir(os.path.join(root, directory))))
            
            filenames = random.sample(os.listdir(os.path.join(root, directory)), num_chosen)

            for file in filenames:
                if action =="move":
                    shutil.move(os.path.join(root, directory, file),
                        os.path.join(save_path, directory, file ) )
                 #   print(os.path.join(root, directory, file))
                 #   print(os.path.join(save_path, 'validation', directory, file ) )
                elif action == "copy":
                    shutil.copy(os.path.join(root, directory, file),
                                os.path.join(save_path, directory, file ) )
                else:
                    print("Unsure of action: must be copy or move")


    
    return True

def combine_train_val():
    '''Used to combine the validation and training sets into the same folder structure'''


def merge_folders(root, save_folder_name, folder_list):
    save_path = os.path.join(root, save_folder_name)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        pass


    for folder in folder_list:

        existing_folder_path = os.path.join(root, folder)
        if os.path.exists(existing_folder_path):
            
            for path, dirs, files in os.walk(existing_folder_path):
                
                for file in files:
                    sub_folder = path[-1:]
              
                    shutil.move(os.path.join(path, file),
                                os.path.join(save_path, sub_folder, file))


def copy_subset(root: str, save_path: str, action: str,  percentage: float =0.0, num_chosen:int = -1):
    ''' Used to copy/move some predefined number or a percentage of the root directory
        to a new directory
        '''
    
    for path, dirs, files in os.walk(root):
      
      #  for directory in dirs:
     
        if num_chosen == -1:
     
            num_chosen = math.floor(percentage* len(os.listdir(root)))
               
        filenames = random.sample(os.listdir(root), num_chosen)
        c=12
        for file in filenames:
            if action =="move":
                shutil.move(os.path.join(root,  file),
                       os.path.join(save_path,  file ) )
                 #   print(os.path.join(root, directory, file))
                 #   print(os.path.join(save_path, 'validation', directory, file ) )
            elif action == "copy":
                shutil.copy(os.path.join(root, file),
                            os.path.join(save_path, file ) )
            else:
                count+=1
                print("Unsure of action: must be copy or move")
            #print(count)



def move_to_annotation_directories(root, save_path):
    c=0
    for path, dirs, files in os.walk(root):
        if c<1:
        
            for filename in files:
                if filename.endswith(".xls"):
                    stored_folder = filename[11:17]
                    print(stored_folder)

                    df = pd.read_excel(os.path.join(root, filename))
                
                    for x, retinopathy_grade, mac_grade in zip(df["Image name"], df["Retinopathy grade"], df["Risk of macular edema "]):
                        original_image_path = os.path.join(root, stored_folder, x)
                     
                        
                        if os.path.isfile(original_image_path):
                            

                            shutil.copy(original_image_path,
                                            os.path.join(save_path, "dr", str(retinopathy_grade), x))
            c+=1
        #print(files)


def MCF_subset(root: str, save_path: str, annotations_path: str, reject_threshold:float  ):
    '''Used to create a subset of images based on MCF net annotations of rejects'''
    df = pd.read_csv(annotations_path)

    df1 = df.loc[(df["Reject"]> reject_threshold)] # create sub dataframe containing images with high reject score
    print(df1)


    for path, dirs, files in os.walk(root):
        if files[0].endswith("jpeg"):  # Used to remove the .jpeg from the file
            string_cutoff =-5
        else:
            string_cutoff =-4 # .jpg, .tif, .png
        
        for filename in files:
           
            if filename[:string_cutoff] in df1["image"].values:
                # we are seeing if the file is contained in the high reject datarame
                continue # We don't continue the rest of four loop if filename has high reject score
                            #skips to next loop iteration
            
            original_path = os.path.join(path, filename)
    

            if os.path.isfile(original_path):
                shutil.copy(original_path,
                             os.path.join(save_path,  filename))

def data_class_check(root, annotations_path):
    """
    Used to check if data is within the correct folder subdirectory 
    according to annotation results.
    """

    df = pd.read_csv(annotations_path)

    for path,dirs, files in os.walk(root):
        num_files = len(files)

        if path.endswith("0"):  #non_referable
            df_subset = df.loc[(df["DR_grade"]==0) | (df["DR_grade"]==1)]
            c=1
        elif path.endswith("1"): # referable
            df_subset = df.loc[(df["DR_grade"]==2) | (df["DR_grade"]==3) | (df["DR_grade"]==4) ]
        for filename in files:
            if filename in df_subset["image"].values:
                num_files -=1
    print("We have {} images that are in the wrong directory".format(num_files))

    df_subset = df.loc[(df["quality"] ==2)]
    for path, dirs, files in os.walk(root):
        num_files_reject =0

        if files[0].endswith("jpeg"):  # Used to remove the .jpeg from the file
            string_cutoff =-5
        else:
            string_cutoff =-4 # .jpg, .tif, .png

        for filename in files:
            if filename in df_subset["image"].values:
                num_files_reject+=1
    
    print("We have {} reject images in this folder".format(num_files_reject))

def rename_files(root):
    root = '/home/miplab/data/Kaggle_Eyepacs/EyeQ/EyeQ_dr'
    for path, dirs, files in os.walk(root):
        for filenames in files:
            new_name = filenames[:-4] +'.jpeg'
            print(path)
            print(filenames)
            print(new_name)
            try:
                os.rename(os.path.join(path,filenames), os.path.join(path,new_name ))
                print("Source path renamed to destination path successfully.")
             
            # If Source is a file 
            # but destination is a directory
            except IsADirectoryError:
                print("Source is a file but destination is a directory.")
  
            # If source is a directory
            # but destination is a file
            except NotADirectoryError:
                print("Source is a directory but destination is a file.")
  
            # For permission related errors
            except PermissionError:
                print("Operation not permitted.")
  
            # For other errors
            except OSError as error:
                print(error)

if __name__ == '__main__':     

    data_class_check("/home/miplab/data/Kaggle_Eyepacs/EyeQ/EyeQ_dr/reject_filtered/val/0",
                    "/home/miplab/data/EyeQ-master/data/Label_EyeQ_train.csv" )

 #   x = split_train_val(root = "/home/miplab/data/Kaggle_Eyepacs/train/val_CLAHE",
 #                   save_path = "/home/miplab/data/Kaggle_Eyepacs/train/val_CLAHE_balanced",
 #                   action = "copy",
 #                   percentage = 0.0,
 #                   nums_chosen= [212,212,212,212,212]
 #                )
   # merge_folders('/home/miplab/data/Kaggle_Eyepacs/train/CLAHE', 'train_CLAHE', ['val_CLAHE'])
   #copy_subset('/home/miplab/data/Kaggle_Eyepacs/train/filtered_CLAHE/2_class/train/1',
   #            '/home/miplab/data/Kaggle_Eyepacs/train/filtered_CLAHE/2_class/val/1',
   #               action = "move",
   #               percentage =0.3)



  #MCF_subset(root = '/home/miplab/data/Kaggle_Eyepacs/test/test_full_CLAHE/2_class/1',
  #              save_path ='/home/miplab/data/Kaggle_Eyepacs/test/filtered_CLAHE_full/2_class/referable',
  #              annotations_path = '/home/miplab/data/EyeQ-master/MCF_Net/result/eyepacs/test_qual.csv',
  #              reject_threshold =0.5)

