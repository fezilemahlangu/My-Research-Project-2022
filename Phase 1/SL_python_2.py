
#-------------------------------IMPORTS-----------------------#

import zipfile_deflate64 as zipfile 
import numpy as np
import pandas as pd

from io import BytesIO

import PIL.Image



import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split

import time 
import csv 

from PIL import Image, ImageOps
'''
For saving results and models:
'''

def save_results(fieldnames,total_time,call_back_time,test_acc,test_loss):
    '''
    save training results to a csv file 

    @Args:
        fieldnames: field names of the csv file 
        total_time: time taken to train 
        call_back_time: callback logs (time)
        test_acc: test accuracy 
        test_loss: test loss 
    @returns: 
        *nothing* 

    '''
    rows=[
        {
            'total_time' : total_time,
            'call_back_time' : call_back_time.logs,
            'test_acc' : test_acc,
            'test_loss' : test_loss
        }
    ]

    with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/results.csv', 'a', encoding='UTF8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def save_model(fieldnames,first_layer,second_layer,third_layer):
    '''
    saves training model hyperparameters to a csv file 

    @Args:
        fieldnames: field names of the csv file 
        first_layer: hyperparameters of the first layer
        second_layer: hyperparameters of the second layer 
        third_layer: hyperparameters of the third layer
    @returns: 
        *nothing* 

    '''
    rows=[
        {
            'first_layer' : first_layer,
            'second_layer' : second_layer,
            'third_layer' : third_layer

        }
    ]

    with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/models.csv', 'a', encoding='UTF8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)



'''
Functions and Classes:
'''

class timing_Callback(keras.callbacks.Callback):
    '''
    This class takes in the callback of the model and saves the times 
    taken to run each epoch during training in the logs array 
    '''
    def __init__(self):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()
    def on_epoch_end(self,epoch,logs={}):
        self.logs.append(time.time()-self.starttime)

def clean_df(df):
    '''
    Function takes in a dataframe and deletes eye-tracking related attributes.
    Cleans data by removing null values (actions that are null)

    @Args:
        df: dataframe of trial information 
    @returns:
        df: cleaned dataframe
        deleted_rows: array containing deleted rows indices 
    '''
    # r = df.shape[0] #get rows and columns of dataframe 
    c = df.shape[1]
    
    #---dropping gaze position features and reconstructing dataframe---#
    df.drop(df.iloc[:,6:c], inplace=True, axis=1) #dropping gaze position features
    dict={'Unnamed: 0':'frame_id','Unnamed: 1':'episode_id','Unnamed: 2':'score','Unnamed: 3':'duration(ms)','Unnamed: 4':'unclipped_reward','Unnamed: 5':'action'}
    df.rename(columns=dict, inplace=True) #renaming columns 

    #--------cleaning data frame-------#
    deleted_rows= df[df.action.isnull()==True].index.tolist()

    deleted_rows_act = df[df.action > 4].index.tolist()

    df=df.dropna(subset = ['action'])

    df= df.drop(deleted_rows_act)
    
    all_deleted_rows = []
    
    for i in deleted_rows:
      all_deleted_rows.append(i)
    for i in deleted_rows_act:
      all_deleted_rows.append(i)


    return df,all_deleted_rows



def create_train_x(zfile,n,deleted_rows):
    '''
    Function that opens a zipfile of images and returns an array of images. 
    The function also reduces the resolution of the images 

    @Args:
        zfile: zipfile that contains images 

        n: number of images in the zipfile 

        deleted_rows: the images whose information have been deleted through cleaning 

    @returns:
        training_X: array of images in the zip file 
    '''

    # scale_percent = 52 #the images will be reduced by 52%
    archive = zipfile.ZipFile(zfile, 'r') #open zip file
    training_X=[] #array that will store images 

    for i in range(n): #scan through zip file 
        if i not in deleted_rows: #dont open image if it's corresponding row has been deleted 
            filename= BytesIO(archive.read(archive.namelist()[i])) #access an image in the zip file 
            image = PIL.Image.open(filename) #open colour image

            #-------reducing the size/resolution of the image------#
            # width = int(image.size[0] * scale_percent / 100)
            # height = int(image.size[1] * scale_percent / 100)
            width = 84
            height = 84
            dim = (width, height)
            image = ImageOps.grayscale(image) #grayscale 
            image = image.resize(dim,PIL.Image.ANTIALIAS)

            #-----converting image to array and appending to training_X-----#
            image=np.array(image) #convert image to array
            image = image[:,:,None] #(84,84,1) 
            training_X.append(image/255.0) #-> normalized 

    return training_X

def create_train_y(df):
    '''
    This function returns the action column/attribute from the dataframe
    These are the y values 

    @Args:
        df: dataframe containing trial information 
    @returns:
        an array of y values. Actions values taken with image 
    '''

    array = df.to_records(index=False)
    return array['action']

def run_model_config(train_x,train_y,val_x,val_y,test_x,test_y,img_shape,num_classes,first_layer, sec_layer,third_layer,fieldnames_results, fieldnames_model):
    '''
    This function takes in the hyperparameters of the model, builds and runs it.
    It also saves the model hyperparameters and results from training and test scores 
    
    @Args: 
        train_x: x values of the training data set 
        train_y: y values of the training data set 
        val_x: x values of the validation data set 
        val_y: y values of the validation data set 
        test_x: x values of the testing data set 
        test_y: y values of the testing data set 
        img_shape: the shape of the images; 3 by 3
        num_classes: number of actions a player can make 
        first_layer: the hyperparameters of the first layer of the CNN
        second_layer: the hyperparameters of the second layer of the CNN
        fieldnames_results: field names of results.csv
        fieldnames_models: field names of models.csv


    @returns: Nothing 
    '''
    #building model 
    model = keras.Sequential()
    model.add(Conv2D(filters=first_layer[0],kernel_size=first_layer[1],strides=first_layer[2], padding=first_layer[3],activation=first_layer[4],input_shape=img_shape))
    model.add(MaxPooling2D(pool_size=first_layer[5]))
    model.add(Dropout(rate=first_layer[6]))
    if(len(sec_layer)>1):
        model.add(Conv2D(filters=sec_layer[0],kernel_size=sec_layer[1],strides=sec_layer[2],padding=sec_layer[3],activation=sec_layer[4]))
        model.add(MaxPooling2D(pool_size=sec_layer[5]))
        model.add(Dropout(rate=sec_layer[6]))
    
    if(len(third_layer)>1):
        model.add(Conv2D(filters=third_layer[0],kernel_size=third_layer[1],strides=third_layer[2],padding=third_layer[3],activation=third_layer[4]))
        model.add(MaxPooling2D(pool_size=third_layer[5]))
        model.add(Dropout(rate=third_layer[6]))

    model.add(Flatten())
    model.add(Dense(units=first_layer[7], activation = 'relu'))
    model.add(Dropout(rate=first_layer[8]))
    model.add(Dense(num_classes, activation = 'softmax'))

    #compile model
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=first_layer[9]),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

    #early stopping 
    early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', min_delta = 0.00001, patience = 3)

    #run model 
    time_callback = timing_Callback() 
    start = time.time()
    Atari_Conf_History =  model.fit(train_x, train_y, epochs = 25, validation_data=(val_x, val_y), callbacks=[time_callback,early_stopping])
    total_time = time.time()-start 

    #save model hyperparameters to csv
    save_model(fieldnames_model,first_layer,sec_layer,third_layer)

    #save results to csv
    # history = Atari_Conf_History.history 
    test_loss, test_acc = model.evaluate(test_x, test_y)
    save_results(fieldnames_results,total_time,time_callback,test_acc,test_loss)

    return  

def get_data(trial):

    game_folder= str("/home-mscluster/fmahlangu/2089676/atari_breakout_data/")

    zipname=game_folder+str(trial)+".zip"
    textfile=game_folder+str(trial)+".txt"
    csv=game_folder+str(trial)+".csv"

    read_file=pd.read_csv(textfile)
    read_file.to_csv(csv,index=True)
    df= pd.read_csv(csv)

    return zipname, df 


def main():

    import csv 

    #-----------------------------------------------------------------#
    #prepare csv for results 
    fieldnames_results = ['total_time', 'call_back_time','test_acc','test_loss'] 
    with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/results.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_results)
        writer.writeheader()

    #prepare csv for results 
    fieldnames_m = ['first_layer','second_layer','third_layer'] 
    with open('/home-mscluster/fmahlangu/2089676/atari_breakout_data/models.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_m)
        writer.writeheader()

    #-----------retrieving training data----------------------------#
    #TO DO: change to function 
    '''
    Extracting data for training 

    Meta_data.csv contains:
    trial_id used to locate the associated .tar.bz2 file and label file 
    total_frame: number of image frame in .bar.b2 repository 

    '''

    metaData_df= pd.read_csv('/home-mscluster/fmahlangu/2089676/atari_breakout_data/meta_data.csv')

    dataframe= metaData_df[metaData_df.GameName=="breakout"]

    #-------------------------------------------------------trial = 58 -------------------------------------------------------
    trial = 58
    zipname,df= get_data(trial) #trial =58
   

    #get images with associated actions for each trial 
    n=dataframe[dataframe.trial_id==trial].total_frame.tolist() #get number of frame/images
    n=n[0]

    df,deleted_rows=clean_df(df)
    images=create_train_x(zipname,n,deleted_rows)
    actions=create_train_y(df)

    trainX, valX, trainY, valY = train_test_split(images, actions, test_size=0.18)
    trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.12)

    #-----------#
    trainX = np.array(trainX)
    valX = np.array(valX)
    testX = np.array(testX)

    trainY = np.array(trainY)
    valY = np.array(valY)
    testY = np.array(testY)


    for i in range(len(trainY)):
        if trainY[i] == 3:
          trainY[i] = 2
        if trainY[i] == 4:
          trainY[i] = 3


    for i in range(len(testY)):
      if testY[i] == 3:
        testY[i] = 2
      if testY[i] == 4:
        testY[i] = 3
    
    for i in range(len(valY)):
      if valY[i] == 3:
        valY[i] = 2
      if valY[i] == 4:
        valY[i] = 3


    #---------converting y values to categorical data (one-hot encoding)---#
    # img_shape = [109, 83, 3]
    img_shape = [84, 84, 1]
    num_classes = 4

    trainY = keras.utils.to_categorical(trainY, num_classes)
    valY = keras.utils.to_categorical(valY, num_classes)
    testY = keras.utils.to_categorical(testY, num_classes)



    #-------------------------------RUN CONFIGURATIONS-----------------------------#

    params = []

    # params.append([[16, 1, 2, 'same', 'relu', 2, 0.05, 128, 0.05, 0.01],[32, 3, 1, 'same', 'relu', 3, 0.25],[32, 3, 1, 'same', 'relu', 3, 0.25]])
    # params.append([[16, 8, 2, 'same', 'relu', 2, 0.05, 512, 0.05, 0.01],[32, 3, 2, 'same', 'relu', 3, 0.25],[32, 3, 1, 'same', 'relu', 3, 0.25]])
    # params.append([[32, 5, 2, 'same', 'relu', 2, 0.05, 512, 0.05, 0.00001],[64, 3, 2, 'same', 'relu', 3, 0.25],[64, 3, 1, 'same', 'relu', 3, 0.25]])
    params.append([[16, 7, 4, 'same', 'relu', 1, 0.05, 512, 0.05, 0.001],[32, 4, 2, 'same', 'relu', 1, 0.25],[64, 3, 1, 'same', 'relu', 1, 0.25]]) #minh
    params.append([[32, 8, 4, 'same', 'relu', 1, 0.05, 256, 0.05, 0.1],[64, 4, 2, 'same', 'relu', 1, 0.25],[128, 3, 1, 'same', 'relu', 1, 0.25]])

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # print(keras.__version___)

    for p in params:
      first = p[0]
      second = p[1]
      third = p[2]

      run_model_config(trainX,trainY,valX,valY,testX,testY,img_shape,num_classes,first,second,third,fieldnames_results,fieldnames_m)
      




    #run models 

    # for fil in Filters:
    #     for ker in Kernels:
    #         for stri in Strides:
    #             for pad in Padding:
    #                 for act in Activations:
    #                     for pool in Pool:
    #                         for dropoutrate in Dropout_rate:
    #                             # for dense in Last_Dense:
    #                                 # for last_dropoutrate in Dropout_rate:
    #                                     # for learn_rate in Learning_rate:
    #                                         run_model_config(alltrainX,alltrainY,allvalX,allvalY,alltestX,alltestY,img_shape,num_classes,first_layer,[fil,ker,stri,pad,act,pool,dropoutrate],fieldnames_results,fieldnames_m)




if __name__ == "__main__":
    main()




