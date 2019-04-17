import numpy as np
import imageio
import scipy.misc
import os
import matplotlib.pyplot as plt
import sys
np.random.seed(14)

def get_batched_dataset(dataset_path,split_ratio,batch_size):
    '''
    Instead of loading the images into memory and then sharding them into
    batches or test or train split we will read the names of file and
    shard them and then load the images batch by batch.
    '''
    #Reading the filenames from the dataset direcotory
    exnames=np.array(os.listdir(dataset_path))
    print("Size of dataset:",exnames.shape,"\n")

    #Shuffling the dataset into train and test split
    np.random.shuffle(exnames)

    #Now splitting the dataset into train and test split
    split_pos=int(split_ratio*exnames.shape[0])
    train_exnames=exnames[0:split_pos]
    valid_exnames=exnames[split_pos:]
    print("Examples in training set:",train_exnames.shape[0])
    print("Examples in validation set:",valid_exnames.shape[0],"\n")

    #Now sharding the dataset into batches
    train_exshard=shard_exnames(train_exnames,batch_size)
    valid_exshard=shard_exnames(valid_exnames,batch_size)
    print("Shards in training dataset:",len(train_exshard))
    print("Shards in validation dataset:",len(valid_exshard),"\n")

    return train_exshard,valid_exshard

def shard_exnames(exnames,batch_size):
    '''
    This function will shard the full dataset into almost equally sized
    batches which will be suitable for the training on the low RAM
    machines.
    '''
    #Calcualting the number of shards
    num_shards=int(exnames.shape[0]/batch_size)

    #Now creating the batches
    batch_exnames=[]
    for snum in range(num_shards):
        #Slicing the examples array
        shard=exnames[snum*batch_size:(snum+1)*batch_size]
        #Appeniding to the batch array
        batch_exnames.append(shard)

    #Now filling the last left out shard
    shard_left=exnames[num_shards*batch_size:]
    if(shard_left.shape[0]!=0):
        batch_exnames.append(shard_left)

    return batch_exnames

def load_batch_into_memory(dataset_path,batch_exnames,imsize=(50,50)):
    '''
    Given a numpy array containg the list of the names of the images,
    load the batch into memory and also generate the labels for them.
    '''
    #Initializingt the variable to hold the examples in batch
    labels=[]
    images=[]

    #Iterating over the examples
    print("Loading the batch to the memory")
    for exnum in range(batch_exnames.shape[0]):
        #Getting the label
        exname=batch_exnames[exnum]
        #Labelling dog:0 (positive doggo) and cat:1 (negetive bille)
        labels.append(int(exname.split(".")[0]=="cat"))

        #Getting the images
        img=imageio.imread(dataset_path+exname,pilmode='L')
        #Resizing the image
        img=scipy.misc.imresize(img,imsize,interp='lanczos')
        #Flattening the image into 1D vector
        img=img.reshape(-1)
        images.append(img)

        print(exname,"\tshape:{} \tdtype:{}".format(img.shape,img.dtype))
        # plt.imshow(img,cmap="gray")
        # plt.show()
        # print()

    #Now converting the dataset into numpy array
    labels=np.array(labels,dtype=np.uint8)
    images=np.array(images)
    print("Batch Created: shape:{} img_dtype:{}".format(images.shape,\
                                                        images.dtype))

    return images,labels



if __name__=="__main__":
    #Reading,shuffling,splitting and sharding the dataset
    dataset_path="dataset/train_valid/"
    train_exshard,valid_exshard=get_batched_dataset(dataset_path,\
                                                    split_ratio=0.85,\
                                                    batch_size=1000)
    #Now testing the conversion
    load_batch_into_memory(dataset_path,train_exshard[0])