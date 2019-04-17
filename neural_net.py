import numpy as np
from data_handling import get_batched_dataset,load_batch_into_memory


class FF_NeuralNetwork():
    '''
    This is the main class handling all the work from variable init,
    to feed-forwarding the network and then tuning the paramenters by
    back-propagation for the simple feed-forward neural network.
    '''
    ########################## ATTRIBUTES ###########################
    layer_params=None       #dictionary holding the params of each layer
    actv_cache=None         #Caching the results of each layer for BProp
    grad_cache=None         #Caching the gradient for parameter update

    ######################## MEMEBER FUNCTION #######################
    #Initializer fucntion
    def __init__(input_dim,layer_config,h_activation,\
                o_activation,loss_type,\
                param_init_type,param_dtype,\
                lr,epochs,\
                dataset_path,split_ratio,batch_size):
        '''
        This function will initialize the Network with the appropriate
        hyperparameters and model
        '''
        #Paramater related attributes
        self.param_dtype=param_dtype
        self.param_init_type=param_init_type

        #Setting the model related variables
        self.input_dim=input_dim
        self.nlayers=len(layer_config)
        self.layer_config=layer_config
        self.h_activation=h_activation
        self.o_activation=o_activation
        self.loss_type=loss_type

        #Optimization related hyperparameter
        self.lr=lr
        self.epochs=epochs

        #Dataset related parameters
        self.dataset_path=dataset_path
        self.split_ratio=split_ratio
        self.batch_size=batch_size

    #Function to initialize the paramters
    def initialize_parameters(self):
        '''
        This function will initialize the paramters based on the network
        configuration.
        '''
        #Intializing the parameter variable
        layer_params={}

        #Now creating varialbe for each layer
        in_dimension=self.input_dim
        for layer in range(self.nlayers):
            #Setting the shape of the variable
            w_shape=[layer_config[layer],in_dimension]
            b_shape=[layer_config[layer]]

            #Now initializing the variable
            if(self.init_type=="glorot"):
                W=np.random.normal(shape=w_shape,dtype=self.param_dtype)\
                            /(np.sqrt(in_dimension))
                #We could keep the bias term as zeros
                b=np.random.normal(shape=b_shape,dtype=self.param_dtype)\
                            /(np.sqrt(in_dimension))
            elif(self.init_type=="zeros"):
                W=np.zeros(shape=w_shape,dtype=self.param_dtype)
                b=np.zeros(shape=b_shape,dtype=self.param_dtype)

            #Saving the parameters in dictionary
            layer_params["W"+str(layer)]=W
            layer_params["b"+str(layer)]=b

            #Now updating the in-dimension
            in_dimension=layer_config[layer]

        #Now assigning these parameters to the class
        self.layer_params=layer_params

    #Function to run one-step of feed-forward through the network
    def feedforward_though_net(self,batch_input):
        '''
        This function will run though the network once taking in the
        input and giving out the output of the Network.
        '''
        #Initializing the cache dict
        actv_cache={}

        #Caching the input to the network
        actv_cache["A0"]=batch_input
        #Initializing the input the the neural network first layer
        A = batch_input
        for layer in range(self.nlayers):
            #Extracting out the corresponding parameters
            W = self.layer_params["W"+str(layer)]
            b = self.layer_params["b"+str(layer)]

            #Now calculating the activation
            Z=np.matmul(W,A)+b
            #Passing the activation through the rectifier
            if(layer == self.nlayers-1):
                assert Z.shape[0]==1,"Binary Classification Bro!!"
                #We have reached the final layer,then apply sigmoid
                A=self.apply_activation(Z,self.o_activation)
            else:
                A=self.apply_activation(Z,self.h_activation)

            #Now caching the results for later use
            actv_cache["A"+str(layer+1)]=A

        #Now assining the cache to the object
        self.actv_cache=actv_cache

    #Function to apply the activation to the input tensor
    def apply_activation(self,Z,activation_name):
        '''
        This function will apply the appropriate activation to the input
        tensor.
        '''
        if(activation_name=="sigmoid"):
            A=1/(1+np.exp(-1*Z))
            return A
        elif(activation_name=="relu"):
            A=np.maximum(Z*0.0,Z)
            return A
        else:
            raise Exception("activation not defined")

    #Function to backpropagate through the network
    def backpropagate_through_net(self,batch_labels):
        '''
        This function will get start from the output layer and then
        propagate the error backward calculating the gradient in the
        way and caching them in a dictionary for gradient update.
        '''
        #Initializing the gradient cache
        grad_cache={}

        #Initializing the stage 2 gradient for the output layer
        dA=(batch_labels-self.actv_cache["A"+str(self.nlayers)])\
                    / (-1*self.batch_size)

        #Now iterating over the subsequent layer to get gradient
        for layer in range(self.nlayers-1,-1,-1):
            #Stage 1: Calculating the gradient in parameter
            A_layer = self.actv_cache["A"+str(layer)]
            dW = np.matmul(dA,A_layer.T)
            db = np.sum(dA,axis=1)
            #Saving the gradient in the cache
            grad_cache["dW"+str(layer)]=dW
            grad_cache["db"+str(layer)]=db

            #Stage 2: Backpropgating the gradient through this layer
            W = self.layer_params["W"+str(layer)]
            dA = np.matmul(W.T,dA) * \
                            self.backprop_activation(A_layer)

        #Now assigning this gradient cache to this object
        self.grad_cache=grad_cache

    #Function to backpropagate the gradient through the activation fn
    def backprop_activation(self,A,activation_name):
        '''
        This function will backpropagate the error though the activation
        performed during the feedforward operation.
        '''
        if(activation_name=="sigmoid"):
            return A*(1-A)
        else if(activation_name=="relu"):
            return (A>0)*1.0
        else:
            raise AssertionError("unsupported activation type!!")

    #Trainer function to handle all the training and parameter update
    def train_the_network(self,valid_freq):
        '''
        This function will handle the main control for the full training
        of the network over the whole dataset in the stochastic manner.
        '''
        #Initializing the dataset
        train_exshard,valid_exshard=get_batched_dataset(
                                                    self.dataset_path,\
                                                    self.split_ratio,\
                                                    self.batch_size)
        #Starting the training loop
        for epoch in self.epochs:
            #Now iterating over all the batches of example
            for batch,ex_shard in enumerate(train_exshard):
                #Loading the mini-batch into the memory
                images,labels=load_batch_into_memory(self.dataset_path,
                                                    ex_shard)
                #Now forward propagating through the network
                self.feedforward_though_net(images)
                #Now calculating the gradient
                self.backpropagate_through_net(labels)
                #Now we will apply the parameter update
                self.update_parameter()

                #Calculating the loss in current epoch
                loss,accuracy=self.calculate_loss_and_accuracy(labels)
                print("Epoch:{} mini-batch:{} loss:{}\t accuracy:{}".format(
                                    epoch,batch,loss,accuracy))

            #Finally one epoch is completed
            print("Training Epoch-{} completed".format(epoch))

            #Starting the validation run
            if(epoch%valid_freq):
                for batch,ex_shard in enumerate(valid_exshard):
                    #Loading the mini-batch into the memory
                    images,labels=load_batch_into_memory(self.dataset_path,
                                                        ex_shard)
                    #Now forward propagating through the network
                    self.feedforward_though_net(images)

                    #Calculating the loss in current epoch
                    loss,accuracy=self.calculate_loss_and_accuracy(labels)
                    print("Epoch:{} mini-batch:{} loss:{}\t accuracy:{}".format(
                                        epoch,batch,loss,accuracy))
            print("Validation Completed")

    #Function to update the parameters finally once the grad is found
    def update_parameter(self):
        '''
        This fucntion will update the paramter using the already
        clculated gradient using the vanilla gradient descent.
        '''
        for param in self.layer_params.keys():
            #Retreiving the corresponding gradient
            grad=self.grad_cache["d"+param]

            #Updating the parameter using vanilla gradient descent
            self.layer_params[param]= self.layer_params[param]\
                                        - self.lr*grad

    #Function to calculate the loss in the current epoch
    def calculate_loss_and_accuracy(self,batch_labels):
        '''
        This function will calculate the loss in the current epoch
        give the labels and the output activation.
        '''
        #Retreiving the outptut of network from cache
        net_output=self.actv_cache["A"+self.nlayers]

        #Calculating the loss
        loss=(batch_labels*np.log(net_output))+\
                (1-batch_labels)*np.log(1-net_output)
        loss=-1*np.mean(loss)

        #Calculating the accuracy of model
        accuracy=np.mean((net_output>0.5)==batch_labels)

        return loss,accuracy
