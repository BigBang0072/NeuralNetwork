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
    def __init__(self,input_dim,layer_config,h_activation,\
                o_activation,loss_type,\
                param_init_type,param_dtype,\
                lr,epochs,\
                dataset_path,split_ratio,batch_size=1000):
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
        print("Initializing the variable")
        in_dimension=self.input_dim
        for layer in range(self.nlayers):
            #Setting the shape of the variable
            w_shape=[self.layer_config[layer],in_dimension]
            b_shape=[self.layer_config[layer]]
            print("Var-Init: w-shape:{} \tb-shape:{}".format(w_shape,\
                                                            b_shape))
            #Now initializing the variable
            if(self.param_init_type=="glorot"):
                W=np.random.randn(self.layer_config[layer],in_dimension)\
                            /(np.sqrt(in_dimension))\
                            .astype(self.param_dtype)
                #We could keep the bias term as zeros
                b=np.random.randn(self.layer_config[layer])\
                            /(np.sqrt(in_dimension))\
                            .astype(self.param_dtype)
            elif(self.param_init_type=="zeros"):
                W=np.zeros(shape=w_shape,dtype=self.param_dtype)
                b=np.zeros(shape=b_shape,dtype=self.param_dtype)

            #Saving the parameters in dictionary
            layer_params["W"+str(layer)]=W
            layer_params["b"+str(layer)]=b

            #Now updating the in-dimension
            in_dimension=self.layer_config[layer]

        #Now assigning these parameters to the class
        self.layer_params=layer_params
        print("Parameter Initialization Done!\n")

    #Function to run one-step of feed-forward through the network
    def feedforward_though_net(self,batch_input):
        '''
        This function will run though the network once taking in the
        input and giving out the output of the Network.
        '''
        #Initializing the cache dict
        actv_cache={}

        #Caching the input to the network
        #print("\nFeed-Forwarding through network")
        actv_cache["A0"]=batch_input
        #Initializing the input the the neural network first layer
        A = batch_input
        for layer in range(self.nlayers):
            #Extracting out the corresponding parameters
            W = self.layer_params["W"+str(layer)]
            b = self.layer_params["b"+str(layer)]

            #Now calculating the activation
            Z=(np.matmul(W,A).T+b).T
            #Passing the activation through the rectifier
            if(layer == self.nlayers-1):
                assert Z.shape[0]==1,"Binary Classification Bro!!"
                #We have reached the final layer,then apply sigmoid
                A=self.apply_activation(Z,self.o_activation)
            else:
                A=self.apply_activation(Z,self.h_activation)

            #Now caching the results for later use
            actv_cache["A"+str(layer+1)]=A
            #print("Shape of layer:{} is:{}".format(layer+1,A.shape))

        #Now assining the cache to the object
        self.actv_cache=actv_cache
        #print("Feed-Forward Completed\n")

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
                            self.backprop_activation(A_layer,"relu")

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
        elif(activation_name=="relu"):
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
        print("# Training Set minbatch:",len(train_exshard))
        print("# Validation Set minibatch:",len(valid_exshard))
        #Initializing the parameter
        self.initialize_parameters()

        #Starting the training loop
        for epoch in range(self.epochs):
            #Now iterating over all the batches of example
            for batch,ex_shard in enumerate(train_exshard):
                #Loading the mini-batch into the memory
                images,labels=load_batch_into_memory(self.dataset_path,
                                                    ex_shard)
                #Now forward propagating through the network
                self.feedforward_though_net(images)
                #Now calculating the gradient
                self.backpropagate_through_net(labels)
                #Testing the gradient numerically first
                self.__gradient_check(images,labels)
                #Now we will apply the parameter update
                self.update_parameter()

                #Calculating the loss in current epoch
                loss,accuracy=self.calculate_loss_and_accuracy(labels)
                print("Epoch:{} mini-batch:{} loss:{}\t accuracy:{}".format(
                                    epoch,batch,loss,accuracy))

            #Finally one epoch is completed
            print("Training Epoch-{} completed".format(epoch))

            #Starting the validation run
            if(epoch%valid_freq==0):
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
    def calculate_loss_and_accuracy(self,batch_labels,epsilon=1e-20):
        '''
        This function will calculate the loss in the current epoch
        give the labels and the output activation.
        '''
        #Retreiving the outptut of network from cache
        net_output=self.actv_cache["A"+str(self.nlayers)]

        #Calculating the loss
        loss=(batch_labels*np.log(net_output+epsilon))+\
                (1-batch_labels)*np.log(1-net_output+epsilon)
        loss=-1*np.mean(loss)

        #Calculating the accuracy of model
        accuracy=np.mean((net_output>0.5)==batch_labels)

        return loss,accuracy

    #Function to apply gradient checking
    def __gradient_check(self,batch_input,batch_labels,epsilon=1e-5):
        '''
        This function will check the gradient calculated by the network
        and calculating the gradient numerically by finite difference
        method. This is only for testing purpose and will be switched
        off during testing.
        '''
        print("Initiating gradient checking!!")
        #getting the current cost to test the correct theta is maintained
        J_actual,_=self.calculate_loss_and_accuracy(batch_labels)

        #Saving a copy of current gradient
        grad_actual,grad_cache_keys=self.__flatten_gradient_to_vec()
        #Initializing the approximate grad list
        grad_approx=[]

        #Calculating the gradient numberically
        for gname in grad_cache_keys:
            pname=gname[1:]
            print(pname)
            #Extracting out the parameter attributes
            param_shape=self.layer_params[pname].shape
            param=self.layer_params[pname]

            #Hacking to make the loop go through parameter b
            if(pname[0]=="b"):
                for i in range(param_shape[0]):
                    #Nudgeing the parameter in right direction
                    param[i]=param[i]+epsilon
                    J_plus=self.__calculate_perturbed_loss(batch_input,
                                                        batch_labels)

                    #Now nudging the parameter in left direction
                    param[i]=param[i]-2*epsilon
                    J_minus=self.__calculate_perturbed_loss(batch_input,
                                                        batch_labels)

                    #Correcting the parameter value
                    param[i]=param[i]+epsilon
                    # J_now=self.__calculate_perturbed_loss(batch_input,
                    #                                     batch_labels)
                    #
                    # print(J_actual,J_now,J_plus,J_minus)

                    #Now calculating the gradient using the loss val
                    grad=(J_plus-J_minus)/(2*epsilon)
                    #Appending the gradient to big grad array
                    grad_approx.append(grad)

                #Now go to the new parameter variable
                continue

            #Nudgeing each of the parameter one by one
            for i in range(param_shape[0]):
                for j in range(param_shape[1]):
                    #Nudgeing the parameter in right direction
                    # print(param[i,j],epsilon)
                    param[i,j]=param[i,j]+epsilon
                    # print(param[i,j])
                    J_plus=self.__calculate_perturbed_loss(batch_input,
                                                        batch_labels)

                    #Now nudging the parameter in left direction
                    param[i,j]=param[i,j]-2*epsilon
                    # print(param[i,j])
                    J_minus=self.__calculate_perturbed_loss(batch_input,
                                                        batch_labels)

                    #Correcting the parameter value
                    param[i,j]=param[i,j]+epsilon
                    # print(param[i,j])
                    # J_now=self.__calculate_perturbed_loss(batch_input,
                    #                                     batch_labels)
                    #
                    # print(J_actual,J_now,J_plus,J_minus)

                    #Now calculating the gradient using the loss val
                    grad=(J_plus-J_minus)/(2*epsilon)
                    #Appending the gradient to big grad array
                    grad_approx.append(grad)

        #Finally its time to see the relative diff of grad with actual
        grad_approx=np.array(grad_approx)
        #Printing out few gradients
        print(grad_actual[-1],grad_approx[-1])

        #Calculating the relative difference
        rel_diff=np.linalg.norm(grad_actual-grad_approx)\
            /(np.linalg.norm(grad_actual)+np.linalg.norm(grad_approx))

        print("Relative difference in gradient: ",rel_diff)


    def __flatten_gradient_to_vec(self):
        '''
        This function will flatten the gradient into one long vector
        so that we could compare the difference in gradient.
        '''
        grad_vec=[]
        grad_cache_keys=self.grad_cache.keys()
        for gname in grad_cache_keys:
            print(gname)
            grad=self.grad_cache[gname]
            grad_copy=np.copy(grad).reshape(-1)
            grad_vec.append(grad_copy)

        #Now concatenating all the gradient into one vector
        grad_copy=np.concatenate(grad_vec)
        print("Gradient Copied, shape:{} dtype:{}".format(grad_copy.shape,
                                                        grad_copy.dtype))

        return grad_copy,grad_cache_keys

    def __calculate_perturbed_loss(self,batch_input,batch_labels):
        '''
        This function will calculate the cost give current parameter
        by first feedforwarding through the net.
        '''
        #Forward propagating the layer with new parameter
        self.feedforward_though_net(batch_input)
        #Now calculating the new cost
        J_pertb,_=self.calculate_loss_and_accuracy(batch_labels)
        return J_pertb

if __name__=="__main__":
    #Testing the implementation
    myNet=FF_NeuralNetwork(input_dim=2500,
                            layer_config=[10,10,1],
                            h_activation="relu",
                            o_activation="sigmoid",
                            loss_type="cross_entropy",
                            param_init_type="glorot",
                            param_dtype=np.float32,
                            lr=0.001,
                            epochs=5,
                            dataset_path="dataset/train_valid/",
                            split_ratio=0.85,
                            batch_size=50)

    #Starting the training procedure
    myNet.train_the_network(valid_freq=1)