
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices


from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs



import random

from config import Config

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

from Memory import Memory
from collections import deque# Ordered collection with ends
from Model import MModel

class Agent:
    def __init__(self):
        self.config=Config()
        self.epsilon=self.config.explore_start
        print("start of epsilon is ",self.epsilon)
        self.brain=MModel(self.config.action_size,self.config.state_size[0],self.config.state_size[1],self.config.state_size[2])
        self.memory=Memory(self.config.memory_size)
        self.num_actions=self.config.action_size
        self.decayStep=0
        #self.stacked_frames is 4 for now


    def act(self,StackOfImage):
        #input is 88,84,4 
        self.exploreLess()
        #input is the stack of images (1,84,84,4)    So input data has a shape of (batch_size, height, width, depth), 
        # it will return numpy , then 
        #choose action to take based on episolon greedy, 


        #actionsToTake = np.zeros([self.num_actions]) # action at t a_t[0,0]
        if  random.random() <= self.epsilon: #randomly explore an action
            #print("----------Random Action----------")
            #print(self.epsilon)
            action_index = random.randrange(self.num_actions) # it will be 0,1
            #actionsToTake[action_index]=1
            #print(action_index)
        else:
            q=self.brain.predict(StackOfImage.reshape((1,*StackOfImage.shape)))
            #print(q)
            #self.display(q[0][0],q[0][1])
            action_index=np.argmax(q)
            #actionsToTake[action_index]=1
            #print(action_index)
        return action_index
    def exploreLess(self):
        self.decayStep+=1
        #this method decay too fast,not gradient is not slowingdown
        # self.epsilon = max(self.config.explore_stop, self.epsilon * np.exp(-self.config.decay_rate*self.decayStep))
        self.epsilon=self.config.explore_stop+(self.config.explore_start-self.config.explore_stop)*np.exp(-self.config.decay_rate*self.decayStep)
    def remember(self,experience):
        self.memory.add(experience)
        #print("inside remember function",experience[0].shape)


    def replay(self):
        #replay means learn, it will decrease episolon also. 




        targets = np.zeros((self.config.batch_size, self.config.action_size))
        batch=self.memory.sample(self.config.batch_size)
        #print("batch size is ",batch[0][0].shape)
        states_mb = np.array([each[0] for each in batch],ndmin=3)
        # ndmin meaning is stack to the last dimention.  from 64 4 84 84    to 64 84 84 4
        actions_mb = np.array([each[1] for each in batch])
        rewards_mb = np.array([each[2] for each in batch]) 
        next_states_mb = np.array([each[3] for each in batch],ndmin=3)
        dones_mb = np.array([each[4] for each in batch])
        Qs_targets= self.getQvalue(states_mb)
        #print(next_states_mb.shape)
        Qs_nextState=self.getQvalue(next_states_mb)  # 64*4 
        
        

        for i in range(0,len(batch)):
            terminal=dones_mb[i]
            if terminal:
                actionNumber=np.argmax(Qs_targets[i])
                Qs_targets[i][actionNumber]=rewards_mb[i]
            else:
                actionNumber=np.argmax(Qs_targets[i])
                Qs_targets[i][actionNumber]=rewards_mb[i]+self.config.gamma*np.max(Qs_nextState[i])    # the bootstraping way to get Q value. 
        #targets_np=np.array(target_Qs_batch)# change to numpy array
        loss=self.brain.train(states_mb,Qs_targets)



    def getQvalue(self,stackOfImage):
        return self.brain.predict(stackOfImage)
    def saveModel(self,name):
        self.brain.save(self.config.checkpoints+'/'+name+'.ckpt')
    def getEpsilon(self):
        return self.epsilon
    