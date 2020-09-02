
import retro                 # Retro Environment


from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames


import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices


from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs

import gym

import random

from config import Config
from collections import deque# Ordered collection with ends





class Game:
    #game maintain the stacked 4 states
    def __init__(self,config):
        self.config=config
        #self.env = retro.make(game='SpaceInvaders-Atari2600')
        self.env=gym.make('Breakout-v0')
        #print("possible action")
        self.config.setNumberOfActions(self.env.action_space.n)
        self.stacked_states_Q  =  deque([np.zeros((self.config.state_size[0:2]), dtype=np.int) for i in range(self.config.stack_size)], maxlen=4)
        #self.stacked_frames is 4 for now
        self.possible_actions = np.array(np.identity(self.env.action_space.n,dtype=int).tolist())
        #print("initialize game ")


    def nextState(self,action):
        next_frame, reward, done, _ = self.env.step(action)
        #print(reward)
        self.stacked_states_Q.append(self.preprocess_frame(next_frame))
        stacked_state_next=np.stack(self.stacked_states_Q,axis=2)

        return stacked_state_next,reward,done 




    def reset(self):
        #return 
        frame = self.env.reset() # it return the first screen shot  [210,160 3]
        state=self.preprocess_frame(frame)   
        #print(state)
        self.stacked_states_Q.append(state)
        self.stacked_states_Q.append(state)
        self.stacked_states_Q.append(state)
        self.stacked_states_Q.append(state)
        stacked_state=np.stack(self.stacked_states_Q, axis=2)


        return stacked_state
    def end(self):
        next_state = np.zeros(self.config.state_size[0:2], dtype=np.int)
        self.stacked_states_Q.append(next_state)
        stacked_state_next=np.stack(self.stacked_states_Q,axis=2)

        return stacked_state_next





    def getActionNumber(self):
        return self.env.action_space.n
    def populateEmptyMemory(self,memory):
        # it is only the batch size. not the whole memory 
        # for memory replay to work, it should contains at least 64. 
        #it is just taking random action.  
        for i in range(self.config.pretrain_length):
            if i == 0:
                frame = self.env.reset() # it return the first screen shot  [210,160 3]
                state=self.preprocess_frame(frame)
                self.stacked_states_Q.append(state)
                self.stacked_states_Q.append(state)
                self.stacked_states_Q.append(state)
                self.stacked_states_Q.append(state)
                stacked_state=np.stack(self.stacked_states_Q, axis=2)
                #print("empty memory satte",stacked_state.shape)

            choice = random.randint(1,self.env.action_space.n)-1 # random number 
            #action = self.possible_actions[choice]     #choose acrtion vector 
            next_frame, reward, done, _ = self.env.step(choice)
        
            self.stacked_states_Q.append(self.preprocess_frame(next_frame))
            stacked_state_next=np.stack(self.stacked_states_Q,axis=2)

        
            if done:
            # We finished the episode
                stacked_state_next = np.zeros(stacked_state.shape)
                
                # Add experience to memory
                memory.add((stacked_state, choice, reward, stacked_state_next, done))

                frame = self.env.reset() # it return the first screen shot  [210,160 3]
                state=self.preprocess_frame(frame)
                self.stacked_states_Q.append(state)
                self.stacked_states_Q.append(state)
                self.stacked_states_Q.append(state)
                self.stacked_states_Q.append(state)
                stacked_state=np.stack(self.stacked_states_Q, axis=2)

                
                
            else:
            # Add experience to memory
                #print("else",stacked_state.shape)
                memory.add((stacked_state, choice, reward, stacked_state_next, done))
                
                # Our new state is now the next_state
                stacked_state = stacked_state_next








    def preprocess_frame(self,frame):
        gray = rgb2gray(frame)
    
    
    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
        #print(gray)
        #cropped_frame = gray[8:-12,4:-12] # numpy select row and column
    
    # Normalize Pixel Values
        normalized_frame = gray/255.0
    
    # Resize
    # Thanks to Mikołaj Walkowiak
        preprocessed_frame = transform.resize(normalized_frame, self.config.state_size[0:2])
    
        return preprocessed_frame # 84x84x1 frame
    
    def render(self):
        self.env.render()
