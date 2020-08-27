

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs

from collections import deque# Ordered collection with ends

import random
from agent import Agent
from game import Game




from config import Config

config=Config()
boy=Agent()
challengGame=Game()



for episode in range(config.total_episodes):
    # Set step to 0
    step = 0
    
    # Initialize the rewards of the episode
    total_rewards =0  #episode reward should be managed by training
    
    # Make a new episode and observe the first state
    challengGame.populateEmptyMemory(boy.memory)
    
    # Remember that stack frame function also call our preprocess function.
    #state, stacked_frames = stack_frames(stacked_frames, state, True)
    state=challengGame.reset()
    #print(state.shape)
    while step < config.max_steps:
        step += 1
        action=boy.act(state)
        next_state,reward,done=challengGame.nextState(action)
        
        if config.episode_render:
            challengGame.render()
        
        # Add the reward to total reward
        print
        total_rewards+=reward
        if done:
            #print("games ending now")
            next_state=challengGame.end()
            #print(next_state.shape)
            step=config.max_steps # to end the while loop
            #print("episode"+episode+"total reward is "+total_rewards)

            boy.remember((state,action,reward,next_state,done))
        else:
            #print("the remember shape is")
            boy.remember((state,action,reward,next_state,done))
            state=next_state
        boy.replay()
    print("episode number is ",episode)
    print("total reward is",total_rewards)
    if episode%2==0:
        print("saving model")
        boy.saveModel(str(episode))



