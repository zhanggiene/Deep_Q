

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs

from collections import deque# Ordered collection with ends

import random
from agent import Agent
from game import Game
import matplotlib.pyplot as pyplot




from config import Config

config=Config()
boy=Agent()
challengGame=Game(config)

score=[]
eps_history=[]


#game need config for initialization Game(Config)
#game sets config during initialization

def plot_learning_curve(x,score,epsilon,path):
    fig=plt.figure()
    ax=fig.add_subplot(111,label="1")
    ax2=fig.add_subplot(111,label='2',frame_on=False)
    ax.plot(x,epsilon,color='C0')
    ax.set_xlabel("training Steps",color='C0')
    ax.set_ylabel("Epsilon",color='C0')
    ax.tick_params(axis='x',colors='C0')
    ax.tick_params(axis='y',colors='C0')

    ax2.scatter(x,score,color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('score',color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y',color='C1')
    plt.savefig(path+"/progress.png")
#plot_learning_curve([1,2,3],[5,6,7],[0.1,0.2,0.3],".")

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
    score.append(total_rewards)
    eps_history.append(boy.getEpsilon)
    if episode%1000==0:
        print("saving model")
        boy.saveModel(str(episode))
x=[i+1 for i in range(config.total_episodes)]

