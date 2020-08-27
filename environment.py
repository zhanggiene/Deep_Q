

#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Space%20Invaders/DQN%20Atari%20Space%20Invaders.ipynb

import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro                 # Retro Environment

print(tf.__version__)

from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames

import matplotlib.pyplot as plt # Display graphs

from collections import deque# Ordered collection with ends

import random

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')


# Create our environment
env = retro.make(game='SpaceInvaders-Atari2600')

print("The size of our frame is: ", env.observation_space)
print("The action size is : ", env.action_space.n)

# Here we create an hot encoded version of our actions
# possible_actions = [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]...]
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
#https://github.com/openai/retro/issues/53 
def preprocess_frame(frame):
    # Greyscale frame 
    gray = rgb2gray(frame)
    
    # Crop the screen (remove the part below the player)
    # [Up: Down, Left: right]
    cropped_frame = gray[8:-12,4:-12]
    
    # Normalize Pixel Values
    normalized_frame = cropped_frame/255.0
    
    # Resize
    # Thanks to Mikołaj Walkowiak
    preprocessed_frame = transform.resize(normalized_frame, [110,84])
    
    return preprocessed_frame # 110x84x1 frame




def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)
    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        # deque is used for pop operation. log(n) of 1.
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames



### MODEL HYPERPARAMETERS
state_size = [110, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels) 
action_size = env.action_space.n # 8 possible actions
learning_rate =  0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50            # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 64                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False




class MModel():
    def __init__(self,numActions,img_rows,img_cols,img_channels):

        input1=Input(shape=(img_cols,img_rows,img_channels))
        c1=Conv2D(32, (8, 8), strides=(4, 4), padding='same',activation='relu')(input1)
        c2=Conv2D(64, (4, 4), strides=(2, 2), padding='same',activation='relu')(c1)
        c3=Conv2D(64, (3, 3), strides=(1, 1), padding='same',activation='relu')(c2)
        f1 = Flatten()(c3)



        d1=Dense(512,activation="relu")(f1)
        state_value = Dense(1)(d1)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(numActions,))(state_value)

        d2=Dense(512,activation="relu")(f1)
        action_advantage = Dense(numActions)(d2)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(numActions,))(action_advantage)
        q1=Add()([state_value,action_advantage])


        model = Model(inputs = input1, outputs = q1)
        adam = Adam(lr=1e-4)
        model.compile(loss="mse", optimizer=adam)
        print("We finish building the Model")
        self.model=model






    def predict(self,imageStack):
        #return action index of the action 
        # So input data has a shape of (batch_size, height, width, depth),1,84,84,4
        #after training , it should produce the q value for each action
        q=self.model.predict(imageStack)
        return q
    def train(self,x,y):
        loss=self.model.train_on_batch(x,y)
        return loss
    def save(self,name):
        self.model.save_weights(name,overwrite=True)
    def load(self,name):
        self.model.load_weights(name)
        print("weight loaded")
    def getModel(self):
        return self.model





class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
            #useful for debugging as it adds it adds/prepend name in front. 
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 84, 84, 4]
            # none means the number of row is unknown. 
            # 

        self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
        self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
        
        # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
        self.target_Q = tf.placeholder(tf.float32, [None], name="target")
        
        """
        First convnet:
        CNN
        ELU
        """
        # Input is 110x84x4
        self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                        filters = 32,
                                        kernel_size = [8,8],
                                        strides = [4,4],
                                        padding = "VALID",
                                        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                        name = "conv1")
        
        self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
        
        """
        Second convnet:
        CNN
        ELU
        """
        self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                filters = 64,
                                kernel_size = [4,4],
                                strides = [2,2],
                                padding = "VALID",
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                name = "conv2")

        self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")            # it is elu , not relu

        self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                filters = 64,
                                kernel_size = [3,3],
                                strides = [2,2],
                                padding = "VALID",
                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                name = "conv3")

        self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
        
        self.flatten = tf.contrib.layers.flatten(self.conv3_out)
        
        self.fc = tf.layers.dense(inputs = self.flatten,
                                units = 512,
                                activation = tf.nn.elu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name="fc1")
        
        self.output = tf.layers.dense(inputs = self.fc, 
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        units = self.action_size, 
                                    activation=None)
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
        # The loss is the difference between our predicted Q_values and the Q_target
        # Sum(Qtarget - Q)^2
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

DQNetwork = DQNetwork(state_size, action_size, learning_rate)


class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size) #once the new item is added, then item will be discarded from the other end. 
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        # np.arange(5)gives [1,2,3,4,5,6,7,8,9,10]
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        # it means out of the list, choose 4. 
        
        return [self.buffer[i] for i in index]




# Instantiate memory
memory = Memory(max_size = memory_size)
stacked_frames  =  deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
print(stack_frames.shape)
for i in range(pretrain_length):
    # If it's the first step
    if i == 0:
        state = env.reset()
        
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    # Get the next_state, the rewards, done by taking a random action
    choice = random.randint(1,len(possible_actions))-1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)
    
    #env.render()
    
    # Stack the frames
    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
    
    
    # If the episode is finished (we're dead 3x)
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)
        
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Start a new episode
        state = env.reset()
        
        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state, done))
        
        # Our new state is now the next_state
        state = next_state