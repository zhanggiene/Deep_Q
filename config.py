import os
class Config():
    '''
    Config class
    '''
    def __init__(self):

        ### MODEL HYPERPARAMETERS
        self.path=" some path"
        self.state_size = [84, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels) 
        self.action_size = 0
        self.learning_rate =  0.00025      # Alpha (aka learning rate)

        ### TRAINING HYPERPARAMETERS
        self.total_episodes = 10000            # Total episodes for training
        self.max_steps = 500              # Max possible steps in an episode
        self.batch_size = 64                # Batch size

        # Exploration parameters for epsilon greedy strategy
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 
        self.decay_rate = 0.0001           # exponential decay rate for exploration prob

        # Q learning hyperparameters
        self.gamma = 0.9                    # Discounting rate

        ### MEMORY HYPERPARAMETERS
        self.pretrain_length = self.batch_size   # Number of experiences stored in the Memory when initialized for the first time
        self.memory_size = 1000000          # Number of experiences the Memory can keep

        ### PREPROCESSING HYPERPARAMETERS
        self.stack_size = 4                # Number of frames stacked

        ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
        self.training = False

        ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
        self.episode_render = False


        self.checkpoints  = self.path+'/checkpoints'     # checkpoints dir
        self.__mkdir(self.checkpoints)

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ',path)
    def setNumberOfActions(self,n):
        self.action_size=n
        print("config action set")

    



