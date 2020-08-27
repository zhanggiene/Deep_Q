from collections import deque# Ordered collection with ends
import numpy as np
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size) #once the new item is added, then item will be discarded from the other end. 
    
    def add(self, experience):
        #stacked_state, action, reward, stacked_state_next, done
        self.buffer.append(experience)
        #print("experience state is",experience[2])
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        # np.arange(5)gives [1,2,3,4,5,6,7,8,9,10]
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        # it means out of the list, choose 4. 
        
        return [self.buffer[i] for i in index]
