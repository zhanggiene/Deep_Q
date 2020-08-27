import numpy as np 
import random
from collections import deque# Ordered collection with ends

stacked_states_Q  =  deque([np.zeros((84,84), dtype=np.int) for i in range(4)], maxlen=4)
print(stacked_states_Q[0].shape)