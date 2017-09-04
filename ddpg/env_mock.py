import numpy as np
import random

class MockEnvironment():
    def __init__(self, s_shape):
        self.s_shape = s_shape

    def reset(self):
        return np.random.rand(*self.s_shape)
    
    def step(self, a):
        return np.random.rand(*self.s_shape), 0.0, False, None