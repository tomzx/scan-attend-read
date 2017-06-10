from keras.layers import LSTM
from keras.models import Sequential

class State:
    def __init__(self, input_shape):
        '''
        The state layer uses the state at time t-1 (s_{t-1}) and the summary of the image at time t (g_t).
        It produces the state s_t.
        '''
        self.model = Sequential(name='state')
        self.model.add(LSTM(256, input_shape=input_shape))