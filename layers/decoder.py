from keras.layers import Dense
from keras.models import Sequential

class Decoder:
    def __init__(self, input_shape):
        '''
        The decoder layer uses the state at time t (s_t) and the summary of the image at time t (g_t).
        It produces the output y_t.
        '''
        self.model = Sequential(name='decoder')
        self.model.add(Dense(256, 'tanh', input_shape=input_shape))