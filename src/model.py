from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten

def get_model(env,nb_actions):
    # モデルの定義
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(24))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model