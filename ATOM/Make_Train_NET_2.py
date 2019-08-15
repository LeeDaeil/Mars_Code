import pickle
import numpy as np

from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop

with open('./OUT_PUT/all_db_min_max_train_DB.bin', 'rb') as file:    # 약 7sec 걸림
    data = pickle.load(file)

time_leg = 10
in_pa = 99
out_pa = 3

state = Input(batch_shape=(None, time_leg, in_pa))
shared = LSTM(256, activation='relu')(state)
shared = Dense(256, activation='relu')(shared)
shared = Dense(512, activation='relu')(shared)
shared = Dense(out_pa, activation='softmax')(shared)

model = Model(inputs=state, outputs=shared)

model.summary()

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit([data[0][:201376]], [data[1][:201376]], epochs=10, batch_size=500)

model.save('./OUT_PUT/model.h5')