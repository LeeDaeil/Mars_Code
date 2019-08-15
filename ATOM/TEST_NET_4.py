import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np

with open('./OUT_PUT/all_db_min_max_train_DB.bin', 'rb') as file:    # 약 7sec 걸림
    data = pickle.load(file)

model = load_model('model.h5')

out = model.predict([data[0][210384:211376]])
real = data[1][210384:211376]

plt.plot(out)
plt.plot(real)
plt.show()
