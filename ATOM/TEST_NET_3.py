import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np

with open('./OUT_PUT/all_db_min_max_train_DB.bin', 'rb') as file:    # 약 7sec 걸림
    data = pickle.load(file)

model = load_model('model.h5')

temp_out = []
out = model.predict([data[0][202368:]])
for el in out:
    temp_out.append(np.argmax(el))

temp_real = []
real = data[1][202368:]
for el in real:
    temp_real.append(np.argmax(el))

cm = confusion_matrix(temp_real, temp_out)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Accident Diagnosis Confusion Matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=['MSLB 499', 'MSLB 699', 'SBLOCA'], yticklabels=['MSLB 499', 'MSLB 699', 'SBLOCA'],
           title='Accident Diagnosis Confusion Matrix',
           ylabel='Test Data label',
           xlabel='LSTM Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

plot_confusion_matrix(temp_real, temp_out, classes='aa', title='Accident Diagnosis Confusion Matrix')
plt.show()