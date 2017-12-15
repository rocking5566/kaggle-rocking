import bcolz
import matplotlib.pyplot as plt
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix


def plot(img):
    plt.imshow(to_plot(img))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_array(file_name, arr):
    c = bcolz.carray(arr, rootdir=file_name, mode='w')
    c.flush()


def load_array(file_name):
    return bcolz.open(file_name)[:]


def get_batches_from_dir(path, gen=image.ImageDataGenerator(), shuffle=True,
                         batch_size=4, class_mode='categorical', target_size=(224, 224)):
    return gen.flow_from_directory(path, target_size=target_size,
                                   class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
