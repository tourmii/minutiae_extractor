import os
from datetime import datetime
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

from . import verify_net_model, utils

PRECISION = 30
DB = 'all'

CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

TRAIN_DATASET_PATH = f'/home/jakub/projects/dp/matcher_training_data/server_dataset/{DB}/{PRECISION}/train'
TEST_DATASET_PATH = f'/home/jakub/projects/dp/matcher_training_data/server_dataset/{DB}/{PRECISION}/test'
LOGS_FOLDER = f'/home/jakub/projects/dp/fingerflow/logs/new/logs-final/scalars/{PRECISION}_{CURRENT_TIMESTAMP}'
MODEL_PATH = f'/home/jakub/projects/dp/fingerflow/models/final/matcher_contrast_weights_{PRECISION}_{CURRENT_TIMESTAMP}.h5'
EPOCHS = 100
BATCH_SIZE = 256

model = verify_net_model.get_verify_net_model(PRECISION)


def preprocess_item(filename, folder):
    raw_minutiae = np.genfromtxt(
        f"{folder}/{filename}", delimiter=",")

    enhanced_minutiae = utils.enhance_minutiae_points(raw_minutiae)

    return enhanced_minutiae


def load_folder_data(folder):
    data = []
    labels = []

    for _, _, files in os.walk(folder):
        raw_data = [preprocess_item(filename, folder) for filename in files]

        labels = [int(filename.split("_")[0]) for filename in files]

        data = np.stack(raw_data)

    data_shuffled, labels_shuffled = shuffle(data, np.array(labels))

    return data_shuffled, labels_shuffled


def load_dataset(dataset_path):
    print('START: loading data => ', dataset_path)
    data, labels = load_folder_data(dataset_path)

    numClasses = np.max(labels) + 1
    idx = [np.where(labels == i)[0] for i in range(0, numClasses)]

    pairImages = []
    pairLabels = []

    for idxA, _ in enumerate(data):
        # grab the current image and label belonging to the current
        # iteration
        currentImage = data[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label

        idxB = np.random.choice(idx[label])

        posData = data[idxB]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        np.random.shuffle(currentImage)
        pairImages.append([currentImage, posData])
        pairLabels.append([1])

        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        # print(labels)
        negIdx = np.where(labels != label)[0]

        negData = data[np.random.choice(negIdx)]
        # prepare a negative pair of images and update our lists
        np.random.shuffle(currentImage)
        pairImages.append([currentImage, negData])
        pairLabels.append([0])

        # return a 2-tuple of our image pairs and labels
    print('FINISH: loading data')

    return (np.array(pairImages), np.array(pairLabels).astype('float32'))


def split_dataset(pairs, labels):
    length, _ = labels.shape

    train_indices = 0, int(length * 0.8)
    val_indices = int(length * 0.8) + 1, length

    train_dataset = (pairs[:train_indices[1]], labels[:train_indices[1]])
    val_dataset = (pairs[val_indices[0]:val_indices[1]], labels[val_indices[0]:val_indices[1]])

    return train_dataset, val_dataset


def load_and_preprocess_dataset():
    train_pairs, train_labels = load_dataset(TRAIN_DATASET_PATH)
    test_pairs, test_labels = load_dataset(TEST_DATASET_PATH)
    train_pairs_shuffled, train_labels_shuffled = shuffle(train_pairs, train_labels)
    test_pairs_shuffled, test_labels_shuffled = shuffle(test_pairs, test_labels)
    train_dataset, val_dataset = split_dataset(train_pairs_shuffled, train_labels_shuffled)

    return train_dataset, val_dataset, (test_pairs_shuffled, test_labels_shuffled)


def train():
    train_dataset, val_dataset, test_dataset = load_and_preprocess_dataset()

    model.summary()

    def scheduler(epoch, lr):
        if epoch < 55:
            return lr

        return lr * tf.math.exp(-0.1)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=LOGS_FOLDER)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor=['val_accuracy'],
        verbose=1, mode='max', save_weights_only=True)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    train_pairs, train_labels = train_dataset
    val_pairs, val_labels = val_dataset
    test_pairs, test_labels = test_dataset

    model.fit(
        [train_pairs[:, 0], train_pairs[:, 1]],
        train_labels,
        validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[tensorboard, checkpoint, lr_scheduler]
    )

    model.evaluate([test_pairs[:, 0], test_pairs[:, 1]], test_labels)
