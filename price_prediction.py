import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

tf.__version__

raw_dataset = pd.read_csv('./data/orders.csv')
df = raw_dataset.copy()

parts_of_day = list(map(lambda t: datetime.strptime(t, '%H:%M').time(),
                        ["00:00", "06:00", "12:00", "18:00", "23:00"]))


def part_of_day(date, parts):
    date = datetime.strptime(date, '%m/%d/%Y %H:%M').time()

    if parts_of_day[0] >= date < parts[1]:
        return 1

    if parts_of_day[1] >= date < parts[2]:
        return 2

    if parts_of_day[2] >= date < parts[3]:
        return 3

    return 4


parts_of_day = list(map(lambda t: datetime.strptime(t, '%H:%M').time(),
                        ["00:00", "06:00", "12:00", "18:00", "23:00"]))


def part_of_day(date, parts):
    date = datetime.strptime(date, '%m/%d/%Y %H:%M').time()

    if parts_of_day[0] >= date < parts[1]:
        return 1

    if parts_of_day[1] >= date < parts[2]:
        return 2

    if parts_of_day[2] >= date < parts[3]:
        return 3

    return 4


def distance(lat1, lon1, lat2, lon2):
    cos_l1 = np.cos(lat1)
    cos_l2 = np.cos(lat2)
    sin_l1 = np.sin(lat1)
    sin_l2 = np.sin(lat2)
    delta = lon1 - lon2

    y = np.sqrt(np.square(cos_l2 * np.sin(delta)) + np.square(cos_l1 * sin_l2 - sin_l1 * cos_l2 * np.cos(delta)))
    x = sin_l1 * sin_l2 + cos_l1 * cos_l2 * np.cos(delta)

    return np.arctan2(y, x) * 6372795


df["distance"] = distance(df["dropoff_lat"], df["dropoff_lon"], df["pickup_lat"], df["pickup_lon"])

df["part_of_day"] = [part_of_day(date, parts_of_day) for date in df["pickupTime"]]
part_of_day = df.pop('part_of_day')
df["night"] = (part_of_day == 1) * 1.0
df["morning"] = (part_of_day == 2) * 1.0
df["afternoon"] = (part_of_day == 3) * 1.0
df["evening"] = (part_of_day == 4) * 1.0

dataset = df[['distance', 'night', 'morning', 'afternoon', 'evening', 'fare']]
dataset = dataset.drop([326112, 734967])
train_dataset = dataset.sample(frac=0.8, random_state=2020)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["distance", "fare"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop('fare')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('fare')
test_labels = test_dataset.pop('fare')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(1e-3)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])

    return model


model = build_model()
model.summary()

EPOCHS = 50

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    normed_train_data, train_labels, batch_size=256,
    epochs=EPOCHS, validation_split=0.2, callbacks=[early_stop])


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [fare]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0.75,2])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$fare^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([7,10])
  plt.legend()
  plt.show()


plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels)

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Fare]')
plt.ylabel('Predictions [Fare]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Fare]")
_ = plt.ylabel("Count")

model.save('./data/model')
