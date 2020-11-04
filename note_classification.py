import tensorflow as tf
from PIL import Image
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt

disk_dir = Path("data/disk/")
test_dir = Path("data/test/")

def read_many_disk(num_images, dir):
    images, labels = [], []

    for image_id in range(num_images):
        images.append(np.array(Image.open(dir / f"{image_id}_gray.png")))

    with open(dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            label_list = row[0].split(',')
            labels.append(int(label_list[1]))
    return images, labels

def remove_channels(num_images):
    for image_id in range(num_images):
        img = Image.open(test_dir / f"{image_id}.png")
        img.save(test_dir / f"{image_id}_gray.png")

# remove_channels(4)
(train_images, train_labels) = read_many_disk(101, disk_dir)
(test_images, test_labels) = read_many_disk(4, test_dir)

# print(train_images[0].shape)
# print(len(train_images))
# print(train_labels)
# print(test_images[0].shape)
# print(len(test_labels))

# train_images = train_images / 255.0
# test_images = test_images / 255.0

note_list = ["double whole", "whole", "half", "quarter", "eighth", "sixteenth"]

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(note_list[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(45, 78)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1000)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)