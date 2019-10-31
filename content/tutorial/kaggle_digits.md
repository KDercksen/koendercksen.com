Title: Kaggle: Digit Recognizer using simple CNN
Date: 03-02-2018
Category: tutorials

In my recent experimenting with Keras I decided to try my hand at the classical
handwritten digit dataset MNIST. MNIST ("Modified National Institute of
Standards and Technology") is considered the standard "hello world" dataset of
computer vision. The Digit Recognizer [Kaggle
competition](https://kaggle.com/c/digit-recognizer) tasks us with correctly
identifying images of handwritten digits.

### MNIST dataset
The training set `train.csv` contains 42000 images of 28x28 pixels (for a total
of 784 pixels). Each image is labeled with which digit it represents (i.e., we
have 10 classes; 0-9). The test set `test.csv` is made up of 28000 images of
the same dimensions, but of course without labels.

Let's first load these files into Python using `pandas`:

    :::python
    import pandas as pd

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

We'll start by splitting up the training images and labels. We will convert the
images into a `numpy` array with dimensions `(samples, height, width,
channels)`. Since the images are grayscale, we only have one channel. The
labels should be of dimension `(samples, label)`. While we're at it, we'll also
reshape the test set to identical dimensions as the training set.

    :::python
    x_train = train.drop(columns=['label']).values.reshape(-1, 28, 28, 1)
    y_train = train['label'].values.reshape(-1, 1)

    x_test = test.values.reshape(-1, 28, 28, 1)

Before we go any further, we'll also normalize the training and test images to
have zero mean and unit standard deviation to make sure we're dealing with
normalized data when training our model. Clipping is performed to ensure we
don't run into errors caused by dividing by zero:

    :::python
    import numpy as np

    x_train = (x_train - x_train.mean(axis=0)) / np.clip(x_train.std(axis=0), 1e-6, None)
    y_train = (y_train - y_train.mean(axis=0)) / np.clip(y_train.std(axis=0), 1e-6, None)

Since we're dealing with categorical class labels, we will one-hot encode the
training labels.  This is necessary for our Keras model, and fortunately Keras
provides a function to do this for us:

    :::python
    from keras.utils import to_categorical

    y_train = to_categorical(y_train)

Easy enough! Now, finally, we split our training set up into a training portion
and a validation portion; this is used to monitor the performance of the model.
Since the number of training samples is so large, we will only use 5% as
validation samples; this still gives us 2100 validation images, approximately
200 from each class.  We will use `sklearn` for this:

    :::python
    from sklearn.model_selection import train_test_split

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.05)

### CNN
Okay, time to define our model! We will define a simple convolutional neural
network (CNN) with 6 convolution layers, dropout and pooling after each set of
two convolution layers, and two densely connected layers on top. The last
layers will use a `softmax` activation function to do our classification; the
rest of the layers use `relu` as their activation function. For the optimizer
we will use `rmsprop`, and as a metric we will use `accuracy`. The loss
function we use is `categorical_crossentropy`, since we're dealing with a
categorical classification problem where we have more than 2 classes.

    :::python
    import keras.models as km
    import keras.layers as kl

    optimizer = 'rmsprop'
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    model = km.Sequential()

    # First convolution pool
    model.add(kl.Conv2D(32, 3, padding='same', input_shape=x_train.shape[1:]))
    model.add(kl.Activation('relu'))
    model.add(kl.Conv2D(32, 3))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPooling2D())
    model.add(kl.Dropout(.25))

    # Second convolution pool
    model.add(kl.Conv2D(64, 3, padding='same'))
    model.add(kl.Activation('relu'))
    model.add(kl.Conv2D(32, 3))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPooling2D())
    model.add(kl.Dropout(.25))

    # Third convolution pool
    model.add(kl.Conv2D(64, 3, padding='same'))
    model.add(kl.Activation('relu'))
    model.add(kl.Conv2D(64, 3))
    model.add(kl.Activation('relu'))
    model.add(kl.MaxPooling2D())
    model.add(kl.Dropout(.25))

    # Dense top layers
    model.add(kl.Flatten())
    model.add(kl.Dense(256))
    model.add(kl.Activation('relu'))
    model.add(kl.Dense(y_train.shape[1]))
    model.add(kl.Activation('softmax'))

    model.compile(optimizer, loss, metrics=metrics)
    model.summary()

`model.summary()` will print a layout of the compiled network.

### Data augmentation
In order to reduce overfitting and improve the generalization of our network,
we will use Keras' `ImageDataGenerator` to provide us with augmented training
data. We can specify various augmentation parameters such as random rotation
range, zoom range, width/height shift range and more. Once fitted to our
training data, the generator will supply the network with batches of training
data augmented on the fly.

    :::python
    from keras.preprocessing.image import ImageDataGenerator

    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=.3,
        height_shift_range=.3,
        zoom_range=.3
    )

    data_gen.fit(x_train)

### Training the model
Time to train the network! We create two callbacks to be used during training.
The first callback will only save the best network (by validation accuracy)
acquired during training to disk. The second callback will reduce the learning
rate of the network during training when a plateau in the validation loss score
is encountered.

    :::python
    import keras.callbacks as kc

    checkpoint = kc.ModelCheckpoint('best_model.h5', monitor='val_acc', save_best_only=True)
    reduce_lr = kc.ReduceLROnPlateau(monitor='val_loss')

Now we define some learning parameters and call the `fit_generator` method on
our model, using the training data generator as input:

    :::python
    n_epoch = 50
    batch_size = 32

    results = model.fit_generator(data_gen.flow(x_train, y_train, batch_size),
                                  validation_data=(x_val, y_val),
                                  callbacks=[checkpoint, reduce_lr],
                                  steps_per_epoch=500,
                                  epochs=n_epoch,
                                  verbose=1)

After training, we can use the obtained `results` object to visualize our accuracy
and loss curves:

    :::python
    import matplotlib.pyplot as plt

    plt.figure(figsize=[10, 4]
    plt.plot(results.history['acc'], 'r')
    plt.plot(results.history['loss'], 'g')
    plt.plot(results.history['val_acc'], 'b')
    plt.plot(results.history['val_loss'], 'y')
    plt.legend(['acc', 'loss', 'val_acc', 'val_loss'])
    plt.suptitle('Training curves')
    plt.xlabel('Number of epochs')
    plt.ylabel('Score')
    plt.show()

![Training visualization]({static}/files/kaggle_digits_trainingplot.png.keep)

This is exactly what we like to see! Training and validation accuracy increase,
while both losses decrease. This indicates that little overfitting occurs and
the network is working well.

### Submission time!
Once training is done, the best model found is saved to `best_model.h5` as we
specified in the checkpoint callback. Now we can load the model, and predict
our test set.  We write the results into a CSV file which we can submit to
Kaggle!

    :::python
    model = km.load_model('best_model.h5')

    predictions = model.predict(x_test)
    # Pick the highest probability class label
    y_pred = predictions.argmax(axis=1)

    with open('submission.csv', 'w') as f:
        f.write('ImageId,Label\n')
        idxs = range(y_pred.shape[0])
        f.write('\n'.join([f'{i+1},{y_pred[i]}' for i in idxs]))

Submitting these predictions should already give you an accuracy upwards of
99%. There is room for improvement; the best submissions on Kaggle have 100%
accuracy! This is where you get creative. :)
