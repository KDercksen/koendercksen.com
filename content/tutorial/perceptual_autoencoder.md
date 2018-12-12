Title: Clustering image encodings with perceptual autoencoders
Date: 12-12-2018
Category: tutorials

Autoencoders are used to learn efficient data encodings in an unsupervised
manner.  When using image data, autoencoders are normally trained by minimizing
their reconstruction error (typically the mean squared error or MSE) in an
identity mapping task. The error then represents the difference between the
input image and the reconstruction of that image. However, it is not necessary
for the reconstructions to be pixel-perfect; the important thing is that the
embeddings contain useful information that separates different classes of
images. In this post, I showcase a different method: train the autoencoder to
minimize a *perceptual* or feature loss function. I use a classification
network to extract features from the images, and measure the reconstruction
error as the MSE between the features of the original image and the
reconstructed image. Applying k-means clustering to the obtained image
embeddings, it is shown that using a perceptual loss function results in a
higher accuracy when assigning unseen images to cluster centroids.

Since I have recently been working with TensorFlow's Eager execution, I will be
using that to implement this idea. I will use Fashion-MNIST as an example.
There's some basic flipping augmentation in order to make the features learned
by the classification network more robust.

First off, imports and defining datasets:

    :::python
    from functools import partial
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf

    tf.enable_eager_execution()

    # Load data and normalize
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[..., None].astype(np.float32) / 255
    x_test = x_test[..., None].astype(np.float32) / 255
    y_train_onehot = tf.keras.utils.to_categorical(y_train)
    y_test_onehot = tf.keras.utils.to_categorical(y_test)

    # Data augmentation functions
    def augment(x, t, flip_target=False):
        if np.random.random() > 0.5:
            x = tf.image.flip_left_right(x)
            if flip_target:
                t = tf.image.flip_left_right(t)
        return x, t

    # Some training parameters
    batchsize = 64
    latent_features = 10
    num_epochs = 20
    num_clusters = 20

    # Classifier datasets
    train = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot))
        .shuffle(len(x_train))
        .map(augment, num_parallel_calls=8)
        .batch(batchsize)
        .repeat()
    )
    test = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test_onehot))
        .batch(64)
        .repeat()
    )

    # Autoencoder datasets
    ae_train = (
        tf.data.Dataset.from_tensor_slices((x_train, x_train))
        .shuffle(len(x_train))
        .map(partial(augment, flip_target=True), num_parallel_calls=8)
        .batch(batchsize)
        .repeat()
    )
    ae_test = (
        tf.data.Dataset.from_tensor_slices((x_test, x_test))
        .batch(64)
        .repeat()
    )

    # The repeat calls are necessary for tf.keras.Model.fit to work

    train_steps = len(x_train) // batchsize
    val_steps = len(x_test) // batchsize

For the purpose of this demonstration, we will use the test portion of the data
for validation purposes. With that out of the way, we'll first train a
classification network to later use as a feature extractor when training the
autoencoder.

    :::python
    class Classifier(tf.keras.Model):

        def __init__(self):
            super().__init__()
            self.features = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
            ])
            self.classifier = tf.keras.layers.Dense(10, activation='softmax')

        def call(self, x, training=False):
            features = self.features(x, training=training)
            return self.classifier(features)


    classifier = Classifier()
    classifier.compile(loss='categorical_crossentropy', optimizer=tf.train.AdamOptimizer())
    classifier.fit(
        train,
        epochs=num_epochs,
        validation_data=test,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=1,
    )

Once this classifier is trained (it achieves around 89% accuracy fairly
quickly), we can continue by building the autoencoder and defining our
perceptual loss function. The encoder converts the 28x28 image into a single
vector of length `latent_features`. The decoder upsamples the embedding back up
to 28x28. The feature loss is the minimum feature error with the target either
flipped or not; we don't care if the reconstruction is flipped, as long as it's
the same object.

    :::python
    class AE(tf.keras.Model):

        def __init__(self, latent_features):
            super().__init__()
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
                tf.keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
                tf.keras.layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_features),
            ])
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(128 * 4 * 4, activation='relu'),
                tf.keras.layers.Reshape([4, 4, 128]),
                tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.UpSampling2D(),
                tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(1, 1, activation='relu'),
            ])

        def call(self, x):
            return self.decoder(self.encoder(x))


    def feature_loss(t, x, model=None):
        target_features = model.features(t, training=False)
        target_features_flipped = model.features(tf.image.flip_left_right(t), training=False)
        prediction_features = model.features(x, training=False)
        return min(
            tf.reduce_sum(tf.square(target_features - prediction_features)),
            tf.reduce_sum(tf.square(target_features_flipped - prediction_features)),
        )


    autoencoder = AE(latent_features)
    # loss='mse' will use regular MSE
    autoencoder.compile(loss=partial(feature_loss, model=classifier), optimizer=tf.train.AdamOptimizer())
    autoencoder.fit(
        ae_train,
        epochs=num_epochs,
        validation_data=ae_test,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=1,
    )

Fairly simple! Once the training is done, we can visualize the reconstructions
created by the autoencoder.

    :::python
    num_imgs = 10
    original_imgs = x_test[np.random.choice(range(len(x_test)), num_imgs)]
    reconstructions = autoencoder(original_imgs)

    fig, axes = plt.subplots(2, num_imgs)
    for i in range(num_imgs):
        axes[0, i].imshow(original_imgs[i].squeeze(), cmap='gray')
        axes[1, i].imshow(reconstructions[i].numpy().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()

![AE reconstructions with 10 latent features and regular MSE]({filename}/files/ae_10_reconstructions_mse.png.keep)
![AE reconstructions with 10 latent features and feature MSE]({filename}/files/ae_10_reconstructions_feature.png.keep)

The first image shows reconstructions by the regularly trained autoencoder; the
second one shows reconstructions from the feature loss autoencoder (first row
is original image, second is reconstruction). Both networks have 10 latent
features. The reconstructions from the feature loss autoencoder are worse, but
we will show that the embeddings contain better information at the end of this
post.

Now we will obtain embeddings for all images in the test set, and cluster them
using k-means. Normally, we would create clusters from the training data;
however, performance on the test set is similar to that on the train set. We
use the test set to save some time, since it only contains 10000 images.

    :::python
    features = []
    # Zip is necessary to circumvent the "repeat()" used in the dataset
    for (imgs, _), _ in zip(test, range(val_steps + 1)):
        features.append(autoencoder(imgs).numpy())
    features = np.concatenate(features)

    km = KMeans(n_clusters=num_clusters)
    km.fit(features)

Now that we have calculated the cluster centroids, we can assign class labels
to them by assigning all test images to their closest centroid and assigning
the majority label from those images to that particular centroid. We create a
dictionary with a mapping from centroid ID to class label for ease of use.

    :::python
    assignments = km.predict(features)
    mapping = {}
    for i in range(len(km.cluster_centers_)):
        idxs = assignments == i
        best_class = np.argmax(np.bincount(y_test[idxs]))
        mapping[i] = best_class

The cluster centers have class labels! Now we can easily calculate the accuracy
of these cluster labels by comparing them with the ground truth:

    :::python
    predictions = [mapping[x] for x in assignments]
    print(accuracy_score(y_test, predictions))

That's it! Some results are listed below. The scores listed were obtained by
using `num_clusters=20`. We can see that the autoencoder trained using feature
loss scores significantly higher than the one trained with regular MSE.
Training longer could improve the results; I limited the number of epochs to 10
for these accuracy scores.  Accuracy for both will increase with a higher
number of clusters and/or latent features, but the pattern remains the same in
these experiments.

| Latent features | MSE | Feature loss |
|:--- |:---:|:---:|
| 5 | 0.7048 | **0.7557**  |
| 10 | 0.6983 | **0.7753**  |
| 20 | 0.7003 | **0.7737**  |
