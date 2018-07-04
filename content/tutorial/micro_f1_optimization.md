Title: Micro-averaged F1 optimization using neural networks
Date: 07-03-2018
Category: tutorials

Recently I've been working on a multilabel classification problem that was
evaluated using micro-averaged F1 score. In this post I will show one approach
to optimizing the thresholding of probabilistic classifiers by means of a toy
problem. Some assumptions have been made to simplify the explanation, but the
general approach should be applicable to any multilabel micro-averaged F1
optimization situation.

In general, we train the model using previous soft classifier predictions as
training data and the actual labels as ground truth for those predictions.

### Toy data
We will create some random data and create random thresholds for each separate
class in order to create the labels for the data. We will be using 100 classes,
100000 training samples and 10000 validaton samples. We will using treat these
validation samples as unseen data and find out how well our NN stretcher works.

    :::python
    import numpy as np

    # Configure random thresholds for each class
    thresholds = np.random.random(100)

    # Some soft predictions from a different classifier are simmulated here.
    X_train = np.random.random((100000, 100))
    y_train = X_train > thresholds

    X_test = np.random.random((10000, 100))
    y_test = X_test > thresholds

Of course there is a big assumption here, namely that the thresholds for the
validation set are gonna be identical to those of the training set. This is
often not completely the case, but if you use good base models to generate the
soft predictions for this approach, it will still yield good improvements.

### Global thresholding
First, let us examine what happens when we try a set of global thresholds in
order to derive our final predictions and calculate F1 scores for each of the
thresholds. This is the most basic form of F1 optimization but does not allow
for class specific thresholding.

    :::python
    from sklearn.metrics import f1_score

    scores = []
    for th in np.linspace(0, 1, num=20)
        preds = X_test > th
        score = f1_score(y_test, preds, average='micro')
        scores.append((th, score))
    best = max(scores, key=lambda x: x[1])
    print(f'Threshold: {best[0]}, F1 score: {best[1]})

This gives us the following output:

    Threshold: 0.42, F1 score: 0.7935

So let us take that as the maximum we can reach using a global thresholding
technique.

### Neural network 'stretching"
Now we define a neural network that we train on the soft predictions given the
ground truth labels.  The structure of the network is up for experimentation,
but with the choices given below I reached a pretty good result on this toy
data. For the loss function I went with binary crossentropy, which worked well
here but you may also consider using a different loss function such as Hamming
Loss. Additionally, because this toy problem is pretty simple a single
fully-connected layers works fine; you may experiment with the number of layers
in your netwerk and changing it based on obversation of the training
convergence.

    :::python
    from keras.layers import Dense, Input
    from keras.models import Model

    # First define the model; we use one fully-connected layer with linear #
    activation function, following by a Dense layer with the number of classes
    # as neurons and a sigmoid activation function to ascertain values between
    # 0 and 1 which is what we want eventually

    inputs = Input(shape=(100,))
    x = Dense(200)(inputs)
    outputs = Dense(100, activation='sigmoid')(x)

    # Now compile and train the model on the soft predictons, using the actual
    # labels as ground truth.

    model = Model(inputs, outputs)
    model.compile('adam', 'binary_crossentropy')
    model.fit(X_train, y_train, epochs=200, batch_size=64, validation_split=.1)

    # After fitting for 200 epochs with standard Adam parameters, we reach a
    # training loss of 0.0239 and a validation loss of 0.0289.

With the given settings, this training does not take long at all. Of course,
your mileage will vary depending on the size of you data, network and number of
soft predictions for each sample.

Once the model is fitted, we can use it to create predictions on the test data.
We assume that the predictions have been sufficiently transformed by the
network to be able to use a single threshold of 0.5, however you can also
experiment with different global thresholds here to see if you can improve the
results even more. From my findings, if the neural network is trained
sufficiently, we end up with a mean of ~0.5, minimum of 0 and maximum of 1.
That's why I choose to just a 0.5 threshold in this next part.

    :::python
    preds = model.predict(X_test) > .5
    score = f1_score(y_test, preds, average='micro')
    print(f'F1 score: {score:.4f}')

This gives us an output as follows:

    F1 score: 0.9900

As you can see, this is a very big improvement over the global thresholding
used in the previous step. We can only use a single thresholds on the
transformed predictions, great decreasting the amount of time spent on
experimenting with different thresholds.

Additionally you could try different thresholds on these transformed soft
predictions, which I did not choose to do here since the result on this toy
data already gives a large improvement over regular global thresholds that we
tried in the previous experiment.

### Discussion
A toy dataset like this is of course very easy to learn for a simple network.
Perhaps there are more sofisticated approaches possible in cases there is a
large discrepancy in the label distribution between train and test sets, in
which case the stretching of predictions performed by the network may not yield
such a big improvement. If you have base models that already achieve a pretty
good F1 score but you want to push that last extra bit, this might be a good
approach. I have also tried this approach on models that hit around 0.7 F1
score using a global threshold, whereas using this approach the F1 improvement
drastically (to around 0.8 in this particular experiment). The biggest downside
is that the training, validation and test sets should be very similar in a way,
with a clear decision boundare that is approximately identical to that of the
training data.

### Concluded results
| Method                 | Threshold | Best F1 obtained |
| ---------------------- | --------- | ---------------- |
| Best global threshold  | 0.37      | 0.7493           |
| Best NN transformation | 0.50      | 0.9900           |

To compare the performance between optimizing global threshold versus using a
neural network to create an predictions 'transformer', if you will,  that
streches predictions in order to use a single global threshold effectly. If the
base models you are using in your projects are suficciently accurate, this will
definitely yield some improvements. Then again, obtaining good base models is a
whole new and harder issues.

Clearly we can observer a large improvement in F1 micro-averaged score between
the two approaches. Keep in mind that this is just a toy problem, and it is
should be very easy to get to a F1 = 1.0 given enough data, training etc.  This
might not always be the case in real-world data science challenges, since often
the test distribution and training distribution is very different. This means
that you can optimize the F1 optimization transformation based on a validation
data portion, but it might not translate very well to the actual test set. It
really depends on the problem.

Just sharing an idea here, take from it what you want! The runtime for this
code is very quick fortunately, so it's easy to just try it and if it doesn't
improve you results, you can move on to other threshold optimization
techniques.
