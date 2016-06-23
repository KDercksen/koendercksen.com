Title: Performance rating for fuzzy binary classifiers
Date: 23-06-2016
Category: tutorials

This post proposes an idea for rating the performance of a fuzzy binary
classifier (yes, I made that name up myself). A friend asked me if I knew of a
way to measure accuracy for a classifier that predicted the emotion showed by
pictures of human faces with a probability distribution (i.e. 60% sad, 30%
angry, 10% happy). The labels for the dataset are however binary (if the
emotion of face `n` is sad, the label set will have a `1` value for the sad
column and zeroes in all other columns).

My idea is actually very simple; sum the probabilities reported for the actual
emotions, and check how large that sum is compared to the total number of
pictures in the dataset. This way of rating performance ensures that the weight
of the prediction correctness (in other words, the probability reported for the
actual emotion) is taken into account.

I wrote a little snippet of Python code to demonstrate my thinking. I generated
some random test data, and generated labels for that test data based on the
following rule: with probability `.7`, the highest probability in the test row
is the actual emotion, and otherwise a random emotion is chosen. This makes the
test labels a little bit more realistic (since the classifier is not going to
be 100% correct everytime).

    :::python
    # Generate classifier output.
    X = np.ones((100, 10))
    for i in range(X.shape[0]):
        # Dirichlet distribution sums to 1
        X[i, :] = np.random.dirichlet(X[i, :], size=1)

    # Generate labels: probability that largest score in output is correct is
    # .7 and with probability .3 a random label is chosen.
    y = np.ones(100)
    for i in range(y.shape[0]):
        p = np.random.rand()
        if p >= 0.7:
            y[i] = np.argmax(X[i, :])
        else:
            y[i] = np.random.randint(10)

So, in summary: `X` is a `100x10` matrix with classifier predictions (100
pictures with probabilities for 10 emotions per picture). `y` is a list of
labels for the test data, where each label is stored as the index of the
correct emotion (some index in the range 0-10).

Now we can measure the performance on these two matrices using the following code:

    :::python
    # Calculate performance by multiplying the label index by the classifier
    # score on that index and dividing by number of samples.
    score = 0
    for i in range(y.shape[0]):
        score += X[i, y[i].astype(int)]

    print('Performance: {:.2f}/{}'.format(score, 100))

Fairly straightforward! This is not the most sophisticated way of doing things,
but it might at least give you an idea of how accurate your classifier actually
is.
