Title: Kaggle: Facial Keypoints Detection
Date: 22-05-2016
Category: tutorial

This week I've been working on the Facial Keypoint Detection competition hosted
by [Kaggle](http://kaggle.com). The objective of the competition is to predict
keypoint positions on face images.

We start out with three files: the training set `training.csv`, the test set
`test.csv` and a list of 27214 keypoints to predict in `IdLookupTable.csv`.

Let's first load these files into Python:

    :::python
    import numpy as np
    import pandas as pd

    df = pd.read_csv('training.csv')
    testdf = pd.read_csv('test.csv')
    lookup = pd.read_csv('IdLookupTable.csv')

Inspecting our training set using `df.info()`, `df.shape` and `df.describe()`
gives us some useful information. We see that there are 31 columns, 30 of which
are keypoint coordinates. The last column is a space-separated string of pixel
values, ordered by rows. The next important thing is that there are quite some
keypoints missing; we'll have to do something about that.

Given this info, we can do some more preprocessing on our data:

    :::python
    # Make image column a numpy array
    for d in [df, testdf]:
        d['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    # fill up missing values with the mean of respective column
    df = df.fillna(df.mean())

    # create training samples/labels for model fitting
    X = np.vstack(df.ix[:, 'Image']).astype(np.float)
    y = df.drop('Image', axis=1).values 

Voila, we can now use `X` and `y` to train a model on our data. We have several
options here: neural networks could potentially be very accurate. However, I'm
not very proficient with those yet, so I chose to use a relatively simple
regression model, k-nearest neighbors. This model predicts a value by finding
it's k-nearest neighbors and returning the mean of those as a prediction.

    :::python
    from sklearn.neighbors import KNeighborsRegressor as KNR

    estim = KNR()
    estim.fit(X, y)

This train a k-nearest neighbors regression model with the default values
`scikit-learn` provides. Using this model, we can do our first submission
to the competition! We need to predict keypoints specified in the `lookup`
data. Relevant fields are `ImageId` (points to a row in `test.csv`) and
`FeatureName` (specifies the feature we need to predict).

I simply predicted every keypoint for the whole test set and then molded the
results into the submission format `RowId, Location`.

    :::python
    # stack all test images into one numpy array
    Y = np.vstack(testdf.ix[:, 'Image']).astype(np.float)

    # predict all keypoints for the images in Y
    predictions = estim.predict(Y)

    # now create the result data and write to csv
    preddf = pd.DataFrame(predictions, columns=df.columns[:-1])
    results = pd.DataFrame(columns=['RowId', 'Location'])

    for i in range(lookup.shape[0]):
        d = lookup.ix[i, :]
        r = pd.Series([d['RowId'], preddf.ix[d['ImageId']-1, :][d['FeatureName']]],
                      index=results.columns)
        results = result.append(r, ignore_index=True)

    results['RowId'] = results['RowId'].astype(int)
    results.to_csv('predictions.csv', index=False)

And we're done! Submitting this placed me halfway up the leaderboard.
Some ways to increase accuracy include experimenting with `k` (default
value is 5), and being smarter about data imputation (we only filled in empty
fields with the column mean; manually labeling the images or otherwise getting
more precise values would be way more accurate).
