import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def aa_encoder(sequences, max_length, dataset):
    encoded = np.zeros((len(sequences), max_length))

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            try:
                encoded[i, j] = dataset[aa]
            except KeyError:
                continue

    return encoded


def encode_sequences(sequences, max_length, datasets):
    """
    Assume each dataset provides a 1D encoding per sequence
    """

    encoded = np.zeros((len(sequences), max_length, len(datasets)))

    for i, dataset in enumerate(datasets):
        encoded[:, :, i] = aa_encoder(sequences, max_length, dataset)

    return encoded


class MinMaxScaler2D(BaseEstimator, TransformerMixin):
    def __init__(self, mask, copy=True):
        """
        Equivalent to StandardScaler but ignores mask, and keeps the value at this position untouched.
        Simpler class as it is only used in this specific case.
        Also is used for RNN processing.
        """
        self.mask = mask
        self.copy = copy

    def fit(self, X, y=None):
        # get min and max of this feature2
        self.max_ = (X[X != self.mask]).max()
        self.min_ = (X[X != self.mask]).min()

        return self

    def transform(self, X, y=None):

        check_is_fitted(self, 'max_')

        if self.copy:
            X = X.copy()
        mask_ = X == self.mask

        X = (X - self.min_) / (self.max_ - self.min_)

        X[mask_] = self.mask
        np.nan_to_num(X, copy=False)

        return X

    def inverse_transform(self, X, y=None):

        check_is_fitted(self, 'max_')

        if self.copy:
            X = X.copy()
        mask_ = X == self.mask

        X = X * (self.max_ - self.min_) + self.min_

        X[mask_] = self.mask

        return X


class MinMaxScaler3D(BaseEstimator, TransformerMixin):
    def __init__(self, mask, copy=True):
        """
        3D matrix scaling for RNN preparation with mask
        """
        self.copy = copy
        self.mask = mask

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : Passthrough for ``Pipeline`` compatibility.
        """
        dims = X.shape
        n_scalers = dims[2]
        # create list of scalers
        self.scalers_ = [MinMaxScaler2D(**self.get_params()) for x in range(n_scalers)]
        #         self.scalers_ = [MinMaxScaler() for x in range(n_scalers)]
        # fit each scaler
        for i in range(n_scalers):
            self.scalers_[i].fit(X[:, :, i])
        return self

    def transform(self, X, y=None):
        result = np.empty((X.shape))
        check_is_fitted(self, 'scalers_')
        # check dims
        if len(self.scalers_) != X.shape[2]:
            raise ValueError("Dim 3 must match!")
        for i in range(X.shape[2]):
            # transform data
            result[:, :, i] = self.scalers_[i].fit_transform(X[:, :, i])
        return result

    def inverse_transform(self, X, y=None):
        result = np.empty((X.shape))
        check_is_fitted(self, 'scalers_')
        # check dims
        if len(self.scalers_) != X.shape[2]:
            raise ValueError("Dim 3 must match!")
        for i in range(X.shape[2]):
            # transform data
            result[:, :, i] = self.scalers_[i].inverse_transform(X[:, :, i])
        return result


def shuffle_array(array1, array2):
    idx = np.random.choice(range(len(array1)), size=len(array1), replace=False)
    new_array_1 = np.empty_like(array1)
    new_array_2 = np.empty_like(array2)

    for i, j in enumerate(idx):
        new_array_1[i] = array1[j]
        new_array_2[i] = array2[j]

    return new_array_1, new_array_2
