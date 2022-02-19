import numpy as np

from .configs import NN_BATCH_SIZE

###


class Generator:

    def __init__(self, X, Y, batch_mode: str, passage_indexes=None):
        assert isinstance(X, list)
        assert isinstance(Y, np.ndarray)
        assert X[0].shape[0] == Y.shape[0]
        self.X = X
        self.Y = Y
        self.nrows = self.X[0].shape[0]

        assert isinstance(batch_mode, str)
        assert batch_mode == "size" or batch_mode == "passage"
        self.batch_mode = batch_mode

        if self.batch_mode == "passage":
            assert isinstance(passage_indexes, np.ndarray)
            assert len(passage_indexes.shape) == 1
            assert X[0].shape[0] == passage_indexes.shape[0]
            self.passage_indexes = passage_indexes
        elif self.batch_mode == "size":
            self.passage_indexes = None

    @property
    def steps_per_epoch(self) -> int:
        if self.batch_mode == "passage":
            return np.unique(self.passage_indexes).shape[0]

        if self.batch_mode == "size":
            if (self.nrows % NN_BATCH_SIZE) == 0:
                return int(self.nrows / NN_BATCH_SIZE)
            else:
                return int(self.nrows / NN_BATCH_SIZE) + 1

        return None

    #

    @staticmethod
    def generate_batches_grouped_by_passage(X, Y, passage_idxs):
        X_batches = []
        Y_batches = []

        for passage_idx in np.unique(passage_idxs):
            passage_rows_subset_idx = np.array(passage_idxs == passage_idx)
            X_batches.append([feature[passage_rows_subset_idx] for feature in X])
            Y_batches.append(Y[passage_rows_subset_idx])

        assert len(X_batches) == len(Y_batches)

        return X_batches, Y_batches

    #

    def __batches_grouped_by_passage(self):
        X, Y = Generator.generate_batches_grouped_by_passage(self.X, self.Y, self.passage_indexes)

        passage_batch_idx = 0

        while True:
            if passage_batch_idx >= len(X):
                passage_batch_idx = 0

            yield X[passage_batch_idx], Y[passage_batch_idx]
            passage_batch_idx += 1

    def __batches_grouped_by_size(self):
        steps = -NN_BATCH_SIZE

        while True:
            if steps > self.nrows:
                steps = 0

            X_list = []
            for feature in self.X:
                X_list.append(feature[steps:NN_BATCH_SIZE + steps])

            yield X_list, self.Y[steps:NN_BATCH_SIZE + steps]
            steps = steps + NN_BATCH_SIZE

    #

    def batches(self):
        if self.batch_mode == "size":
            return self.__batches_grouped_by_size()

        if self.batch_mode == "passage":
            return self.__batches_grouped_by_passage()

        return None
