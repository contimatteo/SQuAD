from .configs import NN_BATCH_SIZE

###


class Generator:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.nrows = self.X[0].shape[0]
        self.steps_per_epoch = self.set_steps_per_epoch()

    def get_steps_per_epoch(self):
        return self.steps_per_epoch

    def set_steps_per_epoch(self):
        # print(self.nrows / NN_BATCH_SIZE)
        if (self.nrows % NN_BATCH_SIZE) == 0:
            return int(self.nrows / NN_BATCH_SIZE)
        else:
            return int(self.nrows / NN_BATCH_SIZE) + 1

    # def generate(self):
    #     steps = NN_BATCH_SIZE
    #     while True:
    #         if steps > self.nrows:
    #             steps = NN_BATCH_SIZE
    #         X_list = []
    #         for i in range(6):
    #             X_list.append(self.X[i][steps - NN_BATCH_SIZE:steps])
    #         yield X_list, self.Y[steps - NN_BATCH_SIZE:steps]
    #         steps = steps + NN_BATCH_SIZE
    #         gc.collect()

    def generate_dynamic_batches(self):
        steps = -NN_BATCH_SIZE

        while True:
            if steps > self.nrows:
                steps = 0

            X_list = []

            for feature in self.X:
                X_list.append(feature[steps:NN_BATCH_SIZE + steps])

            yield X_list, self.Y[steps:NN_BATCH_SIZE + steps]

            steps = steps + NN_BATCH_SIZE
