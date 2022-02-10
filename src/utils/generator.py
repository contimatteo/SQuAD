from .configs import NN_BATCH_SIZE
import gc
# X = [q_tokens, p_tokens, p_match, p_pos, p_ner, p_tf]
# Y = Y


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
        steps = 0
        while True:
            #print(steps)
            if steps > self.nrows:
                #print("steps")
                steps = 0
            X_list = []
            for i in range(6):
                X_list.append(self.X[i][steps:NN_BATCH_SIZE+steps])
            yield X_list, self.Y[steps:NN_BATCH_SIZE+steps]
            steps = steps + NN_BATCH_SIZE
            gc.collect()

    # def generate(self):
    #     steps = 0
    #     nrows = self.X[0].shape[0]
    #     while steps <= nrows:
    #         X_list = []
    #         for i in range(6):
    #             X_list.append(self.X[i][steps:steps + NN_BATCH_SIZE])
    #         yield X_list, self.Y[steps:steps + NN_BATCH_SIZE]
    #
    #         steps = steps + NN_BATCH_SIZE
