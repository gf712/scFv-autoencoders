import keras

class CosineAnnealingAdaptiveLRScheduler(keras.callbacks.Callback):

    """
    Keras adaptive learning scheduler implementation of cosine annealing
    that resets every cycle. A cycle unit is defined as the number of
    batches per epoch. The cycle can then be adjusted by a scalar after
    each cycle ending
    """

    def __init__(self, lr=None, cycle_len=1, cycle_mult=2):

        """
        CosineAnnealingAdaptiveLRScheduler constructor.
        Args:
            lr: base learning rate. If set to None it will be inferred from the model optimiser.
            cycle_len: initial cycle length. If set to 1 it will be equivalent to the number of batches per epoch
            cycle_mult: scalar to adjust cycle length at the end of each epoch
        """

        super(CosineAnnealingAdaptiveLRScheduler, self).__init__()

        self.lr = lr
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.iteration = 0
        self.steps_per_epoch = None
        self.current_cycle_iterations = None

    def on_batch_begin(self, epoch, logs=None):

        """
        Callback to adjust the learning rate at the start of each batch
        Args:
            epoch:
            logs:
        Returns:
        """

        if self.current_cycle_iterations < self.iteration:
            # update cycle length
            self.cycle_len *= self.cycle_mult
            self.current_cycle_iterations = self._get_iter_per_cycle()
            # reset state to start new cycle
            self._reset()

        # cosine annealing -> the self.current_cycle_iterations+1 avoids cos(pi) which is 0
        new_lr = (np.cos(self.iteration / (self.current_cycle_iterations + 1) * np.pi) + 1) * self.lr

        K.set_value(self.model.optimizer.lr, new_lr)

#         print('\nIteration {}/{}: {}\n'.format(self.iteration, self.current_cycle_iterations, new_lr))

        self.iteration += 1

    def _get_iter_per_cycle(self):
        """
        Calculate the number of iterations in current cycle
        Returns:
        """
        return self.steps_per_epoch * self.cycle_len

    # adapted from https://github.com/uber/horovod/blob/master/horovod/keras/callbacks.py
    def on_train_begin(self, logs=None):
        """
        Sets some attributes that can only be inferred at model runtime.
        Args:
            logs:
        Returns:
        """
        if self.lr is None:
            self.lr = K.get_value(self.model.optimizer.lr)
        self.steps_per_epoch = self._autodetect_steps_per_epoch()
        self.current_cycle_iterations = self.steps_per_epoch * self.cycle_len

        # compensates for cosine annealing -> could divide by 2 at each step, or just do it here
        self.lr /= 2

    # https://github.com/uber/horovod/blob/master/horovod/keras/callbacks.py
    def _autodetect_steps_per_epoch(self):
        """
        Determine the number of steps per epoch.
        Returns:
        """
        if self.params.get('steps'):
            # The number of steps is provided in the parameters.
            return self.params['steps']
        elif self.params.get('samples') and self.params.get('batch_size'):
            # Compute the number of steps per epoch using # of samples and a batch size.
            return self.params['samples'] // self.params['batch_size']
        else:
            raise ValueError('Could not autodetect the number of steps per epoch. '
                             'Please specify the steps_per_epoch parameter to the '
                             '%s() or upgrade to the latest version of Keras.'
                             % self.__class__.__name__)

    def _reset(self):
        """
        Reset the state of the learning rate to start a new cycle.
        Returns:
        """
        self.iteration = 0