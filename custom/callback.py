from tensorflow import keras
import tensorflow as tf
import params as par
import sys
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule


class MTFitCallback(keras.callbacks.Callback):

    def __init__(self, save_path):
        super(MTFitCallback, self).__init__()
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.save_path)


class TransformerLoss(keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, from_logits=False, reduction='none', debug=False,  **kwargs):
        super(TransformerLoss, self).__init__(from_logits, reduction, **kwargs)
        self.debug = debug
        pass

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        mask = tf.math.logical_not(tf.math.equal(y_true, par.pad_token))
        mask = tf.cast(mask, tf.float32)
        _loss = super(TransformerLoss, self).call(y_true, y_pred)
        _loss *= mask
        if self.debug:
            tf.print('loss shape:', _loss.shape, output_stream=sys.stdout)
            tf.print('output:', tf.argmax(y_pred,-1), output_stream=sys.stdout)
            tf.print(mask, output_stream=sys.stdout)
            tf.print(_loss, output_stream=sys.stdout)
        return _loss


def transformer_dist_train_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.math.logical_not(tf.math.equal(y_true, par.pad_token))
    mask = tf.cast(mask, tf.float32)

    y_true_vector = tf.one_hot(y_true, par.vocab_size)

    _loss = tf.nn.softmax_cross_entropy_with_logits(y_true_vector, y_pred)
    # print(_loss.shape)
    #
    # _loss = tf.reduce_mean(_loss, -1)
    _loss *= mask

    return _loss


class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=165):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        super(CustomSchedule, self).get_config()

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

#Adapted to use Cosine Annealing (does not depend on the embedding dimension anymore)
class CustomScheduleCA(LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, name='CosineDecay', warmup_steps=165):
        super(CustomScheduleCA, self).__init__()

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.name = name

        # Cosine decay schedule
        self.cosine_decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=self.decay_steps,
            alpha=self.alpha,
            name=self.name
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        if step < self.warmup_steps:
            warmup_lr = self.initial_learning_rate * (step / tf.cast(self.warmup_steps, tf.float32))
            return tf.cast(warmup_lr, tf.float32)
        else:
            cosine_lr = self.cosine_decay_schedule(step - self.warmup_steps)
            return tf.cast(cosine_lr, tf.float32)

    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'alpha': self.alpha,
            'name': self.name,
            'warmup_steps': self.warmup_steps
        }

#Constant learning rate With Linear Warmup (last fine-tune)
class CustomScheduleCWLW(LearningRateSchedule):
    def __init__(self, learning_rate, warmup_steps=165):
        super(CustomScheduleCWLW, self).__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.learning_rate * (step / tf.cast(self.warmup_steps, tf.float32))
        else:
            return self.learning_rate


if __name__ == '__main__':
    import numpy as np
    # loss = TransformerLoss()(np.array([[1],[0],[0]]), tf.constant([[0.5,0.5],[0.1,0.1],[0.1,0.1]]))
    # print(loss)

    import matplotlib.pyplot as plt
    # Usage example
    initial_learning_rate = 0.1
    decay_steps = 10000
    alpha = 0.01
    warmup_steps = 1000

    learning_rate_schedule = CustomSchedule(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha,
        warmup_steps=warmup_steps
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)

    # Define the model properly with an Input layer
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    # Visualize the learning rate schedule
    steps = range(decay_steps + warmup_steps)
    learning_rates = [learning_rate_schedule(step).numpy() for step in steps]

    plt.plot(steps, learning_rates)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Custom Schedule with Warmup and Cosine Decay')
    plt.show()