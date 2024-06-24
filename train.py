from model import MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.keras.optimizers import Adam
from data import Data
import utils
import argparse
import datetime
import sys
import tensorflow as tf
tf.executing_eagerly()

parser = argparse.ArgumentParser()

parser.add_argument('--l_r', default=None, help='학습률', type=float)
parser.add_argument('--batch_size', default=2, help='batch size', type=int)
parser.add_argument('--pickle_dir', default='music', help='데이터셋 경로')
parser.add_argument('--max_seq', default=2048, help='longueur maximale', type=int)
parser.add_argument('--epochs', default=100, help='에폭 수', type=int)
parser.add_argument('--load_path', default=None, help='모델 로드 경로', type=str)
parser.add_argument('--save_path', default="result/dec0722", help='모델 저장 경로')
parser.add_argument('--is_reuse', default=False)
parser.add_argument('--multi_gpu', default=True)
parser.add_argument('--num_layers', default=6, type=int)

args = parser.parse_args()


# set arguments
l_r = args.l_r
batch_size = args.batch_size
pickle_dir = args.pickle_dir
max_seq = args.max_seq
epochs = args.epochs
is_reuse = args.is_reuse
load_path = args.load_path
save_path = args.save_path
multi_gpu = args.multi_gpu
num_layer = args.num_layers


# load data
dataset = Data(pickle_dir)
print(dataset)


# load model
initial_learning_rate = l_r
total_steps = (len(dataset.files) // batch_size) * epochs  #nombre total d'etapes
warmup_steps = int(0.1 * total_steps)  # 10% du nombre total d'etapes (a ajuster)
decay_steps = total_steps - warmup_steps  # etapes de decroissance apres warmup
alpha = 0.1 #fraction de initial_learning_rate qui represente le taux d'apprentissage minimum a la fin de la decroissance. Par exemple, si alpha=0.1 et initial_learning_rate=0.01, le taux d'apprentissage final sera 0.001.
#a modifier en fonction des resultats

# Load model
learning_rate = callback.CustomSchedule(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=alpha,
    warmup_steps=warmup_steps
)
opt = Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


# define model
mt = MusicTransformerDecoder(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=num_layer,
            max_seq=max_seq,
            dropout=0.2,
            debug=False, loader_path=load_path)
mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)


# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/mt_decoder/'+current_time+'/train'
eval_log_dir = 'logs/mt_decoder/'+current_time+'/eval'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)


#define frequency of reports based on the dataset size
if pickle_dir=="/content/MusicTransformerBeethoven/dataset/preprocessed_midi": #general (and big) dataset
    freq = 50
else: #specific (and small) dataset
    freq = 5


# Train Start
idx = 0
for e in range(epochs):
    mt.reset_metrics()
    for b in range(len(dataset.files) // batch_size):
        try:
            batch_x, batch_y, sample_weights_batch = dataset.slide_seq2seq_batch(batch_size, max_seq)
        except:
            continue

        result_metrics = mt.train_on_batch(batch_x, batch_y, sample_weight=sample_weights_batch)

        if b % freq == 0:
            eval_x, eval_y, eval_sample_weights = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval')
            eval_result_metrics, weights = mt.evaluate(eval_x, eval_y, sample_weight=eval_sample_weights)

            mt.save(save_path)
            with train_summary_writer.as_default():
                if b == 0:
                    tf.summary.histogram("target_analysis", batch_y, step=e)
                    tf.summary.histogram("source_analysis", batch_x, step=e)

                tf.summary.scalar('loss', result_metrics[0], step=idx)
                tf.summary.scalar('perplexity', result_metrics[1], step=idx)
                tf.summary.scalar('accuracy', result_metrics[2], step=idx)

            with eval_summary_writer.as_default():
                if b == 0:
                    mt.sanity_check(eval_x, eval_y, step=e)

                tf.summary.scalar('loss', eval_result_metrics[0], step=idx)
                tf.summary.scalar('perplexity', eval_result_metrics[1], step=idx)
                tf.summary.scalar('accuracy', eval_result_metrics[2], step=idx)
                # for i, weight in enumerate(weights):
                #     with tf.name_scope("layer_%d" % i):
                #         with tf.name_scope("w"):
                #             utils.attention_image_summary(weight, step=idx)

            idx += 1
            print('\n====================================================')
            print('Epoch/Batch: {}/{}'.format(e, b))
            print('Train >>>> Loss: {:6.6}, Perplexity: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1], result_metrics[2]))
            print('Eval >>>> Loss: {:6.6}, Perplexity: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1], eval_result_metrics[2]))
