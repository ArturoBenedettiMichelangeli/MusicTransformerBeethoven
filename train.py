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
parser.add_argument('--l_r_schedule', default='Default', type=str, choices=['CosineAnnealing', 'Default', 'Fixed', 'FixedWLinearWarmup'], 
                    help="Choisir la stratégie de learning rate schedule parmi : 'CosineAnnealing', 'Default', 'Fixed', 'FixedWLinearWarmup'")
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
l_r_schedule = args.l_r_schedule
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
dataset = Data(pickle_dir, beethoven_dir="/content/MusicTransformerBeethoven/dataset/std_preprocessed_midi_Beethoven")
print(dataset)


# load model

total_steps = (len(dataset.files) // batch_size) * epochs  #nombre total d'etapes

if pickle_dir=="/content/MusicTransformerBeethoven/dataset/preprocessed_midi" or pickle_dir=="/content/MusicTransformerBeethoven/dataset/preprocessed_midi_maestro":
    warmup_steps = int(0.1 * total_steps)
else:
    warmup_steps = int(0.05 * total_steps) #less warmup steps for fine-tuning

decay_steps = total_steps - warmup_steps  # etapes de decroissance apres warmup dans la stratégie du Cosine Annealing
alpha = 0.1 #fraction de initial_learning_rate qui represente le taux d'apprentissage minimum a la fin de la decroissance. Par exemple, si alpha=0.1 et initial_learning_rate=0.01, le taux d'apprentissage final sera 0.001.
#a modifier en fonction des resultats

# Load model
if l_r_schedule=='CosineAnnealing':
    learning_rate = callback.CustomScheduleCA(
        initial_learning_rate=l_r,
        decay_steps=decay_steps,
        alpha=alpha,
        warmup_steps=warmup_steps
    )
elif l_r_schedule=='Default':
    learning_rate = callback.CustomSchedule(par.embedding_dim, warmup_steps) if l_r is None else l_r
elif l_r_schedule=='Fixed':
    learning_rate = callback.CustomScheduleCWLW(
        learning_rate=l_r,
        warmup_steps=0
    )
elif l_r_schedule=='FixedWLinearWarmup':
    learning_rate = callback.CustomScheduleCWLW(
        learning_rate=l_r,
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

# --- NEW CODE START ---

# Get a sample batch of data to build the model
try:
    # This part gets the first batch of data to explicitly build the model
    # We use a try/except block to handle cases where the dataset might be empty
    if pickle_dir == "/content/MusicTransformerBeethoven/dataset/preprocessed_midi_maestro":
        sample_batch_x, _ = dataset.slide_seq2seq_batch(batch_size, max_seq, 'train_pretraining')
    else:
        sample_batch_x, _ = dataset.slide_seq2seq_batch(batch_size, max_seq, 'train_finetuning')
    
    # Build the model by calling it on the sample data
    # This is the key step to fix the ValueError
    _ = mt(sample_batch_x)
    print("Model has been built with a sample batch of data.")

    # Now that the model is built, we can safely load the weights if a path is provided
    if load_path:
        # Load the weights from the specified checkpoint path
        mt.load_ckpt_file(load_path)
        print(f"Weights loaded successfully from {load_path}")
    
except Exception as e:
    print(f"Error during model building or weight loading: {e}")
    sys.exit() # Exit the program if there's an error here, as training cannot continue.

# --- NEW CODE END ---

mt.compile(optimizer=opt, loss=callback.transformer_dist_train_loss)


# define tensorboard writer
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_dir = 'logs/mt_decoder/'+current_time+'/train'
eval_log_dir = 'logs/mt_decoder/'+current_time+'/eval'
test_log_dir = 'logs/mt_decoder/'+current_time+'/test'

train_summary_writer = tf.summary.create_file_writer(train_log_dir)
eval_summary_writer = tf.summary.create_file_writer(eval_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)



#define frequency of reports based on the dataset size
if pickle_dir=="/content/MusicTransformerBeethoven/dataset/preprocessed_midi": #general (and big) dataset
    freq = 1000
elif (pickle_dir == "/content/MusicTransformerBeethoven/dataset/preprocessed_midi_beethoven" or
      pickle_dir == "/content/MusicTransformerBeethoven/dataset/preprocessed_First_mov" or
      pickle_dir == "/content/MusicTransformerBeethoven/dataset/std_preprocessed_midi_Beethoven"):  # specific (and small) dataset
    freq = 100
else: #maestro dataset
    freq = 1000




# Train Start (without maestro)
if pickle_dir != "/content/MusicTransformerBeethoven/dataset/preprocessed_midi_maestro":
    print("\n\nNOT MAESTRO TRAINING\n\n")
    idx = 0

    for e in range(epochs):
        mt.reset_metrics()
        for b in range(len(dataset.files) // batch_size):
            try:
                #batch_x, batch_y, sample_weights_batch = dataset.slide_seq2seq_batch_with_weights(batch_size, max_seq, 'train_finetuning')
                batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'train_finetuning')
            
            except:
                continue

            #result_metrics = mt.train_on_batch(batch_x, batch_y, sample_weight=sample_weights_batch)
            result_metrics = mt.train_on_batch(batch_x, batch_y)

            if b % freq == 0:
                #eval_x, eval_y, eval_sample_weights = dataset.slide_seq2seq_batch_with_weights(batch_size, max_seq, 'eval_finetuning')
                eval_x, eval_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval_finetuning')

                #eval_result_metrics, weights = mt.evaluate(eval_x, eval_y, sample_weight=eval_sample_weights)
                eval_result_metrics, _ = mt.evaluate(eval_x, eval_y)

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
                
                # Test set evaluation
                #test_x, test_y, test_sample_weights = dataset.slide_seq2seq_batch_with_weights(batch_size, max_seq, 'test_finetuning')
                test_x, test_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'test_finetuning')

                test_result_metrics, _ = mt.evaluate(test_x, test_y)

                # Test metrics logging
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', test_result_metrics[0], step=idx)
                    tf.summary.scalar('perplexity', test_result_metrics[1], step=idx)
                    tf.summary.scalar('accuracy', test_result_metrics[2], step=idx)

                idx += 1
                print('\n====================================================')
                print('Epoch/Batch: {}/{}'.format(e, b))
                print('Train >>>> Loss: {:6.6}, Perplexity: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1], result_metrics[2]))
                print('Eval >>>> Loss: {:6.6}, Perplexity: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1], eval_result_metrics[2]))
                print('Test  >>>> Loss: {:6.6}, Perplexity: {:6.6}, Accuracy: {}'.format(test_result_metrics[0], test_result_metrics[1], test_result_metrics[2]))

else: # maestro dataset only
    print("\n\nMAESTRO TRAINING\n\n")
    idx = 0

    for e in range(epochs):
        mt.reset_metrics()
        for b in range(len(dataset.files) // batch_size):
            try:
                batch_x, batch_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'train_pretraining')
            except:
                continue

            # Train
            result_metrics = mt.train_on_batch(batch_x, batch_y)

            if b % freq == 0:
                # Evaluation on eval set
                eval_x, eval_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'eval_pretraining')
                eval_result_metrics, weights = mt.evaluate(eval_x, eval_y)

                # Save model
                mt.save(save_path)

                # Train metrics logging
                with train_summary_writer.as_default():
                    if b == 0:
                        tf.summary.histogram("target_analysis", batch_y, step=e)
                        tf.summary.histogram("source_analysis", batch_x, step=e)

                    tf.summary.scalar('loss', result_metrics[0], step=idx)
                    tf.summary.scalar('perplexity', result_metrics[1], step=idx)
                    tf.summary.scalar('accuracy', result_metrics[2], step=idx)

                # Eval metrics logging
                with eval_summary_writer.as_default():
                    if b == 0:
                        mt.sanity_check(eval_x, eval_y, step=e)

                    tf.summary.scalar('loss', eval_result_metrics[0], step=idx)
                    tf.summary.scalar('perplexity', eval_result_metrics[1], step=idx)
                    tf.summary.scalar('accuracy', eval_result_metrics[2], step=idx)

                # Test set evaluation
                test_x, test_y = dataset.slide_seq2seq_batch(batch_size, max_seq, 'test_pretraining')
                test_result_metrics, _ = mt.evaluate(test_x, test_y)

                # Test metrics logging
                with test_summary_writer.as_default():  # Ensure you have a test summary writer initialized like train/eval
                    tf.summary.scalar('loss', test_result_metrics[0], step=idx)
                    tf.summary.scalar('perplexity', test_result_metrics[1], step=idx)
                    tf.summary.scalar('accuracy', test_result_metrics[2], step=idx)

                # Print metrics
                idx += 1
                print('\n====================================================')
                print('Epoch/Batch: {}/{}'.format(e, b))
                print('Train >>>> Loss: {:6.6}, Perplexity: {:6.6}, Accuracy: {}'.format(result_metrics[0], result_metrics[1], result_metrics[2]))
                print('Eval  >>>> Loss: {:6.6}, Perplexity: {:6.6}, Accuracy: {}'.format(eval_result_metrics[0], eval_result_metrics[1], eval_result_metrics[2]))
                print('Test  >>>> Loss: {:6.6}, Perplexity: {:6.6}, Accuracy: {}'.format(test_result_metrics[0], test_result_metrics[1], test_result_metrics[2]))
