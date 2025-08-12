from model import MusicTransformer, MusicTransformerDecoder
from custom.layers import *
from custom import callback
import params as par
from tensorflow.keras.optimizers import Adam
from data import Data
import utils
import datetime
import argparse
from processor import decode_midi, encode_midi
import tensorflow as tf
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument('--max_seq', default=2048, help='최대 길이', type=int)
parser.add_argument('--load_path', default="result/dec0722", help='모델 로드 경로', type=str)
parser.add_argument('--mode', default='dec')
parser.add_argument('--beam', default=None, type=int)
parser.add_argument('--length', default=2048, type=int)
parser.add_argument('--save_path', default='/content/generated.mid', type=str)
parser.add_argument('--pickle_dir', default='/content/MusicTransformerBeethoven/dataset/preprocessed_std_beethoven_transposed', help='Chemin du répertoire pickle', type=str)


args = parser.parse_args()

# set arguments
max_seq = args.max_seq
load_path = args.load_path
mode = args.mode
beam = args.beam
length = args.length
save_path = args.save_path
pickle_dir = args.pickle_dir

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)

# --- DEBUT DU CODE CORRIGÉ ---

try:
    if mode == 'enc-dec':
        print(">> generate with original seq2seq wise... beam size is {}".format(beam))
        mt = MusicTransformer(
                embedding_dim=256,
                vocab_size=par.vocab_size,
                num_layer=6,
                max_seq=2048,
                dropout=0.4,
                debug=False)
    
    else:
        print(">> generate with decoder wise... beam size is {}".format(beam))
        mt = MusicTransformerDecoder()

    # Build the model by calling the build() method with the correct input shape
    mt.build(input_shape=(None, par.max_seq))
    print("Model has been built with the input shape.")

    # Now that the model is built, we can safely load the weights
    if load_path:
        mt.load_ckpt_file(load_path)
        print(f"Weights loaded successfully from {load_path}")
    
except Exception as e:
    print(f"Error during model building or weight loading: {e}")
    sys.exit()

# Compile the model
mt.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=par.l_r))

# --- FIN DU CODE CORRIGÉ ---

inputs = encode_midi('/content/MusicTransformerBeethoven/dataset/std_beethoven/sonate_01_chisamori_standardized.mid') 

with gen_summary_writer.as_default():
    result = mt.generate(inputs[:200], beam=beam, length=length, tf_board=True)

for i in result:
    print(i)

if mode == 'enc-dec':
    decode_midi(list(inputs[-1*par.max_seq:]) + list(result[1:]), file_path=save_path)
else:
    decode_midi(result, file_path=save_path)
