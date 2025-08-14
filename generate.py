from model import MusicTransformer, MusicTransformerDecoderWrapper
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
parser.add_argument('--config_path', default=None)


args = parser.parse_args()

# set arguments
max_seq = args.max_seq
load_path = args.load_path
mode = args.mode
beam = args.beam
length = args.length
save_path = args.save_path
pickle_dir = args.pickle_dir
config_path= args.config_path


# Chargez le fichier de configuration correspondant aux poids
import json
config_path = f"{args.load_path}/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = tf.summary.create_file_writer(gen_log_dir)

# --- DEBUT DU CODE CORRIGÉ ---

try:
    if mode == 'enc-dec':
    # ... (laissez ce bloc si vous n'avez pas de Wrapper pour MusicTransformer) ...
        print(">> generate with original seq2seq wise... beam size is {}".format(beam))
        mt = MusicTransformer(
            embedding_dim=256,
            vocab_size=par.vocab_size,
            num_layer=6,
            max_seq=2048,
            dropout=0.2,
            debug=False)
    else:
        print(">> generate with decoder wise... beam size is {}".format(beam))
    
        # Instanciez le Wrapper en utilisant les paramètres du config.json
        mt = MusicTransformerDecoderWrapper(
            embedding_dim=config['embedding_dim'],
            vocab_size=config['vocab_size'],
            num_layer=config['num_layer'],
            max_seq=config['max_seq'],
            dropout=0.2,
            debug=False
        )

  # Construire le modèle de manière explicite (le Wrapper le gère)
    mt.build(input_shape=(None, config['max_seq']))
    print("Model has been built with the input shape.")

  # Maintenant que le modèle est construit, nous pouvons charger les poids
    if load_path:
        mt.load_weights(load_path)
        print(f"Weights loaded successfully from {load_path}")
  
except Exception as e:
  print(f"Error during model building or weight loading: {e}")
  sys.exit()

# Compile le modèle (le Wrapper passera l'appel)
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
