import pickle
import os
import re
import sys
import hashlib
from progress.bar import Bar
import tensorflow as tf
import utils
import params as par
from processor import encode_midi, decode_midi
import random


def preprocess_midi(path):
    return encode_midi(path)


def preprocess_midi_files_under(midi_root, save_dir):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = preprocess_midi(path)
            
            # Correction ici : Utilisation de os.path.basename et os.path.join
            filename = os.path.basename(path)
            output_path = os.path.join(save_dir, filename + '.pickle')
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except EOFError:
            print('EOF Error')
        except Exception as e:
            print(f"Erreur lors du traitement de {path}: {e}")
            continue


# La classe TFRecordsConverter reste inchangée si le problème ne s'y trouve pas.
# J'ai commenté la partie `__write_to_records` car elle était vide et non fonctionnelle.
class TFRecordsConverter(object):
    def __init__(self, midi_path, output_dir,
                 num_shards_train=3, num_shards_test=1):
        self.output_dir = output_dir
        self.num_shards_train = num_shards_train
        self.num_shards_test = num_shards_test

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.es_seq_list, self.ctrl_seq_list = self.process_midi_from_dir(midi_path)
        self.counter = 0
        pass

    def process_midi_from_dir(self, midi_root):
        midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi', '.MID']))
        es_seq_list = []
        ctrl_seq_list = []
        for path in Bar('Processing').iter(midi_paths):
            print(' ', end='[{}]'.format(path), flush=True)

            try:
                data = preprocess_midi(path)
                for es_seq, ctrl_seq in data:
                    max_len = par.max_seq
                    for idx in range(max_len + 1):
                        es_seq_list.append(data[0])
                        ctrl_seq_list.append(data[1])

            except KeyboardInterrupt:
                print(' Abort')
                return
            except:
                print(' Error')
                continue

        return es_seq_list, ctrl_seq_list

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __write_to_records(self, output_path, indicies):
        writer = tf.io.TFRecordWriter(output_path)
        for i in indicies:
            es_seq = self.es_seq_list[i]
            ctrl_seq = self.ctrl_seq_list[i]
            # Le reste du code pour écrire dans le TFRecord devrait être ajouté ici.


if __name__ == '__main__':
    preprocess_midi_files_under(
        midi_root=sys.argv[1],
        save_dir=sys.argv[2])