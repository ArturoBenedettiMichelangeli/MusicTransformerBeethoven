import utils
import random
import pickle
from tensorflow.python import keras
import numpy as np
import params as par
import glob
from pathlib import Path
import os



class Data:
    # def __init__(self, dir_path):
    #     self.files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
    #     self.file_dict = {
    #         'train': self.files[:int(len(self.files) * 0.8)],
    #         'eval': self.files[int(len(self.files) * 0.8): int(len(self.files) * 0.9)],
    #         'test': self.files[int(len(self.files) * 0.9):],
    #     }
    #     self._seq_file_name_idx = 0
    #     self._seq_idx = 0
    #     self.sample_weights = self.weights_init(dir_path)
    #     pass

    def __init__(self, dir_path=None, finetuning_dir=None, mode='pretraining'):
        """
        Args:
        - dir_path (str): chemin du dataset général pour le pré-entrainement.
        - finetuning_dir (str): chemin du dataset de fine-tuning.
        - mode (str) : 'pretraining' ou 'finetuning'
        """
        self.mode = mode
        
        self.general_files = []
        if self.mode == 'pretraining' and dir_path:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Le chemin du dataset général n'existe pas : {dir_path}")
            self.general_files = list(utils.find_files_by_extensions(dir_path, ['.pickle']))
            random.shuffle(self.general_files)
        
        self.finetuning_files = []
        if finetuning_dir:
            if not os.path.exists(finetuning_dir):
                raise FileNotFoundError(f"Le chemin du dataset pour l'affinage n'existe pas : {finetuning_dir}")
            self.finetuning_files = list(utils.find_files_by_extensions(finetuning_dir, ['.pickle']))
            random.shuffle(self.finetuning_files)

        # --- Découpage des ensembles de données ---

        # Ensembles pour la phase de PRÉ-ENTRAÎNEMENT (avec les données générales)
        total_general = len(self.general_files)
        train_pre_split = int(total_general * 0.8)
        eval_pre_split = int(total_general * 0.9) # 80% train, 10% eval, 10% test

        self.file_dict = {
            'train_pretraining': self.general_files[:train_pre_split],
            'eval_pretraining': self.general_files[train_pre_split:eval_pre_split],
            'test_pretraining': self.general_files[eval_pre_split:],
        }
        
        # Ensembles pour la phase de FINE-TUNING (avec les données de Beethoven)
        if self.finetuning_files:
            total_beethoven = len(self.finetuning_files)
            train_fine_split = int(total_beethoven * 0.8)
            eval_fine_split = int(total_beethoven * 0.9) # 80% train, 10% eval, 10% test

            self.file_dict.update({
                'train_finetuning': self.finetuning_files[:train_fine_split],
                'eval_finetuning': self.finetuning_files[train_fine_split:eval_fine_split],
                'test_finetuning': self.finetuning_files[eval_fine_split:],
            })
        
        # Attributs pour les itérateurs de batch (inchangés)
        self._seq_file_name_idx = 0
        self._seq_idx = 0
        
        if self.mode == 'pretraining':
            self.files = self.file_dict['train_pretraining']
        elif self.mode == 'finetuning':
            self.files = self.file_dict['train_finetuning']
        else:
            raise ValueError("Le mode doit être 'pretraining' ou 'finetuning'.")
        
    def __repr__(self):
        return '<class Data has "'+str(len(self.files))+'" files>'

    def weights_init(self, dir_path):
        if dir_path == "/content/MusicTransformerBeethoven/dataset/preprocessed_midi":
            # Répertoire contenant les fichiers
            directory = Path("/content/MusicTransformerBeethoven/dataset/midi_transposed")

            # Initialisation de la liste pour stocker les poids
            weights = []

            # Parcours des fichiers dans le répertoire
            for filepath in directory.glob('*'):
                # Vérification si le fichier est un fichier ordinaire
                if filepath.is_file():
                    # Extraction du nom de fichier
                    filename = filepath.name
                    # Extraction du préfixe du nom de fichier
                    if filename.startswith("alb"):
                        weights.append(1)
                    elif filename.startswith("bach"):
                        weights.append(5)
                    elif filename.startswith("bartok"):
                        weights.append(1)
                    elif filename.startswith("bor"):
                        weights.append(1)
                    elif filename.startswith("br"):
                        weights.append(2)
                    elif filename.startswith("burg"):
                        weights.append(1)
                    elif filename.startswith("ch"):
                        weights.append(2)
                    elif filename.startswith("clementi"):
                        weights.append(10)
                    elif filename.startswith("gra"):
                        weights.append(1)
                    elif filename.startswith("grieg"):
                        weights.append(1) 
                    elif filename.startswith("haendel"):
                        weights.append(5)
                    elif filename.startswith("hay"):
                        weights.append(20)
                    elif filename.startswith("hummel"):
                        weights.append(20) 
                    elif filename.startswith("li"):
                        weights.append(2)
                    elif filename.startswith("mendel"):
                        weights.append(2)
                    elif filename.startswith("mos"):
                        weights.append(1)
                    elif filename.startswith("moz") or filename.startswith("mz"):
                        weights.append(10)
                    elif filename.startswith("muss"):
                        weights.append(1)
                    elif filename.startswith("rac"):
                        weights.append(1)
                    elif filename.startswith("ravel"):
                        weights.append(1)
                    elif filename.startswith("satie"):
                        weights.append(1)
                    elif filename.startswith("schn") or filename.startswith("schum") or filename.startswith("scn"):
                        weights.append(2)
                    elif filename.startswith("schub"):
                        weights.append(20)        
                    elif filename.startswith("scriabine"):
                        weights.append(1)
                    elif filename.startswith("ty"):
                        weights.append(1)                     
                    # Ajouter d'autres conditions pour d'autres préfixes si nécessaire
                    else:
                        # Par défaut, ajouter 0 si aucun préfixe correspondant n'est trouvé
                        weights.append(0)
            return weights
        elif dir_path == "/content/MusicTransformerBeethoven/dataset/preprocessed_midi_beethoven" or dir_path == "/content/MusicTransformerBeethoven/dataset/preprocessed_First_mov":
            weights = [1 for _ in range(23*10)] + [2 for _ in range(23*10)] + [3 for _ in range(23*12)]
            return weights
        else:
            return None

    def batch(self, batch_size, length, mode):

        batch_files = random.sample(self.file_dict[mode], k=batch_size)

        batch_data = [
            self._get_seq(file, length)
            for file in batch_files
        ]
        return np.array(batch_data), batch_files  # batch_size, seq_len

    def seq2seq_batch(self, batch_size, length, mode):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length]
        y = data[:, length:]
        return x, y

    def smallest_encoder_batch(self, batch_size, length, mode):
        data = self.batch(batch_size, length * 2, mode)
        x = data[:, :length//100]
        y = data[:, length//100:length//100+length]
        return x, y

    def slide_seq2seq_batch(self, batch_size, length, mode):
        data, _ = self.batch(batch_size, length+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        return x, y

    def slide_seq2seq_batch_with_weights(self, batch_size, length, mode):
        data, batch_files = self.batch(batch_size, length+1, mode)
        x = data[:, :-1]
        y = data[:, 1:]
        
        # Créer un dictionnaire pour faire correspondre chaque fichier à son poids d'échantillon
        file_to_weight = {file: self.sample_weights[self.files.index(file)] for file in self.files}
        
        # Utiliser ce dictionnaire pour récupérer les poids d'échantillon pour les fichiers dans le batch actuel
        sample_weights_batch = np.array([file_to_weight[file] for file in batch_files])
        
        return x, y, sample_weights_batch
    
    def random_sequential_batch(self, batch_size, length):
        batch_files = random.sample(self.files, k=batch_size)
        batch_data = []
        for i in range(batch_size):
            data = self._get_seq(batch_files[i])
            for j in range(len(data) - length):
                batch_data.append(data[j:j+length])
                if len(batch_data) == batch_size:
                    return batch_data

    def sequential_batch(self, batch_size, length):
        batch_data = []
        data = self._get_seq(self.files[self._seq_file_name_idx])

        while len(batch_data) < batch_size:
            while self._seq_idx < len(data) - length:
                batch_data.append(data[self._seq_idx: self._seq_idx + length])
                self._seq_idx += 1
                if len(batch_data) == batch_size:
                    return batch_data

            self._seq_idx = 0
            self._seq_file_name_idx = self._seq_file_name_idx + 1
            if self._seq_file_name_idx == len(self.files):
                self._seq_file_name_idx = 0
                print('iter intialized')

    def _get_seq(self, fname, max_length=None):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        if max_length is not None:
            if max_length <= len(data):
                start = random.randrange(0,len(data) - max_length)
                data = data[start:start + max_length]
            else:
                data = np.append(data, par.token_eos)
                while len(data) < max_length:
                    data = np.append(data, par.pad_token)
        return data


class PositionalY:
    def __init__(self, data, idx):
        self.data = data
        self.idx = idx

    def position(self):
        return self.idx

    def data(self):
        return self.data

    def __repr__(self):
        return '<Label located in {} position.>'.format(self.idx)


def add_noise(inputs: np.array, rate:float = 0.01): # input's dim is 2
    seq_length = np.shape(inputs)[-1]

    num_mask = int(rate * seq_length)
    for inp in inputs:
        rand_idx = random.sample(range(seq_length), num_mask)
        inp[rand_idx] = random.randrange(0, par.pad_token)

    return inputs


if __name__ == '__main__':
    import pprint
    def count_dict(max_length, data):
        cnt_arr = [0] * max_length
        cnt_dict = {}
        # print(cnt_arr)
        for batch in data:
            for index in batch:
                try:
                    cnt_arr[int(index)] += 1

                except:
                    print(index)
                try:
                    cnt_dict['index-'+str(index)] += 1
                except KeyError:
                    cnt_dict['index-'+str(index)] = 1
        return cnt_arr

    # print(add_noise(np.array([[1,2,3,3,4,5,6]]), rate=0.2))


    # print(par.vocab_size)
    # data = Data('dataset/processed')
    # # ds = DataSequence('dataset/processed', 10, 2048)
    # sample = data.seq2seq_batch(1000, 100)[0]
    # pprint.pprint(list(sample))
    # arr = count_dict(par.vocab_size+3,sample)
    # pprint.pprint(
    #     arr)
    #
    # from sequence import EventSeq, Event
    #
    # event_cnt = {
    #     'note_on': 0,
    #     'note_off': 0,
    #     'velocity': 0,
    #     'time_shift': 0
    # }
    # for event_index in range(len(arr)):
    #     for event_type, feat_range in EventSeq.feat_ranges().items():
    #
    #         if feat_range.start <= event_index < feat_range.stop:
    #             print(event_type+':'+str(arr[event_index])+' event cnt: '+str(event_cnt))
    #             event_cnt[event_type] += arr[event_index]
    #
    # print(event_cnt)

    # print(np.max(sample), np.min(sample))
    # print([data._get_seq(file).shape for file in data.files])
    #while True:
    # print(ds.__getitem__(10)[1].argmax(-1))
