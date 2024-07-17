import mido
from mido import MidiFile, MidiTrack, Message
import os

def transpose_midi(input_path, output_path, semitones):
    mid = MidiFile(input_path)
    transposed_mid = MidiFile()

    for track in mid.tracks:
        transposed_track = MidiTrack()
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                # Transpose the note by the given number of semitones
                msg = msg.copy(note=min(127, max(0, msg.note + semitones)))
            transposed_track.append(msg)
        transposed_mid.tracks.append(transposed_track)

    transposed_mid.save(output_path)

# Directory containing the original MIDI files
input_directory = "C:\\Users\\rafso\\Downloads\\file_to_transpose\\"
output_directory = "C:\\Users\\rafso\\Documents\\Dauphine\\StageM1\\MusicTransformer\\dataset\\sonates_classiques\\"

# Supported MIDI file extensions
midi_extensions = ('.mid', '.midi', '.kar')

# Transpose and save each MIDI file in the directory in all keys (-11 to +11 semitones)
for filename in os.listdir(input_directory):
    if filename.lower().endswith(midi_extensions):
        input_midi_path = os.path.join(input_directory, filename)
        
        for semitones in range(-11, 12):
            if semitones == 0:
                # For 0 semitones, save the file with a "_original" suffix
                output_midi_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_original.mid")
            else:
                # For other semitones, add the transposition suffix
                output_midi_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_transposed_{semitones}.mid")
            
            transpose_midi(input_midi_path, output_midi_path, semitones)
            print(f'Saved transposed MIDI to {output_midi_path}')


