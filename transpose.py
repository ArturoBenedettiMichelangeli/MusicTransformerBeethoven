# import mido
# from mido import MidiFile, MidiTrack, Message
# import os

# def transpose_midi(input_path, output_path, semitones):
#     mid = MidiFile(input_path)
#     transposed_mid = MidiFile()

#     for track in mid.tracks:
#         transposed_track = MidiTrack()
#         for msg in track:
#             if msg.type in ['note_on', 'note_off']:
#                 # Transpose the note by the given number of semitones
#                 msg = msg.copy(note=min(127, max(0, msg.note + semitones)))
#             transposed_track.append(msg)
#         transposed_mid.tracks.append(transposed_track)

#     transposed_mid.save(output_path)

# # Directory containing the original MIDI files
# input_directory = "C:\\Users\\vivo-\\Documents\\Dauphine\\StageM1\\file_to_transpose"
# output_directory = "C:\\Users\\vivo-\\Documents\\Dauphine\\StageM1"

# # Supported MIDI file extensions
# midi_extensions = ('.mid', '.midi', '.kar')

# # Transpose and save each MIDI file in the directory in all keys (-11 to +11 semitones)
# for filename in os.listdir(input_directory):
#     if filename.lower().endswith(midi_extensions):
#         input_midi_path = os.path.join(input_directory, filename)
        
#         for semitones in range(-11, 12):
#             if semitones == 0:
#                 # For 0 semitones, save the file with a "_original" suffix
#                 output_midi_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_original.mid")
#             else:
#                 # For other semitones, add the transposition suffix
#                 output_midi_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_transposed_{semitones}.mid")
            
#             transpose_midi(input_midi_path, output_midi_path, semitones)
#             print(f'Saved transposed MIDI to {output_midi_path}')

import pretty_midi
import os
import glob

def transpose_midi_pretty(input_path, output_path, semitones):
    """
    Transposes a MIDI file using the pretty_midi library.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(input_path)
        for instrument in midi_data.instruments:
            # Transpose notes
            for note in instrument.notes:
                note.pitch = min(127, max(0, note.pitch + semitones))
        
        midi_data.write(output_path)
        print(f"Saved transposed MIDI to {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

# Directory containing the original MIDI files
input_directory = "C:\\Users\\vivo-\\Documents\\Dauphine\\StageM1\\maestro"
output_directory = "C:\\Users\\vivo-\\Documents\\Dauphine\\StageM1\\MAESTRO_transposed"

# Supported MIDI file extensions
midi_extensions = ('.mid', '.midi')

# Transpose and save each MIDI file in the directory
for filename in os.listdir(input_directory):
    if filename.lower().endswith(midi_extensions):
        input_midi_path = os.path.join(input_directory, filename)
        
        for semitones in range(-11, 12):
            # Create a new filename for each transposition
            output_midi_path = os.path.join(
                output_directory,
                f"{os.path.splitext(filename)[0]}_transposed_{semitones}.mid"
            )
            
            transpose_midi_pretty(input_midi_path, output_midi_path, semitones)