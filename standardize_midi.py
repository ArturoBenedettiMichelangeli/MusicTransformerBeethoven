import py_midicsv
import tempfile
import os
import subprocess
import glob

def standardize_midi(tmp_csv_file_path, output_midi_path=None):
    try:
        # Load the CSV data from the provided file
        with open(tmp_csv_file_path, 'r') as f:
            csv_data = f.readlines()

        # Initialize a list to hold the standardized CSV lines
        standardized_data = []
        tempo_found = False
        unified_track_data = []

        # Separate initial lines (0 and 1 as first column) from the instrument data
        for line in csv_data:
            if line.startswith("0,") or line.startswith("1,"):
                standardized_data.append(line)
            elif "MIDI_port" in line:
                # Skip MIDI_port lines
                continue
            else:
                # Only process Tempo line if it's not yet found and within track 1 section
                if "Tempo" in line and not tempo_found and line.startswith("1,"):
                    standardized_data.append(line)
                    tempo_found = True
                else:
                    # Collect instrument lines for further processing
                    unified_track_data.append(line)

        # Update the unified track data to set the first column to 2 and channel to 0
        updated_track_data = []
        for line in unified_track_data:
            parts = line.split(",")
            parts[0] = "2"  # Set track number to 2 for unified track
            if "Control_c" in parts[2] or "Note_on_c" in parts[2] or "Note_off_c" in parts[2] or "Program_c" in parts[2]:
                parts[3] = "0"  # Set channel to 0 for unified track
            updated_track_data.append(",".join(parts))

        # Sort by time (second column) and remove extra "End_track" lines, keeping only one
        sorted_track_data = sorted(
            updated_track_data, key=lambda x: int(x.split(",")[1].strip())
        )
        sorted_track_data = [line for line in sorted_track_data if "End_track" not in line]

        # Add a single "End_track" at the end of the unified track
        if sorted_track_data:
            last_event = sorted_track_data[-1]
            last_parts = last_event.split(",")
            last_time = last_parts[1].strip()  # Get the time value from the last event
            sorted_track_data.append(f"2, {last_time}, End_track")

        # Preserve the original "End_of_file" line at the end
        end_of_file = next((line for line in csv_data if "End_of_file" in line), None)
        if end_of_file:
            sorted_track_data.append(end_of_file)

        # Combine all data back into one list
        result_data = standardized_data + sorted_track_data

        # Apply post-processing instructions
        final_data = []
        seen_control_events = set()
        start_track_seen = False

        for line in result_data:
            if "End_of_file" in line:
                continue  # We'll handle End_of_file at the end

            if line.startswith("2,") and "Control_c" in line:
                # Remove duplicate Control_c lines
                if line not in seen_control_events:
                    final_data.append(line)
                    seen_control_events.add(line)
            elif line.startswith("2,") and "Start_track" in line:
                # Only keep the first Start_track line
                if not start_track_seen:
                    final_data.append(line)
                    start_track_seen = True
            elif not ("Title_t" in line or "Copyright_t" in line):
                final_data.append(line)

        # Add back the final End_of_file line
        if end_of_file:
            final_data.append(end_of_file)

        # Write to output file if an output path is provided
        if output_midi_path:
            with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmpfile:
                tmpfile.write("\n".join(final_data) + "\n")  # Ensure each line is separated
                tmpfile_path = tmpfile.name

            # Convert temporary CSV file back to a MIDI object
            midi_object = py_midicsv.csv_to_midi(tmpfile_path)

            # Write the MIDI object to the specified output path
            with open(output_midi_path, "wb") as output_file:
                midi_writer = py_midicsv.FileWriter(output_file)
                midi_writer.write(midi_object)

            os.remove(tmpfile_path)
            print(f"Standardized MIDI file saved as '{output_midi_path}'")
        
        return final_data

    except Exception as e:
        print(f"Error processing MIDI file: {e}")

if __name__ == "__main__":
    # Directories
    midi_directory = "/home/aldo/aDauphine/MusicTransformerBeethoven/dataset/midi_Beethoven"
    standardized_midi_directory = "/home/aldo/aDauphine/MusicTransformerBeethoven/dataset/std_midi_Beethoven"
    standardized_csv_directory = "/home/aldo/aDauphine/MusicTransformerBeethoven/dataset/std_csv"

    # Create output directories if they do not exist
    os.makedirs(standardized_midi_directory, exist_ok=True)
    os.makedirs(standardized_csv_directory, exist_ok=True)

    # Process each MIDI file in the specified directory
    for midi_file_path in glob.glob(os.path.join(midi_directory, "*.mid")):
        # Temporary CSV file path
        tmp_csv_file_path = midi_file_path.replace(".mid", ".csv")
        
        # Output paths for standardized files
        output_midi_path = os.path.join(standardized_midi_directory, os.path.basename(midi_file_path).replace(".mid", "_standardized.mid"))
        output_csv_path = os.path.join(standardized_csv_directory, os.path.basename(midi_file_path).replace(".mid", "_standardized.csv"))

        # Convert MIDI to CSV using midicsvpy
        try:
            subprocess.run(["midicsvpy", midi_file_path, "-n", tmp_csv_file_path], check=True)  # option -n to ignore parsing error
            
            # Standardize the MIDI from CSV
            std_csv = standardize_midi(tmp_csv_file_path, output_midi_path)
            print(f"Standardized MIDI for '{midi_file_path}' saved as '{output_midi_path}'.")

            # Convert the standardized MIDI into CSV to check conformity
            subprocess.run(["midicsvpy", "-n", output_midi_path, output_csv_path], check=True)  # option -n to ignore parsing error
            print(f"Converted '{output_midi_path}' to '{output_csv_path}'.")

        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing '{midi_file_path}': {e}")

        finally:
            # Remove the temporary CSV file after processing
            if os.path.exists(tmp_csv_file_path):
                os.remove(tmp_csv_file_path)
                print(f"Temporary CSV file '{tmp_csv_file_path}' has been deleted.")


# if __name__ == "__main__":
#     # Replace with the path to your MIDI file and desired output path
#     midi_file_path = '/home/aldo/aDauphine/MusicTransformerBeethoven/dataset/midi_Beethoven/sonate_32_chisamori.mid'
#     output_midi_path = "/home/aldo/standardized_sonate_32.mid"
    
#     # Get the standardized CSV data
#     csv = standardize_midi(midi_file_path, output_midi_path=output_midi_path)
    
#     # # Print the first 100 lines of the CSV data
#     # for line in csv[-100:]:
#     #     print(line)

#     # import py_midicsv as pm
#     # import os

#     # # Define the dummy CSV data
#     # dummy_data = [
#     #     "0, 0, Header, 1, 2, 480",
#     #     "1, 0, Start_track",
#     #     "1, 0, Tempo, 500000",
#     #     "1, 0, Time_signature, 4, 2, 24",
#     #     "1, 1000, End_track",
#     #     "2, 0, Start_track",
#     #     "2, 0, Program_c, 0, 0",
#     #     "2, 0, Note_on_c, 0, 60, 90",
#     #     "2, 500, Note_off_c, 0, 60, 0",
#     #     "2, 1000, End_track",
#     #     "0, 0, End_of_file"
#     # ]

#     # # Convert to CSV string format
#     # csv_data = [line + "\n" for line in dummy_data]

#     # # Output path for the new MIDI file
#     # output_path = "/home/aldo/test_output.mid"

#     # # Ensure output directory exists
#     # output_dir = os.path.dirname(output_path)
#     # if not os.path.exists(output_dir):
#     #     os.makedirs(output_dir)

#     # # Convert CSV data to a MIDI object
#     # try:
#     #     midi_object = pm.csv_to_midi(csv_data)
#     #     print("MIDI object created successfully.")

#     #     # Save the MIDI object to a file
#     #     with open(output_path, "wb") as output_file:
#     #         midi_writer = pm.FileWriter(output_file)
#     #         midi_writer.write(midi_object)
#     #     print(f"Test MIDI file written to: {output_path}")

#     # except Exception as e:
#     #     print(f"An error occurred: {e}")

#     # # Verify if the file was created
#     # if os.path.exists(output_path):
#     #     print("The MIDI file exists.")
#     # else:
#     #     print("The MIDI file does not exist; something went wrong.")
