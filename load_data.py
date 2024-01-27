import os
import glob

def get_speaker_to_utterance(data_dir):
    """
    Returns a dictionary with speaker_id(keys) mapping to lists with utterance file path(values)
    data_sir - folder containing train data
    """

    speaker_to_utterance = dict()

    # Get all the files in data folder 
    files_list = glob.glob(os.path.join(data_dir, "*"))

    for file in files_list:
        # Get speaker's id from the file name (eg: a single train data file is named "spk_2-2_1_1_0_d5_ch5.wav")
        speaker_id = os.path.basename(file)
        speaker_id = speaker_id.split("-")[0]

        # Add file path to list
        if speaker_id not in speaker_to_utterance:
            speaker_to_utterance[speaker_id] = []
        speaker_to_utterance[speaker_id].append(file)

    return speaker_to_utterance