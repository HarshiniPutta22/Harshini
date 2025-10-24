import soundfile as sf
import resampy
import numpy as np

# Load the audio file
audio_path = '/data/dialog.wav'
audio_data, sampling_rate = sf.read(audio_path)

# Target sampling rate for pyannote.audio
target_sampling_rate = 16000

# Check and resample if necessary
if sampling_rate != target_sampling_rate:
    audio_data = resampy.resample(audio_data, sampling_rate, target_sampling_rate)
    sampling_rate = target_sampling_rate

# Store the processed audio data and sampling rate
processed_audio_data = audio_data
processed_sampling_rate = sampling_rate

print(f"Processed audio data shape: {processed_audio_data.shape}")
print(f"Processed sampling rate: {processed_sampling_rate}")

"""**Reasoning**:
The file path `/data/dialog.wav` seems incorrect. I need to check the available files and use the correct path for the audio file.


"""

import os

print(os.listdir('/content'))

"""**Reasoning**:
The previous command showed that 'audio.mp3' is in the '/content' directory. However, the subtask requires a '.wav' file and the available files list shows only 'audio.mp3'. I should use the provided '.mp3' file and convert it to '.wav' before proceeding with diarization, as pyannote.audio typically works with WAV files.


"""

from pydub import AudioSegment
import soundfile as sf
import resampy
import numpy as np

# Define the input and output paths
mp3_audio_path = '/content/audio.mp3'
wav_audio_path = '/content/audio.wav'

# Convert MP3 to WAV
audio = AudioSegment.from_mp3(mp3_audio_path)
audio.export(wav_audio_path, format="wav")

# Load the converted WAV audio file
audio_data, sampling_rate = sf.read(wav_audio_path)

# Target sampling rate for pyannote.audio
target_sampling_rate = 16000

# Check and resample if necessary
if sampling_rate != target_sampling_rate:
    audio_data = resampy.resample(audio_data, sampling_rate, target_sampling_rate)
    sampling_rate = target_sampling_rate

# Store the processed audio data and sampling rate
processed_audio_data = audio_data
processed_sampling_rate = sampling_rate

print(f"Processed audio data shape: {processed_audio_data.shape}")
print(f"Processed sampling rate: {processed_sampling_rate}")

"""## Perform speaker diarization

### Subtask:
Use `pyannote.audio` to identify speaker turns and create segments with speaker labels.

**Reasoning**:
Import the Pipeline class and instantiate it for speaker diarization, then apply it to the processed audio data and print the result.
"""
from pyannote.audio import Pipeline

# Instantiate a Pipeline object for speaker diarization with authentication
# Replace YOUR_HF_TOKEN with your actual Hugging Face authentication token.
# You can generate a token from your Hugging Face account settings.
# Visit https://hf.co/settings/tokens to create a token.
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", token="YOUR_HF_TOKEN")

# Apply the instantiated pipeline to the processed audio data
diarization = pipeline({"waveform": processed_audio_data, "sample_rate": processed_sampling_rate})

# Print the diarization object to inspect the speaker segments and labels
print(diarization)

"""## Integrate with stt output

### Subtask:
Sync the speaker segments with the existing Speech-to-Text (STT) output to create a diarized transcript with timestamps.

**Reasoning**:
Define a variable `stt_output` and assign to it a list of dictionaries, where each dictionary represents a word from the STT output and contains 'word', 'start_time', and 'end_time' keys. These timestamps should be in seconds.
"""

stt_output = [
    {'word': 'This', 'start_time': 0.1, 'end_time': 0.5},
    {'word': 'is', 'start_time': 0.6, 'end_time': 0.8},
    {'word': 'a', 'start_time': 0.9, 'end_time': 1.0},
    {'word': 'test', 'start_time': 1.1, 'end_time': 1.5},
    {'word': 'sentence', 'start_time': 1.6, 'end_time': 2.2},
    {'word': 'from', 'start_time': 2.3, 'end_time': 2.6},
    {'word': 'the', 'start_time': 2.7, 'end_time': 2.9},
    {'word': 'STT', 'start_time': 3.0, 'end_time': 3.3},
    {'word': 'output.', 'start_time': 3.4, 'end_time': 4.0},
    {'word': 'Another', 'start_time': 5.1, 'end_time': 5.5},
    {'word': 'sentence', 'start_time': 5.6, 'end_time': 6.2},
    {'word': 'for', 'start_time': 6.3, 'end_time': 6.6},
    {'word': 'diarization.', 'start_time': 6.7, 'end_time': 7.5},
    {'word': 'And', 'start_time': 8.0, 'end_time': 8.3},
    {'word': 'finally', 'start_time': 8.4, 'end_time': 8.9},
    {'word': 'a', 'start_time': 9.0, 'end_time': 9.1},
    {'word': 'third', 'start_time': 9.2, 'end_time': 9.5},
    {'word': 'one.', 'start_time': 9.6, 'end_time': 10.0}
]

"""**Reasoning**:
Iterate through the diarization object and the stt_output list to find words within each segment and assign the speaker label to the corresponding word dictionaries. Store the result in `diarized_stt_output`.


"""

diarized_stt_output = []
stt_index = 0

for segment, _, speaker in diarization.itertracks(yield_label=True):
    segment_start = segment.start
    segment_end = segment.end

    while stt_index < len(stt_output):
        word_info = stt_output[stt_index]
        word_start = word_info['start_time']
        word_end = word_info['end_time']

        # Check if the word falls within the current speaker segment
        # A word is considered within the segment if its start or end time is within the segment,
        # or if the segment is entirely within the word's duration.
        if (word_start >= segment_start and word_start < segment_end) or \
           (word_end > segment_start and word_end <= segment_end) or \
           (segment_start >= word_start and segment_end <= word_end):

            word_info['speaker'] = speaker
            diarized_stt_output.append(word_info)
            stt_index += 1
        elif word_start >= segment_end:
            # The word is after the current segment, move to the next segment
            break
        else:
            # The word is before the current segment, this should not happen
            # if we process segments and words in chronological order.
            # However, as a safeguard, we'll still advance the word index.
            stt_index += 1

# Handle any remaining words in stt_output that were not assigned to a speaker segment
# This could happen if there are silences at the end not covered by diarization
while stt_index < len(stt_output):
    diarized_stt_output.append(stt_output[stt_index])
    stt_index += 1

display(diarized_stt_output)

"""## Integrate with stt output retry 1

### Subtask:
Retry syncing the speaker segments with the existing Speech-to-Text (STT) output to create a diarized transcript with timestamps. This retry addresses the issue where the `diarization` object was not available in the previous attempt.

**Reasoning**:
Initialize the necessary variables and iterate through the diarization object and STT output to sync them based on timestamps.
"""

diarized_stt_output = []
stt_index = 0

for segment, _, speaker in diarization.itertracks(yield_label=True):
    segment_start = segment.start
    segment_end = segment.end

    while stt_index < len(stt_output):
        word_info = stt_output[stt_index]
        word_start = word_info['start_time']
        word_end = word_info['end_time']

        # Check if the word falls within the current speaker segment
        # A word is considered within the segment if its start or end time is within the segment,
        # or if the segment is entirely within the word's duration.
        if (word_start >= segment_start and word_start < segment_end) or \
           (word_end > segment_start and word_end <= segment_end) or \
           (segment_start >= word_start and segment_end <= word_end):

            word_info['speaker'] = speaker
            diarized_stt_output.append(word_info)
            stt_index += 1
        elif word_start >= segment_end:
            # The word is after the current segment, move to the next segment
            break
        else:
            # The word is before the current segment, this should not happen
            # if we process segments and words in chronological order.
            # However, as a safeguard, we'll still advance the word index.
            stt_index += 1

# Handle any remaining words in stt_output that were not assigned to a speaker segment
# This could happen if there are silences at the end not covered by diarization
while stt_index < len(stt_output):
    diarized_stt_output.append(stt_output[stt_index])
    stt_index += 1

display(diarized_stt_output)

formatted_transcript = []
current_speaker = None

for word_info in diarized_stt_output:
    speaker = word_info.get('speaker')
    word = word_info.get('word')

    if speaker is not None and speaker != current_speaker:
        formatted_transcript.append(f"[{speaker}]:")
        current_speaker = speaker

    if word is not None:
        formatted_transcript.append(word)

formatted_transcript_string = " ".join(formatted_transcript)
print(formatted_transcript_string)

# 1. Acknowledge that direct DER calculation using the AMI corpus is not feasible
# within this environment without the actual reference diarization data.
print("Direct calculation of Diarization Error Rate (DER) using the AMI corpus sample is not feasible within this environment.")
print("This is because we do not have access to the ground truth reference diarization data for these specific AMI corpus samples.")
print("-" * 30)

# 2. Explain that evaluating diarization accuracy would typically involve comparing
# the generated `diarization` object against a ground truth reference diarization
# using metrics like DER.
print("Evaluating the accuracy of speaker diarization typically involves comparing the output of the diarization system (the 'diarization' object)")
print("against a human-annotated ground truth reference diarization.")
print("Metrics such as Diarization Error Rate (DER) are used for this comparison.")
print("-" * 30)

# 3. State that due to the failure in obtaining the `diarization` object in the
# previous steps, this evaluation step cannot be fully executed as intended.
print("However, in the previous steps, we were unable to successfully obtain the 'diarization' object due to issues with accessing the pre-trained model.")
print("Therefore, this evaluation step, which relies on the 'diarization' object, cannot be fully executed as intended.")
print("-" * 30)

# 4. Mention that if the `diarization` object were available, libraries like
# `pyannote.metrics` could be used to compute the DER.
print("If the 'diarization' object were available, libraries specifically designed for evaluating diarization, such as 'pyannote.metrics',")
print("could be used to compute metrics like the Diarization Error Rate (DER) by comparing the generated diarization with a reference.")
print("For example, the `DiarizationErrorRate` class from `pyannote.metrics.diarization` would be used.")
print("-" * 30)

# 5. Conclude by summarizing that while the code for evaluation could be outlined,
# the actual calculation and assessment cannot be performed due to the missing
# `diarization` output from the previous steps.
print("In summary, while the conceptual approach and the necessary tools for evaluating diarization accuracy (like 'pyannote.metrics') exist and could be outlined in code,")
print("the actual calculation and assessment of Diarization Error Rate (DER) cannot be performed at this time.")
print("This is directly due to the failure in generating the required 'diarization' output in the preceding steps.")

# Although we cannot compute DER, here is an outline of how it would be done
# if the diarization object and a reference were available:
# from pyannote.metrics.diarization import DiarizationErrorRate
#
# # Assuming 'diarization' is the output from pyannote.audio and 'reference_diarization' is the ground truth
# # diarization_metric = DiarizationErrorRate()
# # der = diarization_metric(reference_diarization, diarization)
# # print(f"Diarization Error Rate (DER): {der:.2f}")

from pyannote.audio import Pipeline
import os

def perform_diarization(audio_data, sampling_rate, hf_token):
    """
    Performs speaker diarization on the provided audio data.

    Args:
        audio_data (np.ndarray): The processed audio data as a numpy array.
        sampling_rate (int): The sampling rate of the audio data.
        hf_token (str): Your Hugging Face authentication token.

    Returns:
        pyannote.core.Annotation: The diarization result.
    """
    # Instantiate the pyannote.audio.Pipeline for speaker diarization
    # This requires setting the HF_TOKEN environment variable with your Hugging Face token
    # or passing it directly.
    # Visit https://hf.co/settings/tokens to create a token.
    try:
        # Set the HF_TOKEN environment variable for the pipeline to pick it up
        # This is an alternative to passing it directly to from_pretrained,
        # which might be necessary depending on the pyannote.audio version.
        # os.environ['HF_TOKEN'] = hf_token

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=hf_token)

        # Apply the instantiated pipeline to the input audio data and sampling rate
        diarization_result = pipeline({"waveform": audio_data, "sample_rate": sampling_rate})

        return diarization_result
    except Exception as e:
        print(f"An error occurred during diarization: {e}")
        return None



