# Keyword Spotting

Speech keyword detection using an LSTM model

## Preprocessing

For training, the model expects audio files to be
* in WAV format
* of 16000 Hz sampling rate
* Mono
* of 3 second duration
* having a single utterance of keyword

To convert recorded mp3 audio to WAV, preprocess_mp3.sh script can be used. For mp3 files shorter than 3 seeconds, it adds silence at the end and for files longer than 3 seconds, it trims the end.

```bash
$./preprocess_mp3.sh mp3Folder wavFolder
```
Note - preprocess_mp3.sh deletes all previous files in wavFolder, so exercise caution.

## Input Features

The model uses MFCCs(20 for each window) and its deltas and double-deltas(10 for each window), making a total of 40 features for each window.

The parameters for MFCC are 
* FFT size = 256 samples (also the window size)
* Hop Length = 128 samples (window shifts by this number)

Note - On changing MFCC parameters, variable INPUT_SHAPE in train_lstm.py will also need to be changed.

## Architecture

The model is a bidirectional LSTM with 128 hidden units. It is followed by a global max pooling layer and then a final dense layer with one ouput neuron.

## Training

Specify the training and testing formatted WAV audio folders at the root of train_lstm.py in variables KEYWORD_FOLDER, KEYWORD_FOLDER_TEST, etc. 

Negative data is also needed for which audio from other keywords and scraped Youtube videos is used.

```bash
$python3 train_lstm.py
```

## Data

Audio data for keywords "Help" and "Bachao" is available, containing roughly 900 samples for each. "Raw" folder contains mp3 files as they were recorded. "WAV Formatted" contains WAV files in an appropriate format, ready to fed into the model and also split into train and test set.

Download Link - https://drive.google.com/open?id=1H7cAovrYCK5t5s3GpoZl5hBKgnkPJjPj

## Deployment

Before deploying model to device, it needs to be converted from .h5 format to .pb format using [keras_to_tensorflow](https://github.com/amir-abdi/keras_to_tensorflow). 

```bash
python keras_to_tensorflow.py 
    --input_model="path/to/keras/model.h5" 
    --output_model="path/to/save/model.pb"
```

The .pb file can then used to classify using Tensorflow for Andrdoid through TensorFlowInferenceInterface class.

## Prerequisites

* Python 3
* Sox (for preprocess_mp3.sh)
* GNU coreutils (for macOS users)

#### Python Libraries 

* Librosa (for MFCC extraction)
* Keras 
* Tensorflow
* Numpy
* Scikit-learn

The programs have been tested to work in macOS. Some minor changes need to be made to make it work in Ubuntu. One is change  call of "gcp" in preprocess_mp3.sh to "cp".


