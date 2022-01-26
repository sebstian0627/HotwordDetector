import librosa
from keras.layers import Dense, LSTM, GlobalMaxPool1D, Bidirectional
from keras.models import Sequential
import numpy as np
import os
from sklearn.utils import shuffle

KEYWORD_FOLDER = 'Help_Data/'
NEGATIVE_FOLDER = 'Negative_Data/'
OPPPOSITE_KEYWORD_FOLDER = 'Bachao_Data/'
KEYWORD_FOLDER_TEST = 'Help_Data_Test/'
NEGATIVE_FOLDER_TEST = 'Negative_Data_Test/'
OPPPOSITE_KEYWORD_FOLDER_TEST = 'Bachao_Data_Test/'

INPUT_SHAPE = (376, 40)
	
def create_model():
	print ('Creating model...')
	model = Sequential()
	model.add(Bidirectional(LSTM(units=128, input_shape=INPUT_SHAPE, return_sequences=True)))
	model.add(GlobalMaxPool1D())
	model.add(Dense(units=1,activation='sigmoid'))	

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	return model

def count_files(folder, extension):
	count = 0
	for file in os.listdir(folder):
		if file.endswith(extension):
			file_path = os.path.join(folder, file)
			count += 1
	return count

def load_data_folder(folder, is_keyword):
	num_samples = count_files(folder, '.wav')
	data_X = np.zeros((num_samples, INPUT_SHAPE[0], INPUT_SHAPE[1]), dtype=np.float64)
	data_Y = np.zeros((num_samples), dtype=np.float64)

	count = 0
	for file in os.listdir(folder):
		if file.endswith('.wav'):
			file_path = os.path.join(folder, file)
			y, sr = librosa.load(file_path,sr=None)
			mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=128, n_fft=256, n_mfcc=20)
			mfcc_delta = librosa.feature.delta(mfcc)[:10, :]
			mfcc_double_delta = librosa.feature.delta(mfcc, order=2)[:10, :]
			data_X[count, :, :20] = mfcc.T
			data_X[count, :, 20:30] = mfcc_delta.T
			data_X[count, :, 30:] = mfcc_double_delta.T
			data_Y[count] = int(is_keyword)
			count += 1

	return data_X, data_Y

def load_data(folders):
	num_samples = sum([count_files(folder, '.wav') for folder, is_keyword in folders])
	data_X = np.zeros((num_samples, INPUT_SHAPE[0], INPUT_SHAPE[1]), dtype=np.float64)
	data_Y = np.zeros((num_samples), dtype=np.float64)
	count = 0
	for folder, is_keyword in folders:
		num_samples_folder = count_files(folder, '.wav')
		data_X[count:count+num_samples_folder, :, :], data_Y[count:count+num_samples_folder] = (
			load_data_folder(folder, is_keyword))
		count += num_samples_folder
	return shuffle(data_X, data_Y, random_state=0)

def load_train_data():
	folders = [(KEYWORD_FOLDER, True), (OPPPOSITE_KEYWORD_FOLDER, False), (NEGATIVE_FOLDER, False)]
	return load_data(folders)

def load_test_data():
	folders = [(KEYWORD_FOLDER_TEST, True), (OPPPOSITE_KEYWORD_FOLDER_TEST, False), (NEGATIVE_FOLDER_TEST, False)]
	return load_data(folders)

train_X, train_Y = load_train_data()
test_X, test_Y = load_test_data()
model = create_model()

print('Training ...\n')
model.fit(train_X, train_Y, epochs=30, validation_data=(test_X, test_Y))

model.save('help_model.h5')