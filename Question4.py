import tensorflow as tf
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data_and_labels(kinematics_folder, meta_file):
    # Correctly read the meta file, accounting for potential extra tabs
    meta_df = pd.read_csv(meta_file, sep='\t+', engine='python', header=None, names=['file', 'skill_level'])
    
    # Function to load data from a given folder
    def load_folder_data(folder_name):
        folder_data = {}
        folder_path = Path(kinematics_folder) / folder_name
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                base_name = file_name[:-4]
                skill_level = meta_df.loc[meta_df['file'] == base_name, 'skill_level'].values[0]
                file_path = folder_path / file_name
                kinematics_df = pd.read_csv(file_path, sep=' ', header=None)
                folder_data[base_name] = {'data': kinematics_df, 'skill_level': skill_level}
        return folder_data
    
    # Load training and testing data
    training_data = load_folder_data('training')
    testing_data = load_folder_data('testing')
    
    return training_data, testing_data

from tensorflow.keras.preprocessing.sequence import pad_sequences

def extract_features_and_labels(data, max_len=1000):
    X = []
    y = []
    for entry in data.values():
        features = entry['data'].values  # Extracting feature matrix
        label = entry['skill_level']  # Extracting label
        X.append(features)
        y.append(label)
    
    # Pad sequences to ensure uniform length
    X_padded = pad_sequences(X, maxlen=max_len, dtype='float32', padding='post', truncating='post')
    
    return np.array(X_padded), np.array(y)




def main(kinematics_path, meta_file_path):
    # Load training and testing data from the specified paths
    training_data, testing_data = load_data_and_labels(kinematics_path, meta_file_path)
    print(training_data)
    print(testing_data)

    # Preparing training data
    X_train, y_train = extract_features_and_labels(training_data)

    # Preparing testing data
    X_test, y_test = extract_features_and_labels(testing_data)

    # Encode labels into a one-hot format
    encoder = LabelEncoder()
    y_train_encoded = to_categorical(encoder.fit_transform(y_train))
    y_test_encoded = to_categorical(encoder.transform(y_test))

    # Define the model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(y_train_encoded.shape[1], activation='softmax')  # Output layer nodes equal to number of classes
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test_encoded)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')




training_folder = './path_to_training_data'
testing_folder = './path_to_testing_data'

kinematics_path = './simple_jigsaws_suturing/simple_kinematics'
meta_file_path = './simple_jigsaws_suturing/simple_meta_file.txt'
main(kinematics_path, meta_file_path)
