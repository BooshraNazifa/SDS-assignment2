import tensorflow as tf
import numpy as np
import os
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.utils import to_categorical
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

def flatten_and_pad_data(data_dict):
    X, y = [], []
    for folder_data in data_dict.values():
        for record in folder_data.values():
            try:
                # Flatten each sequence
                flat_data = record['data'].values.flatten()
                X.append(flat_data)
                y.append(record['skill_level'])
            except KeyError as e:
                print(f"Missing key in record: {e}")
                continue  # Skip this record or handle as needed
    # Pad sequences for uniform length
    X_padded = pad_sequences(X, padding='post', dtype='float')
    return np.array(X_padded), np.array(y)


def preprocess_data(X, y):
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))  # Reshape for StandardScaler
    X_scaled = X_scaled.reshape(X.shape)  # Reshape back to original shape
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    return X_scaled, y_categorical, encoder.classes_

# Define the neural network model
def build_model(input_shape, num_classes):
    model = Sequential([
        Flatten(input_shape=input_shape),  
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main(kinematics_path, meta_file_path):
    # Load training and testing data from the specified paths
    training_data, testing_data = load_data_and_labels(kinematics_path, meta_file_path)
    
    # Flatten, pad, and preprocess training data
    X_train, y_train = flatten_and_pad_data(training_data)
    X_train_preprocessed, y_train_preprocessed, classes = preprocess_data(X_train, y_train)
    
    # Flatten, pad, and preprocess testing data
    X_test, y_test = flatten_and_pad_data(testing_data)
    scaler = StandardScaler().fit(X_train.reshape(X_train.shape[0], -1))  # Fit scaler to training data
    X_test_preprocessed = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
    encoder = LabelEncoder().fit(y_train)  # Fit encoder to training labels
    y_test_encoded = encoder.transform(y_test)
    y_test_preprocessed = to_categorical(y_test_encoded)
    
    # Determine input shape for the model based on preprocessed training data
    input_shape = X_train_preprocessed.shape[1:]
    num_classes = y_train_preprocessed.shape[1]
    
    # Build the model
    model = build_model(input_shape, num_classes)
    
    # Train the model
    model.fit(X_train_preprocessed, y_train_preprocessed, epochs=10, batch_size=32, validation_split=0.2)
    
    # Evaluate the model on the preprocessed testing data
    loss, accuracy = model.evaluate(X_test_preprocessed, y_test_preprocessed)
    print(f'Test accuracy: {accuracy * 100:.2f}%')



training_folder = './path_to_training_data'
testing_folder = './path_to_testing_data'

kinematics_path = './simple_jigsaws_suturing/simple_kinematics'
meta_file_path = './simple_jigsaws_suturing/simple_meta_file.txt'
main(kinematics_path, meta_file_path)
