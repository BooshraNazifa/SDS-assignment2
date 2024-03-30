import numpy as np
import pandas as pd
from scipy.special import legendre
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pathlib import Path
import os

def legendre_transform(data, order=3, channels=12):
    # Normalize time to [-1, 1]
    time_normalized = np.linspace(-1, 1, len(data))
    transformed_data = np.zeros((len(data), order + 1, channels))
    
    # Compute Legendre polynomials for each order and channel
    for i in range(order + 1):
        P = legendre(i)(time_normalized)
        for j in range(channels):
            transformed_data[:, i, j] = P * data[:, j]
            
    # Flatten the transformed data into a 48-element vector per recording
    return transformed_data.reshape(len(data), -1)

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


def transform_and_prepare_data(data_dict):
    X, y = [], []
    for key, value in data_dict.items():
        transformed_data = legendre_transform(value['data'].values)
        X.append(transformed_data.mean(axis=0))  # Example of aggregating features
        y.append(value['skill_level'])
    return np.array(X), np.array(y)

def main(training_data, testing_data):
    # Transform and prepare training and testing data
    X_train, y_train = transform_and_prepare_data(training_data)
    X_test, y_test = transform_and_prepare_data(testing_data)
    
    # Data normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train_scaled, y_train)
    
    # Evaluate the classifier
    predictions = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Classifier Accuracy: {accuracy*100:.2f}%")



kinematics_path = './simple_jigsaws_suturing/simple_kinematics'
meta_file_path = './simple_jigsaws_suturing/simple_meta_file.txt'
training_data, testing_data = load_data_and_labels(kinematics_path, meta_file_path)

main(training_data, testing_data)
