import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from pathlib import Path

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

def frechet_distance(data1, data2):
    # Extract translation components (first 3 columns)
    translations1, translations2 = data1.iloc[:, :3].mean(), data2.iloc[:, :3].mean()
    
    # Calculate Euclidean distance for translations using scipy's euclidean function
    translation_distance = euclidean(translations1, translations2)
    
    # For rotations, continue using numpy operations or any other method you prefer
    rotations1, rotations2 = data1.iloc[:, 3:], data2.iloc[:, 3:]
    rotation_distance = norm(rotations1.mean() - rotations2.mean())
    
    # Combine the distances
    combined_distance = translation_distance + rotation_distance
    
    return combined_distance

# def frechet_distance(data1, data2):
#     # Extract translation components (first 3 columns) assuming data1 and data2 are NumPy arrays
#     translations1, translations2 = data1[:, :3].mean(axis=0), data2[:, :3].mean(axis=0)
    
#     # Calculate Euclidean distance for translations
#     translation_distance = np.linalg.norm(translations1 - translations2)
    
#     # For rotations, assuming rotations data starts from column 3 to the end
#     rotations1, rotations2 = data1[:, 3:], data2[:, 3:]
#     rotation_distance = norm(rotations1.mean(axis=0) - rotations2.mean(axis=0))
    
#     # Combine the distances
#     combined_distance = translation_distance + rotation_distance
    
#     return combined_distance


# def normalize_features(training_data, testing_data):
#     scaler = StandardScaler()
#     # Flatten training data for scaling
#     all_training = np.vstack([v['data'] for v in training_data.values()])
#     scaler.fit(all_training)
    
#     # Apply normalization to training and testing data
#     for key in training_data:
#         training_data[key]['data'] = scaler.transform(training_data[key]['data'])
#     for key in testing_data:
#         testing_data[key]['data'] = scaler.transform(testing_data[key]['data'])
        
    return training_data, testing_data

def nearest_neighbor_classifier(test_data, train_data):
    predictions = []
    for test_key, test_value in test_data.items():
        min_distance = float('inf')
        predicted_skill_level = None
        for train_key, train_value in train_data.items():
            distance = frechet_distance(test_value['data'], train_value['data'])
            if distance < min_distance:
                min_distance = distance
                predicted_skill_level = train_value['skill_level']
        predictions.append((test_key, predicted_skill_level, test_value['skill_level']))
    return predictions

# Assuming the dataset has been extracted and paths set accordingly
kinematics_path = './simple_jigsaws_suturing/simple_kinematics'
meta_file_path = './simple_jigsaws_suturing/simple_meta_file.txt'

# Load the dataset
training_data, testing_data = load_data_and_labels(kinematics_path, meta_file_path)
# training_data, testing_data = normalize_features(training_data, testing_data)

print("Training data keys:", training_data.keys())
print("Testing data keys:", testing_data.keys())

# Evaluate the classifier
predictions = nearest_neighbor_classifier(testing_data, training_data)

# Calculate and print the accuracy
correct_predictions = sum(1 for _, predicted, actual in predictions if predicted == actual)
total_predictions = len(predictions)
accuracy = correct_predictions / total_predictions
print(f"Classifier Accuracy: {accuracy*100:.2f}%")

