import pandas as pd

class Data:
    phase_sequence = [0, 1, 2, 3, 4, 5, 6]
    
    def __init__(self, observation_file, model_files):
        self.observation_vector = self.load_observation_vector(observation_file)
        self.model_matrices = self.load_model_matrices(model_files)

    @staticmethod
    def load_observation_vector(file_path):
        return pd.read_csv(file_path, header=None)

    @staticmethod
    def load_model_matrices(file_paths):
        return [pd.read_csv(file_path, header=None) for file_path in file_paths]

# Sample usage:

# Define the paths to your CSV files
observation_file_path = "path/to/your/observation_vector.csv"
model_files_paths = ["path/to/your/model1.csv", "path/to/your/model2.csv", ...]  # and so on for all 7 models

data = Data(observation_file_path, model_files_paths)
