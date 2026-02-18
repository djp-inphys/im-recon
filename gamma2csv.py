import os
import struct
from typing import List
import pandas as pd
import numpy as np

class Data:
    phase_sequence: List[int] = [0, 1, 2, 3, 4, 5, 6]
    num_bins: int = 600
    num_channels: int = 64

    def __init__(self, index_file: str, gamma_file: str) -> None:
        self.gamma_index = self.load_gamma_index(index_file)
        self.gamma_file = gamma_file
        self.obs = self.read_gamma_obs()

    @staticmethod
    def load_gamma_index(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def read_gamma_obs(self) -> pd.DataFrame:
        return read_gamma_obs(self, self.gamma_file)


def read_gamma_obs(data: Data, gamma_file_name: str) -> pd.DataFrame:
    obs_data_list: List[np.ndarray] = []
    phase_list: List[int] = []
    channel_list: List[int] = []
    gamma_index_data: pd.DataFrame = data.gamma_index
    num_bins: int = data.num_bins
    num_channels: int = data.num_channels

    with open(gamma_file_name, 'rb') as file:
        for i, row in gamma_index_data.iterrows():
            file_position: int = int(row['filePosition'])
            phase: int = int(row['phase'])

            if i < len(gamma_index_data) - 1:
                next_file_position: int = int(gamma_index_data.iloc[i + 1]['filePosition'])
                obs_size: int = (next_file_position - file_position) // 4
            else:
                file.seek(0, os.SEEK_END)
                end_of_file: int = file.tell()
                obs_size: int = (end_of_file - file_position) // 4

            file.seek(file_position)
            obs = [struct.unpack('i', file.read(4))[0] for _ in range(obs_size)]

            if len(obs) < num_channels * num_bins:
                obs += [0] * (num_channels * num_bins - len(obs))

            if len(obs) == num_channels * num_bins:
                obs_array: np.ndarray = np.array(obs).reshape(num_channels, num_bins)
                obs_data_list.append(obs_array)
                phase_list.append(phase)
                channel_list.extend(range(num_channels))

    # Flatten the list of numpy arrays to create a 2D array
    flat_obs_data = np.vstack(obs_data_list).reshape(-1, num_bins)

    # Create a DataFrame from the 2D array with specified bin column names
    obs_df = pd.DataFrame(flat_obs_data, columns=[f'bin_{i}' for i in range(num_bins)])

    # Add 'phase' as the first column
    obs_df.insert( 0, 'phase', np.repeat(phase_list, num_channels) )
 
    # Insert 'channel' as the first column
    obs_df.insert(0, 'channel', channel_list)
    
    return obs_df

def get_nth_obs(df: pd.DataFrame, phase: int, instance_n: int, num_channels: int) -> pd.DataFrame:
    # Filter the DataFrame for the specified phase
    phase_df = df[df['phase'] == phase]
    
    # Calculate the starting and ending indices for the nth instance
    start_index = instance_n * num_channels
    end_index = (instance_n + 1) * num_channels

    # Check if the start index is within the bounds of the DataFrame
    if start_index >= len(phase_df):
        # If the instance number is too great, return an empty DataFrame
        return pd.DataFrame()
    
    # Select the nth instance of observation data
    nth_observation = phase_df.iloc[start_index:end_index]
    
    return nth_observation

def main() -> None:
    index_file: str = "GammaIndex.csv"
    gamma_file: str = "gamma.dat"
    data = Data(index_file, gamma_file)

    # Data is already a DataFrame with the phase included
    for phase in data.phase_sequence:
        phase_df = data.obs[data.obs['phase'] == phase]
        print(f"Phase {phase} has {len(phase_df)} observations")
        print(phase_df.head())

        if not phase_df.empty:
            phase_df.to_csv(f"gamma_phase{phase}.csv", index=False)

    # Replace 'observations_df' with your actual DataFrame variable name
    nth_instance_df = get_nth_obs( data.obs, phase=0, instance_n=0, num_channels=data.num_channels)
    print(nth_instance_df.head())

if __name__ == '__main__':
    main()
