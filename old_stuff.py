
# # Plot the individual splines for each column
#         if col > sz_target_axis - 5:
#             plt.figure()
#             plt.scatter(x, y, color='r', label='Intermediate Data')
#             plt.plot(x_vals, y_vals, label='Spline')
#             plt.title(f'Column-wise Spline for column {col}')
#             plt.legend()
#             plt.show()
# def model_recon(self, gamma_data: np.ndarray, bin_low: int, bin_high: int) -> np.ndarray:
#     n_active_phases = self.get_n_active_phases()
#     n_channels = self.data.n_channels
#     actual = np.zeros(n_channels * n_active_phases)
#     sub_model = np.zeros(n_channels * n_active_phases)

#     regression = np.zeros(self.data.n_rows)

#     # Prepare phase ADC sums
#     sum_engy_rnge = np.zeros((self.data.n_phases, self.data.n_channels))
#     #  I think this makes two systems c# and python equivalent
#     for phase in range(n_active_phases):
#         sum_engy_rnge[phase] = self.sum_for_energy_range(
#             gamma_data=gamma_data, phase=phase, bin_low=bin_low, bin_high=bin_high)

#     if n_active_phases == 0:  # return empty array
#         return regression

#     max_val = float("-inf")
#     min_val = float("inf")
#     q_max = float("-inf")
#     q_min = float("inf")

#     n_rows = int(self.data.n_rows)
#     n_cols = int(self.data.n_cols)
#     s_models = self.s_models
#     phase_sequence = self.get_phase_sequence()

#     for src_y_pos in range(n_rows):
#         for src_x_pos in range(n_cols):
#             src_pos_index = src_y_pos * n_rows + src_x_pos
#             ch_max = np.full(n_channels, float("-inf"))
#             ch_min = np.full(n_channels, float("inf"))

#             for ch in range(n_channels):
#                 # inverted detector
#                 invertChannel = 63 - ch
#                 logicalChannel = self.channelMap[invertChannel]

#                 for phase in range(n_active_phases):
#                     gamma_val = sum_engy_rnge[phase][logicalChannel]
#                     # not sure whether to include this scaling stuff
#                     # ch_max[logicalChannel] = max(gamma_val, ch_max[logicalChannel])
#                     # ch_min[logicalChannel] = min(gamma_val, ch_min[logicalChannel])

#                     # min_val = min(gamma_val, min_val)
#                     # int corrIndex = logicalChannel * numActivePhases + phase;
#                     # corr_index = logicalChannel + phase * n_channels

#                     corr_index = logicalChannel * n_active_phases + phase
#                     actual[corr_index] = gamma_val
#                     sub_model[corr_index] = s_models[ch,
#                                                      src_pos_index][phase_sequence[phase]]

#             # scaled_actual = self.scale_array(actual, ch_min, ch_max)
#             # q, p_value = pearsonr(sub_model, scaled_actual)
#             q, p_value = pearsonr(sub_model, actual)

#             q_max = max(q, q_max)
#             q_min = min(q, q_min)

#             regression[(self.data.n_rows - 1) - src_pos_index] = q
#             actual.fill(0)
#             sub_model.fill(0)

#     reg_list = self.apply_max_mins(
#         regression, max_val, min_val, q_max, q_min)
#     regr_splined = Splines.spline(
#         np.array(reg_list), self.data.n_spline_size, data=self.data)

#     return regr_splined


# def check():
#     # Define the paths to your CSV files
#     index_file = "GammaIndex.csv"
#     gamma_file = "gamma.dat"
#     model_path = "./model"
#     # reads in the observation data and model data
#     data = Data(index_file, gamma_file)
#     # automatically skips the initial captures in the observation data
#     # Save the observation data to a CSV file
#     data.obs.to_csv("gamma_obs.csv", index=False)

#     # Process each observation cycle
#     max_vals = []
#     obs = np.zeros((data.n_phases, data.n_channels, data.n_bins))
#     num_cycles = data.get_num_cycles()
#     for cycle in range(num_cycles):
#         print(f"processing instance {cycle}")
#         for phase in create_sequence(data.n_phases, cycle):
#             print(f"Phase {phase}")
#             # access the data for the nth observation cycle and the mth phase
#             obs_n_df = data.get_nth_obs_df(phase=phase, instance_n=cycle)
#             obs[phase] = obs_n_df.values  # convert to numpy array
#             print(f"obs[phase] {obs[phase].max()}")
#             max_vals.append(obs[phase].max())

#     # Read the CSV file into a Pandas DataFrame
#     df = pd.read_csv('Gamma_converted.csv')
#     #  select every 4th row starting from 3
#     df2 = df.iloc[3::4]
#     # Find the maximum values in each row
#     max_values_in_rows = pd.DataFrame(df2.max(axis=1))

#     plt.figure()
#     plt.plot([i for i in range(len(max_values_in_rows))],
#              max_values_in_rows, 'o-')
#     plt.plot(max_vals, 'ro-')
#     plt.title(f'max value for each frame of data')
#     plt.legend()
#     plt.grid()
#     plt.show()

# def main() -> None:
#     index_file: str = "GammaIndex.csv"
#     gamma_file: str = "gamma.dat"
#     data = Data(index_file, gamma_file)

#     # Data is already a DataFrame with the phase included
#     for phase in data.phase_sequence:
#         phase_df = data.obs[data.obs['phase'] == phase]
#         print(f"Phase {phase} has {len(phase_df)} observations")
#         print(phase_df.head())

#         if not phase_df.empty:
#             phase_df.to_csv(f"gamma_phase{phase}.csv", index=False)

#     # Replace 'observations_df' with your actual DataFrame variable name
#     nth_instance_df = get_nth_obs( data.obs, phase=0, instance_n=0, num_channels=data.num_channels)
#     print(nth_instance_df.head())

# if __name__ == '__main__':
#     main()

# def read_gamma_obs(data: Data, gamma_file_name: str) -> pd.DataFrame:
#     obs_data_list: List[np.ndarray] = []  # Initialize the list here
#     obs_data_temp: List[np.ndarray] = []
#     phase_list: List[int] = []
#     channel_list: List[int] = []
#     index_list: List[int] = []
#     gamma_index_data: pd.DataFrame = data.gamma_index
#     num_bins: int = data.num_bins
#     num_channels: int = data.num_channels
#     last_phase: int = -1  # Initialize with a value that won't match any phase

#     with open(gamma_file_name, 'rb') as file:
#         for idx, row in gamma_index_data.iterrows():
#             file_position = int(row['filePosition'])
#             phase = int(row['phase'])
#             index = row.name  # Get the index of the current row

#             # Detect phase change and save the previous phase's last observation
#             if phase != last_phase and idx != 0:
#                 obs_data_list.extend(obs_data_temp)
#                 phase_list.append(last_phase)
#                 channel_list.extend(list(range(num_channels)) * len(obs_data_temp))
#                 index_list.extend([last_index] * num_channels * len(obs_data_temp))
#                 obs_data_temp = []  # Reset temporary list for new phase data

#             last_phase = phase
#             last_index = index  # Keep track of the last index

#             if idx < len(gamma_index_data) - 1:
#                 next_file_position: int = int(gamma_index_data.iloc[idx + 1]['filePosition'])
#                 obs_size: int = (next_file_position - file_position) // 4
#             else:
#                 file.seek(0, os.SEEK_END)
#                 end_of_file: int = file.tell()
#                 obs_size: int = (end_of_file - file_position) // 4

#             file.seek(file_position)
#             obs = [struct.unpack('i', file.read(4))[0] for _ in range(obs_size)]

#             if len(obs) < num_channels * num_bins:
#                 obs += [0] * (num_channels * num_bins - len(obs))

#             if len(obs) == num_channels * num_bins:
#                 obs_array: np.ndarray = np.array(obs).reshape(num_channels, num_bins)
#                 obs_data_temp = [obs_array]  # Replace with the new observation data

#     # After the loop, add the last phase data
#     obs_data_list.extend(obs_data_temp)
#     phase_list.append(last_phase)
#     channel_list.extend(list(range(num_channels)) * len(obs_data_temp))
#     index_list.extend([last_index] * num_channels * len(obs_data_temp))

#     # Flatten the list of numpy arrays to create a 2D array
#     flat_obs_data = np.vstack(obs_data_list).reshape(-1, num_bins)

#     # Create a DataFrame from the 2D array with specified bin column names
#     obs_df = pd.DataFrame(flat_obs_data, columns=[f'bin_{i}' for i in range(num_bins)])

#     # Add 'phase' and 'channel' columns
#     obs_df.insert(0, 'phase', np.repeat(phase_list, num_channels))
#     obs_df.insert(1, 'channel', channel_list)

#     # Insert 'index' as the first column
#     obs_df.insert(0, 'index', index_list)

#     return obs_df
