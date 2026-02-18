from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import os
import struct
import logging
from typing import List, Any, cast
from data import obsData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FrameData:
    """A class to handle reading and managing frame data from gamma index and gamma.dat files."""
    index_filename: str
    gamma_filename: str
    n_bins: int = 600 # todo put this in a config dictionary
    n_channels: int = 64
    stride: int = 1

    def __post_init__(self) -> None:
        """Initialize the frame data by loading the gamma index and reading the gamma observations."""
        logger.info(f"Initializing FrameData with index file: {self.index_filename} and gamma file: {self.gamma_filename}")
        self.gamma_index = self._load_gamma_index()
        self.frames = self._read_gamma_frames()
        logger.info(f"Successfully loaded {self.get_num_frames()} frames")

    def _load_gamma_index(self) -> pd.DataFrame:
        """Load the gamma index file into a DataFrame."""
        logger.info(f"Loading gamma index from {self.index_filename}")
        try:
            index_df = pd.read_csv(self.index_filename)
            logger.info(f"Successfully loaded index with {len(index_df)} entries")
            return index_df
        except Exception as e:
            logger.error(f"Failed to load gamma index: {str(e)}")
            raise

    def _read_gamma_frames(self) -> pd.DataFrame:
        """Read the gamma frames from the gamma.dat file."""
        logger.info(f"Reading gamma frames from {self.gamma_filename}")
        frames_list: List[np.ndarray] = []
        phase_list: List[int] = []
        channel_list: List[int] = []
        index_list: List[int] = []

        try:
            with open(self.gamma_filename, 'rb') as file:
                total_frames = len(self.gamma_index)
                logger.info(f"Processing {total_frames} frames with stride {self.stride}")
                
                for idx, row in enumerate(self.gamma_index.itertuples(index=False), 1):
                    logger.info(f"Processing frame {idx}/{total_frames}")
                    if idx % self.stride == 0:
                        if idx % 100 == 0:  # Log progress every 100 frames
                            logger.info(f"Processing frame {idx}/{total_frames}")
                            
                        # Convert pandas scalar values to Python int
                        file_position = int(cast(Any, row.filePosition))
                        phase = int(cast(Any, row.phase))
                        index = idx - 1

                        if idx < len(self.gamma_index):
                            next_file_position = int(
                                self.gamma_index.iloc[idx]['filePosition'])
                            frame_size = (
                                next_file_position - file_position) // 4
                        else:
                            file.seek(0, os.SEEK_END)
                            end_of_file = file.tell()
                            frame_size = (end_of_file - file_position) // 4

                        file.seek(file_position)
                        frame_data: List[int] = [struct.unpack('i', file.read(4))[0] for _ in range(frame_size)]

                        if len(frame_data) < self.n_channels * self.n_bins:
                            logger.warning(f"Frame {idx} has insufficient data. Padding with zeros.")
                            frame_data += [0] * (self.n_channels * self.n_bins - len(frame_data))

                        if len(frame_data) == self.n_channels * self.n_bins:
                            frame_array = np.array(frame_data).reshape(self.n_channels, self.n_bins)
                            frames_list.append(frame_array)
                            phase_list.append(phase)
                            channel_list.extend(range(self.n_channels))
                            index_list.append(index)
                        else:
                            logger.error(f"Frame {idx} has invalid size: {len(frame_data)}")

            # Create DataFrame from the frame data
            logger.info("Creating DataFrame from processed frames")
            flat_frames = np.vstack(frames_list).reshape(-1, self.n_bins)
            frames_df: pd.DataFrame = pd.DataFrame(flat_frames, columns=[f'bin_{i}' for i in range(self.n_bins)])

            # Add metadata columns
            frames_df.insert(0, 'phase', np.repeat(phase_list, self.n_channels))
            frames_df.insert(1, 'channel', channel_list)
            frames_df.insert(0, 'index', np.repeat(index_list, self.n_channels))

            logger.info(f"Successfully created DataFrame with shape {frames_df.shape}")
            return frames_df
            
        except Exception as e:
            logger.error(f"Error reading gamma frames: {str(e)}")
            raise

    def get_frame(self, frame_index) -> pd.DataFrame:
        """Get a specific frame by its index."""
        logger.debug(f"Retrieving frame {frame_index}")
        start_idx = frame_index * self.n_channels
        end_idx = (frame_index + 1) * self.n_channels
        return self.frames.iloc[start_idx:end_idx]

    def get_phase_frames(self, phase: int) -> pd.DataFrame:
        """Get all frames for a specific phase."""
        logger.debug(f"Retrieving frames for phase {phase}")
        return self.frames[self.frames['phase'] == phase]

    def get_num_frames(self) -> int:
        """Get the total number of frames."""
        return len(self.frames) // self.n_channels

    def get_frame_data(self, frame_index: int) -> np.ndarray:
        """Get the raw frame data as a numpy array."""
        logger.debug(f"Retrieving raw frame data for frame {frame_index}")
        frame = self.get_frame(frame_index)
        return frame.filter(like='bin_').values


if __name__ == "__main__":
    logger.info("Starting FrameData processing")
    
    frame_data = FrameData(
        index_filename="GammaIndex.csv",
        gamma_filename="gamma.dat",
        stride=1  # Optional: process every nth frame
    )

    # Get a specific frame
    frame = frame_data.get_frame(0)
    logger.info(f"Retrieved frame 0 with shape {frame.shape}")

    # Get all frames for a specific phase
    phase_frames = frame_data.get_phase_frames(phase=6)
    logger.info(f"Retrieved {len(phase_frames)} frames for phase 6")

    # Get raw frame data
    frame_data_array = frame_data.get_frame_data(0)
    logger.info(f"Retrieved raw frame data with shape {frame_data_array.shape}")

    print(frame)
    print(phase_frames)
    print(frame_data_array)
