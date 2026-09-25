import cv2
import zarr
import numpy as np
from abc import ABC, abstractmethod


class VideoStream(ABC):
    """Abstract base class defining the video stream interface."""
    fps = 60
    
    @property
    @abstractmethod
    def total_frames(self) -> int:
        """Return the total number of frames in the video."""
        pass
    
    @abstractmethod
    def get_frame(self, frame_idx: int):
        """
        Get the image for the specified frame.
        
        Args:
            frame_idx: Frame index.
            
        Returns:
            (success: bool, frame: np.ndarray | None): Success flag and frame image.
        """
        pass
    
    @abstractmethod
    def get_eef_pos(self, frame_idx: int):
        """
        Get the end-effector position for the specified frame.
        
        Args:
            frame_idx: Frame index.
            
        Returns:
            (success: bool, eef_pos: np.ndarray | None): Success flag and end-effector position.
        """
        pass
    
    @abstractmethod
    def close(self):
        """Close the video stream and release resources."""
        pass


class ZarrVideoStream(VideoStream):
    """Video stream implementation backed by Zarr storage."""

    def __init__(self, zarr_group: zarr.Group, episode_idx: int, fps: int = 60) -> None:
        episode_ends = zarr_group['meta/episode_ends']
        if episode_idx < 1:
            self.episode_start = 0
        else:
            self.episode_start = episode_ends[episode_idx - 1]
        self.episode_end = episode_ends[episode_idx]
        # self.frames = zarr_group['data/camera0_rgb'][self.episode_start:self.episode_end]
        self.zarr_group = zarr_group
        self._total_frames = self.episode_end - self.episode_start
        self.current_frame = 0
        self.fps = fps
    
    @property
    def total_frames(self) -> int:
        return self._total_frames
    
    def get_frame(self, frame_idx):
        if frame_idx >= self.total_frames:
            return False, None
        else:
            return True, cv2.cvtColor(self.zarr_group['data/camera0_rgb'][frame_idx + self.episode_start], cv2.COLOR_RGB2BGR)
    
    def get_eef_pos(self, frame_idx):
        if frame_idx >= self.total_frames:
            return False, None
        else:
            return True, self.zarr_group['data/robot0_eef_pos'][frame_idx + self.episode_start]

    def close(self):
        del self.zarr_group


class LerobotVideoStream(VideoStream):
    """Video stream implementation backed by LeRobotDataset."""

    def __init__(self, dataset, episode_idx: int, fps: int = 10, image_key: str = 'observation.images.scene_left_0',
                 eef_key: str = 'observation.state') -> None:
        """
        Initialize LerobotVideoStream.

        Args:
            dataset: LeRobotDataset object.
            episode_idx: Episode index.
            image_key: Image data key. Defaults to 'observation.image'.
        """
        self.dataset = dataset
        self.episode_idx = episode_idx
        self.image_key = image_key
        self.eef_key = eef_key
        self.fps = fps

        index = self.dataset.episode_data_index
        episode_from = index['from'][episode_idx]
        episode_to = index['to'][episode_idx]

        self._episode_from = int(episode_from.item()) if hasattr(episode_from, 'item') else int(episode_from)
        self._episode_to = int(episode_to.item()) if hasattr(episode_to, 'item') else int(episode_to)
        self._total_frames = self._episode_to - self._episode_from

        self.current_frame = 0
    
    @property
    def total_frames(self) -> int:
        return self._total_frames
    
    def get_frame(self, frame_idx: int):
        """
        Get the image for the specified frame.
        
        Args:
            frame_idx: Frame index relative to the episode start.
            
        Returns:
            (success: bool, frame: np.ndarray | None): Success flag and frame image.
        """
        if frame_idx >= self.total_frames or frame_idx < 0:
            return False, None
        
        try:
            global_idx = self._episode_from + frame_idx
            sample = self.dataset[global_idx]
            frame = sample[self.image_key]
            frame = frame.permute(1, 2, 0)
            
            # Ensure frame is a numpy array.
            if not isinstance(frame, np.ndarray):
                frame = np.array(frame)
            
            # Convert float images to uint8 to avoid PIL issues with float data.
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.clip(0, 255).astype(np.uint8)

            # Convert RGB to BGR for OpenCV.
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return True, frame
        except (IndexError, KeyError, TypeError):
            return False, None
    
    def get_eef_pos(self, frame_idx: int):
        """
        Get the end-effector position for the specified frame.
        
        Args:
            frame_idx: Frame index relative to the episode start.
            
        Returns:
            (success: bool, eef_pos: np.ndarray | None): Success flag and end-effector position.
        """
        if frame_idx >= self.total_frames or frame_idx < 0:
            return False, None
        
        global_idx = self._episode_from + frame_idx
        sample = self.dataset[global_idx]
        eef_pos = sample[self.eef_key]
        
        # Ensure the result is a numpy array.
        if not isinstance(eef_pos, np.ndarray):
            eef_pos = np.array(eef_pos)
        
        return True, eef_pos

    def close(self):
        """Close the video stream and release resources."""
        self.dataset = None


VIDEO_STREAM_REGISTRY = {
    "zarr": ZarrVideoStream,
    "lerobot": LerobotVideoStream
}
