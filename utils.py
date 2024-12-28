import torch
import numpy as np
import os, subprocess
import imageio
from tqdm import tqdm
from PIL import Image

class config:
    l_frames = 3
    l_channels = 256
    l_height = 7
    l_width = 7
    


def decode_video_tensor_to_gif(video_tensor, output_path, fps=30, grid_size=None):
    """
    Decodes a video tensor batch back into a video grid and saves it as a GIF.

    Args:
        video_tensor (torch.Tensor): A tensor of shape (B, C, T, H, W) representing a batch of videos.
            - B: Batch size (number of videos)
            - C: Number of channels (e.g., 3 for RGB)
            - T: Number of frames
            - H: Height of the frame
            - W: Width of the frame
        output_path (str): Path to the output GIF file.
        fps (int, optional): Frames per second for the output GIF. Defaults to 30.
        grid_size (tuple[int, int], optional): Tuple of (rows, cols) for creating a grid of videos, default is None,
                                                which will use a horizontal grid of all videos in batch.
    """

    if not isinstance(video_tensor, torch.Tensor):
        raise TypeError("Input video_tensor must be a torch.Tensor")

    if video_tensor.ndim != 5:
        raise ValueError(
            f"Video tensor must have 5 dimensions (B, C, T, H, W), got {video_tensor.ndim}"
        )

    batch_size, channels, num_frames, height, width = video_tensor.shape

    if grid_size is None:
        grid_rows = 1
        grid_cols = batch_size
    else:
        grid_rows, grid_cols = grid_size
        if grid_rows * grid_cols != batch_size:
            raise ValueError(
                f"grid_size rows * cols {grid_rows * grid_cols} must match batch size {batch_size}"
            )

    frames_batch_list = []

    for frame_idx in range(num_frames):
        # create a grid of frames from each video
        frames_batch = []
        for row_idx in range(grid_rows):
            frame_row = []
            for col_idx in range(grid_cols):
                video_idx = row_idx * grid_cols + col_idx
                frame = video_tensor[video_idx, :, frame_idx, :, :]
                frame = frame.permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(
                    np.uint8
                )  # Ensure frame is in [0, 255] range
                if frame.shape[2] == 1:  # convert graycale to RGB
                    frame = np.concatenate(
                        [frame, frame, frame], axis=2
                    )  # convert to RGB
                frame_row.append(frame)
            frame_row = np.concatenate(frame_row, axis=1)
            frames_batch.append(frame_row)
        frames_batch = np.concatenate(frames_batch, axis=0)

        frames_batch_list.append(frames_batch)
    # Save as GIF
    imageio.mimsave(output_path, frames_batch_list, fps=fps)
    
    return output_path

def save_video(final_frames, output_path, fps=30):
    frame_paths = []
    for i, frame in tqdm(enumerate(final_frames), desc=''):
        frame = (frame * 255).astype(np.uint8)
        frame_img = Image.fromarray(frame)
        frame_path = os.path.join(output_path, f"frame_{i:04d}.png")
        frame_img.save(frame_path)
        frame_paths.append(frame_path)

    frame_pattern = os.path.join(output_path, "frame_%04d.png")
    ffmpeg_cmd = f"ffmpeg -y -r {fps} -i {frame_pattern} -vcodec libx264 -pix_fmt yuv420p -preset veryfast {output_path}"
    try:
        subprocess.run(
            ffmpeg_cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running ffmpeg:\n{e.stderr.decode()}")
