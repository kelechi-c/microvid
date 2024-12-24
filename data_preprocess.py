import torch
import cv2, os, gc, click
import numpy as np
from pathlib import Path
from torchvision import transforms
from typing import Tuple
from datasets import load_dataset
from diffusers import AutoencoderKLLTXVideo
from huggingface_hub import login

login("")

vae_id = "genmo/mochi-1-preview"
source_data_id = "Doubiiu/webvid10m_motion"
smol_data = "tensorkelechi/tinyvirat"
latents_id = "tensorkelechi/tiny_webvid_latents"
split_size = 1024

ltx_vae = AutoencoderKLLTXVideo.from_pretrained(
    "Lightricks/LTX-Video", subfolder="vae", torch_dtype=torch.bfloat16
).to("cuda")
ltx_vae.enable_tiling()
ltx_vae.eval()

vid_data = (
    load_dataset(source_data_id, split="train")
    .filter(lambda x: x["dynamic_source_category"] != "none")
    .take(split_size)
)

print("loaded dataset and video VAE")

## preprocessing video, from url to tensors
transform_list = [transforms.ToTensor()]
transform_list.append(
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)
transform = transforms.Compose(transform_list)


def vid2tensor(
    batch,
    target_fps: int = 4,
    target_duration: int = 5,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:

    # if isinstance(video_path, str):
    video_path = Path(batch["video"])

    # Read video
    frames = []
    cap = cv2.VideoCapture(str(video_path))

    target_frames = target_fps * target_duration

    while len(frames) < target_frames:
        ret, frame = cap.read()
        if not ret:
            # If video is shorter, loop from beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        # Resize frame
        frame = cv2.resize(frame, target_size)

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        frame_tensor = transform(frame)
        frames.append(frame_tensor)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from video: {video_path}")

    # Stack frames
    video_tensor = torch.stack(frames)

    # Rearrange from (T, C, H, W) to (C, T, H, W)
    batch["video_tensor"] = video_tensor.permute(1, 0, 2, 3).detach().numpy()
    batch["shape"] = batch["video_tensor"].shape
    torch.cuda.empty_cache()
    gc.collect()

    return batch


class VideoProcessor:
    def __init__(
        self,
        output_dir="processed_videos",
        target_size=(224, 224),
        target_fps=4,
        target_duration=5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.target_size = target_size
        self.target_fps = target_fps
        self.target_duration = target_duration

    def download_video(self, video_data):
        """Download video using the contentUrl."""
        video_id = str(video_data["videoid"])
        output_path = self.output_dir / f"{video_id}_raw.mp4"

        try:
            import requests

            response = requests.get(video_data["contentUrl"], stream=True)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path
        except Exception as e:
            print(f"Error downloading video {video_id}: {str(e)}")
            return None

    def process_video(self, input_path, video_data):
        """Process the video to meet target specifications."""
        if not input_path or not input_path.exists():
            return None

        video_id = str(video_data["videoid"])
        output_path = self.output_dir / f"{video_id}_processed.mp4"

        cap = cv2.VideoCapture(input_path)

        # Calculate frames needed for 5 seconds at target FPS
        total_frames_needed = self.target_fps * self.target_duration

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path), fourcc, self.target_fps, self.target_size
        )

        frames_written = 0

        while frames_written < total_frames_needed:
            ret, frame = cap.read()
            if not ret:
                # If video is shorter than 5 seconds, loop from beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            # Resize frame
            resized_frame = cv2.resize(frame, self.target_size)
            out.write(resized_frame)
            frames_written += 1

            # Add delay to achieve target FPS
            cv2.waitKey(int(1000 / self.target_fps))

        cap.release()
        out.release()

        # Clean up raw video
        input_path.unlink()

        return output_path if frames_written > 0 else None

    def process_video_dataset(self, batch):
        # Download video
        raw_path = self.download_video(batch)

        # Process video
        batch["video"] = str(self.process_video(raw_path, batch))
        gc.collect()

        batch = vid2tensor(batch)

        return batch


processor = VideoProcessor(
    output_dir="video_dataset", target_size=(256, 256), target_fps=4, target_duration=5
)


def encode_video(batch):
    with torch.no_grad():
        video_tensor = (
            torch.tensor(batch["video_tensor"])[None].cuda().to(torch.bfloat16)
        )
        latents = ltx_vae.tiled_encode(video_tensor)[0]
        batch["video_latents"] = latents  # .sample()
        batch["latent_shape"] = batch["video_latents"].shape
        del latents
        torch.cuda.empty_cache()
        gc.collect()

    return batch


print(f"start processing/downloads..")
vid_data = vid_data.map(
    processor.process_video_dataset, writer_batch_size=256
).remove_columns(
    [
        "videoid",
        "video",
        "duration",
        "page_dir",
        "dynamic_confidence",
        "dynamic_wording",
    ]
)
vid_data.push_to_hub(smol_data)
print(f"finished downloading and processing {split_size} videos from {source_data_id}")

latent_data = vid_data.map(encode_video).remove_columns(["video_tensor", "shape"])
latent_data.push_to_hub(latents_id)
print(f"dataset preprocessing and latent encoding complete! pushed to {latents_id}")
