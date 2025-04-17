from inspect import cleandoc
from comfy.utils import ProgressBar
import torch

class VideosConcatenate:
    """
    Video Concatenate Node
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "imagesL": ("IMAGE", { "tooltip": "Left image batch" }),
                "imagesR": ("IMAGE", { "tooltip": "Right image batch" }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "concate_videos"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Video Processing Tool"

    def concate_videos(self, imagesL, imagesR):
        # Concatenate two videos horizontally (side by side)
        # In ComfyUI, images are typically PyTorch tensors with shape [batch_size, height, width, channels]
        
        # Convert inputs to PyTorch tensors if they aren't already
        if not isinstance(imagesL, torch.Tensor):
            imagesL = torch.from_numpy(imagesL)
        if not isinstance(imagesR, torch.Tensor):
            imagesR = torch.from_numpy(imagesR)
        
        # Check if number of frames, height, and channels are compatible
        if imagesL.shape[0] != imagesR.shape[0] or imagesL.shape[1] != imagesR.shape[1] or imagesL.shape[3] != imagesR.shape[3]:
            raise ValueError("Both video sequences must have the same number of frames, height, and channels for horizontal stacking")
        
        # Print some info about the input image batches
        print(f"Horizontally stacking videos: {imagesL.shape[0]} frames")
        print(f"Video 1 dimensions: {imagesL.shape[1]}x{imagesL.shape[2]} with {imagesL.shape[3]} channels")
        print(f"Video 2 dimensions: {imagesR.shape[1]}x{imagesR.shape[2]} with {imagesR.shape[3]} channels")
        
        # For larger videos, process in chunks to show progress
        num_frames = imagesL.shape[0]
        result_frames = []
        pbar = ProgressBar(num_frames)
        for i in range(num_frames):
            # Get individual frames
            frame1 = imagesL[i:i+1]
            frame2 = imagesR[i:i+1]
            
            # Concatenate along the width dimension (axis 2)
            concatenated_frame = torch.cat([frame1, frame2], dim=2)
            result_frames.append(concatenated_frame)
            pbar.update(1)
        
        # Combine all frames
        concatenated_images = torch.cat(result_frames, dim=0)
        
        print(f"Concatenation complete. Result dimensions: {concatenated_images.shape[1]}x{concatenated_images.shape[2]}")
        return (concatenated_images,)

NODE_CLASS_MAPPINGS = {
    "VideosConcatenate": VideosConcatenate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideosConcatenate": "Videos Concatenate"
}
