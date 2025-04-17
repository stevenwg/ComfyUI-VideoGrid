from inspect import cleandoc
from comfy.utils import ProgressBar
import torch

class VideosConcateHorizontal:
    """
    Video Concatenate Node
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_left": ("IMAGE", { "tooltip": "Left image batch" }),
                "images_right": ("IMAGE", { "tooltip": "Right image batch" }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "concate_videos"

    #OUTPUT_NODE = False
    #OUTPUT_TOOLTIPS = ("",) # Tooltips for the output node

    CATEGORY = "Video Grid"

    def concate_videos(self, images_left, images_right):
        # Concatenate two videos horizontally (side by side)
        # In ComfyUI, images are typically PyTorch tensors with shape [batch_size, height, width, channels]
        
        # Convert inputs to PyTorch tensors if they aren't already
        if not isinstance(images_left, torch.Tensor):
            images_left = torch.from_numpy(images_left)
        if not isinstance(images_right, torch.Tensor):
            images_right = torch.from_numpy(images_right)
        
        # Check if number of frames, height, and channels are compatible
        if images_left.shape[0] != images_right.shape[0] or images_left.shape[1] != images_right.shape[1] or images_left.shape[3] != images_right.shape[3]:
            raise ValueError("Both video sequences must have the same number of frames, height, and channels for horizontal stacking")
        
        # Print some info about the input image batches
        print(f"Horizontally stacking videos: {images_left.shape[0]} frames")
        print(f"Video 1 dimensions: {images_left.shape[1]}x{images_left.shape[2]} with {images_left.shape[3]} channels")
        print(f"Video 2 dimensions: {images_right.shape[1]}x{images_right.shape[2]} with {images_right.shape[3]} channels")
        
        # For larger videos, process in chunks to show progress
        num_frames = images_left.shape[0]
        result_frames = []
        pbar = ProgressBar(num_frames)
        for i in range(num_frames):
            # Get individual frames
            frame1 = images_left[i:i+1]
            frame2 = images_right[i:i+1]
            
            # Concatenate along the width dimension (axis 2)
            concatenated_frame = torch.cat([frame1, frame2], dim=2)
            result_frames.append(concatenated_frame)
            pbar.update(1)
        
        # Combine all frames
        concatenated_images = torch.cat(result_frames, dim=0)
        
        print(f"Concatenation complete. Result dimensions: {concatenated_images.shape[1]}x{concatenated_images.shape[2]}")
        return (concatenated_images,)

class VideosConcateVertical:
    """
    Video Concatenate Vertically Node
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_top": ("IMAGE", { "tooltip": "Top image batch" }),
                "images_bottom": ("IMAGE", { "tooltip": "Bottom image batch" }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = cleandoc(__doc__)
    FUNCTION = "concate_videos"

    CATEGORY = "Video Grid"

    def concate_videos(self, images_top, images_bottom):
        # Concatenate two videos vertically (top and bottom)
        # In ComfyUI, images are typically PyTorch tensors with shape [batch_size, height, width, channels]
        
        # Convert inputs to PyTorch tensors if they aren't already
        if not isinstance(images_top, torch.Tensor):
            images_top = torch.from_numpy(images_top)
        if not isinstance(images_bottom, torch.Tensor):
            images_bottom = torch.from_numpy(images_bottom)
        
        # Check if number of frames, width, and channels are compatible
        if images_top.shape[0] != images_bottom.shape[0] or images_top.shape[2] != images_bottom.shape[2] or images_top.shape[3] != images_bottom.shape[3]:
            raise ValueError("Both video sequences must have the same number of frames, width, and channels for vertical stacking")
        
        # Print some info about the input image batches
        print(f"Vertically stacking videos: {images_top.shape[0]} frames")
        print(f"Video 1 dimensions: {images_top.shape[1]}x{images_top.shape[2]} with {images_top.shape[3]} channels")
        print(f"Video 2 dimensions: {images_bottom.shape[1]}x{images_bottom.shape[2]} with {images_bottom.shape[3]} channels")
        
        # For larger videos, process in chunks to show progress
        num_frames = images_top.shape[0]
        result_frames = []
        pbar = ProgressBar(num_frames)
        for i in range(num_frames):
            # Get individual frames
            frame1 = images_top[i:i+1]
            frame2 = images_bottom[i:i+1]
            
            # Concatenate along the height dimension (axis 1)
            concatenated_frame = torch.cat([frame1, frame2], dim=1)
            result_frames.append(concatenated_frame)
            pbar.update(1)
        
        # Combine all frames
        concatenated_images = torch.cat(result_frames, dim=0)
        
        print(f"Concatenation complete. Result dimensions: {concatenated_images.shape[1]}x{concatenated_images.shape[2]}")
        return (concatenated_images,)

NODE_CLASS_MAPPINGS = {
    "VideosConcateHorizontal:": VideosConcateHorizontal,
    "VideosConcateVertical": VideosConcateVertical
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideosConcateHorizontal:": "Videos Concatenate (Horizontal)",
    "VideosConcateVertical": "Videos Concatenate (Vertical)"
}
