"""Top-level package for comfyui_videogrid."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """ComfyUI-VideoGrid"""
__email__ = "you@gmail.com"
__version__ = "0.0.1"

from .src.comfyui_videogrid.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_videogrid.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
