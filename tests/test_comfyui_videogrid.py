#!/usr/bin/env python

"""Tests for `comfyui_videogrid` package."""

import pytest
import torch
from src.comfyui_videogrid.nodes import VideosConcateHorizontal, VideosConcateVertical

@pytest.fixture
def horizontal_concate_node():
    """Fixture to create a VideosConcateHorizontal node instance."""
    return VideosConcateHorizontal()

@pytest.fixture
def vertical_concate_node():
    """Fixture to create a VideosConcateVertical node instance."""
    return VideosConcateVertical()

def test_horizontal_concate_node_initialization(horizontal_concate_node):
    """Test that the horizontal concatenation node can be instantiated."""
    assert isinstance(horizontal_concate_node, VideosConcateHorizontal)

def test_horizontal_concate_metadata():
    """Test the horizontal concatenation node's metadata."""
    assert VideosConcateHorizontal.RETURN_TYPES == ("IMAGE",)
    assert VideosConcateHorizontal.RETURN_NAMES == ("images",)
    assert VideosConcateHorizontal.FUNCTION == "concate_videos"
    assert VideosConcateHorizontal.CATEGORY == "Video Grid"

def test_horizontal_concate_input_types():
    """Test the horizontal concatenation node's input types."""
    input_types = VideosConcateHorizontal.INPUT_TYPES()
    assert "required" in input_types
    assert "images_left" in input_types["required"]
    assert "images_right" in input_types["required"]
    assert input_types["required"]["images_left"][0] == "IMAGE"
    assert input_types["required"]["images_right"][0] == "IMAGE"

def test_vertical_concate_node_initialization(vertical_concate_node):
    """Test that the vertical concatenation node can be instantiated."""
    assert isinstance(vertical_concate_node, VideosConcateVertical)

def test_vertical_concate_metadata():
    """Test the vertical concatenation node's metadata."""
    assert VideosConcateVertical.RETURN_TYPES == ("IMAGE",)
    assert VideosConcateVertical.RETURN_NAMES == ("images",)
    assert VideosConcateVertical.FUNCTION == "concate_videos"
    assert VideosConcateVertical.CATEGORY == "Video Processing Tool"

def test_vertical_concate_input_types():
    """Test the vertical concatenation node's input types."""
    input_types = VideosConcateVertical.INPUT_TYPES()
    assert "required" in input_types
    assert "images_top" in input_types["required"]
    assert "images_bottom" in input_types["required"]
    assert input_types["required"]["images_top"][0] == "IMAGE"
    assert input_types["required"]["images_bottom"][0] == "IMAGE"

@pytest.mark.parametrize(
    "left_shape,right_shape,expected_success",
    [
        # Same dimensions should succeed
        ((2, 100, 200, 3), (2, 100, 200, 3), True),
        # Different frame count should fail
        ((2, 100, 200, 3), (3, 100, 200, 3), False),
        # Different height should fail
        ((2, 100, 200, 3), (2, 150, 200, 3), False),
        # Different channels should fail
        ((2, 100, 200, 3), (2, 100, 200, 4), False),
        # Different width is allowed
        ((2, 100, 200, 3), (2, 100, 300, 3), True),
    ]
)
def test_horizontal_concate_validation(horizontal_concate_node, left_shape, right_shape, expected_success):
    """Test validation of input dimensions for horizontal concatenation."""
    left = torch.zeros(left_shape)
    right = torch.zeros(right_shape)
    
    if expected_success:
        try:
            result = horizontal_concate_node.concate_videos(left, right)
            # Check result dimensions
            expected_width = left_shape[2] + right_shape[2]
            assert result[0].shape == (left_shape[0], left_shape[1], expected_width, left_shape[3])
        except ValueError:
            pytest.fail("Concatenation should have succeeded with these dimensions")
    else:
        with pytest.raises(ValueError):
            horizontal_concate_node.concate_videos(left, right)

@pytest.mark.parametrize(
    "top_shape,bottom_shape,expected_success",
    [
        # Same dimensions should succeed
        ((2, 100, 200, 3), (2, 100, 200, 3), True),
        # Different frame count should fail
        ((2, 100, 200, 3), (3, 100, 200, 3), False),
        # Different width should fail
        ((2, 100, 200, 3), (2, 100, 300, 3), False),
        # Different channels should fail
        ((2, 100, 200, 3), (2, 100, 200, 4), False),
        # Different height is allowed
        ((2, 100, 200, 3), (2, 150, 200, 3), True),
    ]
)
def test_vertical_concate_validation(vertical_concate_node, top_shape, bottom_shape, expected_success):
    """Test validation of input dimensions for vertical concatenation."""
    top = torch.zeros(top_shape)
    bottom = torch.zeros(bottom_shape)
    
    if expected_success:
        try:
            result = vertical_concate_node.concate_videos(top, bottom)
            # Check result dimensions
            expected_height = top_shape[1] + bottom_shape[1]
            assert result[0].shape == (top_shape[0], expected_height, top_shape[2], top_shape[3])
        except ValueError:
            pytest.fail("Concatenation should have succeeded with these dimensions")
    else:
        with pytest.raises(ValueError):
            vertical_concate_node.concate_videos(top, bottom)
