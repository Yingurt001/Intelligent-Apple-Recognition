# Q1: Apple Detection and Counting

## Overview
This notebook implements the first phase of the intelligent apple recognition system - basic apple detection and counting using computer vision techniques.

## Features
- **Image Preprocessing**: Gaussian filtering and color enhancement
- **Apple Detection**: HSV-based red object detection with morphological operations  
- **Apple Counting**: Contour-based counting with area filtering
- **Location Extraction**: Center coordinates and bounding box information
- **Batch Processing**: Process multiple images efficiently
- **Result Visualization**: Step-by-step processing visualization

## Input Data
- **Source**: `data/Attachment 1/`
- **Format**: 200 RGB apple orchard images (270×180 pixels)
- **Content**: Natural orchard scenes with various occlusion types

## Output
- **Result Images**: Processed images with detection overlays
- **Excel Summary**: `results/Q1_results/apple_detection_summary.xlsx`
- **Statistics**: Count totals, averages, and area measurements

## Algorithm Pipeline

1. **Red Saturation Enhancement**: Boost red color channels in HSV space
2. **Gaussian Blur**: Noise reduction (5×5 kernel, σ=1.5)
3. **Color Thresholding**: Extract red objects using BGR range
4. **Morphological Opening**: Remove noise with 5×5 kernel, 3 iterations
5. **Contour Detection**: Find and filter contours by minimum area (100px)
6. **Result Drawing**: Overlay bounding boxes and center points

## Key Parameters
- `min_area = 100`: Minimum contour area to consider as apple
- `red_factor = 1.5`: Red saturation enhancement factor
- `kernel_size = (5,5)`: Morphological operation kernel size

## Usage
1. Ensure data is in `data/Attachment 1/` directory
2. Run all cells in sequence
3. Results will be saved to `results/Q1_results/`
4. Check console output for processing statistics

## Dependencies
- opencv-python>=4.5.0
- numpy>=1.21.0  
- matplotlib>=3.3.0
- pandas>=1.3.0
- openpyxl>=3.0.7

## Notes
- The algorithm focuses on red color detection, suitable for ripe apples
- Area filtering helps eliminate noise and small artifacts
- Results may vary based on lighting conditions and apple ripeness 