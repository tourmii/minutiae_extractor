# Minutiae Extractor
This patch updates and fixes issues in the legacy version of **MinutiaeNet** and **FingerFlow** improving compatibility with newer dependencies and ensuring smoother model loading, inference, and fingerprint preprocessing. It preserves the original architecture and training logic while resolving broken imports and outdated function calls.

![Detection Results](https://github.com/tourmii/minutiae_extractor/blob/main/assets/result.png)

## Features

- **Minutiae Detection**: Extract minutiae points from fingerprint images with high accuracy
- **Minutiae Classification**: Classify minutiae into 6 types:
  - Ending
  - Bifurcation
  - Fragment
  - Enclosure
  - Crossbar
  - Other
- **Core Detection**: Detect and localize fingerprint core regions
- **Easy Integration**: Simple API for seamless integration into your projects

### Minutiae Patches Examples

FingerFlow extracts detailed minutiae patches with precise location and orientation information:

![Minutiae Patches](https://github.com/tourmii/minutiae_extractor/blob/main/assets/patches.png)

## Requirements

- Python >= 3.11
- uv (recommended) or pip for package management

## Installation

### Step 1: Install Models

Before installing the package, you need to download the pre-trained models from Hugging Face:

```bash
python scripts/install_model.py
```

This will download the MinutiaeNet models to the `models/` directory.

### Step 2: Set Up Environment with uv

We recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management.

#### Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Install FingerFlow

```bash
git clone https://github.com/tourmii/minutiae_extractor
cd minutiae_extractor

uv sync
```

Alternatively, if you prefer using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import cv2
import numpy as np
from fingerflow.extractor import Extractor

extractor = Extractor("coarse_net", "fine_net", "classify_net", "core_net")

image = cv2.imread("path/to/your/fingerprint.png")

extracted_minutiae = extractor.extract_minutiae(image)
```

### Detection Pipeline

The extraction process includes multiple stages:

![Detection Pipeline](https://github.com/tourmii/minutiae_extractor/blob/main/assets/minutiae_extract.png)

1. **Original Detection**: Initial minutiae detection identifies all potential points
2. **Otsu Thresholding**: Image segmentation to isolate fingerprint region
3. **Filtered Result**: Final minutiae with accurate orientations and classifications

### Working with Results

The `extract_minutiae` method returns an object containing:

#### Minutiae DataFrame

A Pandas DataFrame with extracted and classified minutiae points:

| Column | Description |
|--------|-------------|
| `x` | X coordinate of the minutiae point |
| `y` | Y coordinate of the minutiae point |
| `angle` | Direction of minutiae point rotation (in radians) |
| `score` | Extraction confidence score (0-1) |
| `class` | Minutiae type: `ending`, `bifurcation`, `fragment`, `enclosure`, `crossbar`, or `other` |

Example:
```python
minutiae_df = extracted_minutiae.minutiae
print(f"Found {len(minutiae_df)} minutiae points")

high_confidence = minutiae_df[minutiae_df['score'] > 0.8]

endings = minutiae_df[minutiae_df['class'] == 'ending']
bifurcations = minutiae_df[minutiae_df['class'] == 'bifurcation']
```

#### Core DataFrame

A Pandas DataFrame with detected fingerprint cores:

| Column | Description |
|--------|-------------|
| `x1` | Left coordinate of bounding box |
| `y1` | Top coordinate of bounding box |
| `x2` | Right coordinate of bounding box |
| `y2` | Bottom coordinate of bounding box |
| `score` | Core detection confidence score (0-1) |
| `w` | Width of bounding box |
| `h` | Height of bounding box |

Example:
```python
cores_df = extracted_minutiae.core
print(f"Found {len(cores_df)} core region(s)")

if len(cores_df) > 0:
    best_core = cores_df.loc[cores_df['score'].idxmax()]
    print(f"Core at ({best_core['x1']}, {best_core['y1']})")
```

## API Reference

### Extractor

The main class for fingerprint feature extraction.

#### Constructor

```python
Extractor(coarse_net, fine_net, classify_net, core_net)
```

**Parameters:**
- `coarse_net` (str): Path to the coarse network model
- `fine_net` (str): Path to the fine network model
- `classify_net` (str): Path to the classification network model
- `core_net` (str): Path to the core detection network model

#### Methods

##### `extract_minutiae(image_data)`

Extracts minutiae points and detects fingerprint cores from an input image.

**Parameters:**
- `image_data` (numpy.ndarray): Input image as a 3D matrix (e.g., output of `cv2.imread()`)

**Returns:**
- An object containing:
  - `minutiae`: Pandas DataFrame with minutiae information
  - `core`: Pandas DataFrame with core detection information

## Project Structure

```
fingerflow/
├── models/              
├── minutiae_patches/    
├── sample/              
├── scripts/             
│   └── install_model.py 
├── src/                 
│   └── fingerflow/      
├── test/                
├── demo.ipynb           
├── pyproject.toml       
├── requirements.txt     
└── README.md            
```

## Development
### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use FingerFlow in your research, please cite:

https://arxiv.org/pdf/1712.09401

## Acknowledgments

Models are hosted on Hugging Face: [tourmaline05/MinutiaeNet](https://huggingface.co/tourmaline05/MinutiaeNet)

## Support

For issues, questions, or contributions, please open an issue on the project repository.

## License

[MIT](https://choosealicense.com/licenses/mit/)