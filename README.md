# SquirrelXAI: Photorealistic Inpainting for Object-Level Model Explanations

A complete pipeline for generating perturbation-based explanations using object detection and photorealistic inpainting, inspired by the Splice-XAI research paper. This project uses YOLOv9 for squirrel detection and Stable Diffusion for inpainting to analyze how removing detected objects affects model confidence.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Outputs](#outputs)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a complete explainability pipeline that:

1. **Detects objects** using YOLOv9 (squirrels in Duke dataset)
2. **Generates masks** from bounding boxes or uses existing segmentation masks
3. **Performs inpainting** using Stable Diffusion via Replicate API
4. **Analyzes perturbations** by measuring confidence changes in a CNN classifier
5. **Visualizes results** with comprehensive side-by-side comparisons

The pipeline helps understand which objects are most important for model predictions by measuring how removing them affects classifier confidence.

*For complete implementation details, code and results see the [end-to-end demo notebook](end_to_end_demo.ipynb)*


## 📁 Project Structure

```
.
├── model/
│   └── best.pt                    # Trained YOLOv9 model weights
├── squirrel_data/
│   ├── data.yaml                  # Dataset configuration
│   ├── test/images/               # Test images (65 images)
│   ├── train/images/              # Training images
│   └── valid/images/              # Validation images
├── masks2/                        # Pre-generated masks (optional)
├── outputs/                       # Generated outputs (created during execution)
│   ├── detections/                # YOLO detection results
│   ├── masks/                     # Generated masks
│   ├── inpainted/                 # Inpainted images
│   ├── analysis/                  # Perturbation analysis results
│   └── final_visuals/             # Visualization outputs
├── detect.py                      # Stage 1: YOLOv9 detection
├── generate_masks.py              # Stage 2: Mask generation
├── inpaint.py                     # Stage 3: Inpainting
├── perturbation_analysis.py       # Stage 4: Analysis
├── visualize.py                   # Stage 5: Visualizations
├── end_to_end_demo.ipynb          # Complete demo notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended for faster processing)

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Replicate API** (for inpainting):
   - Sign up for a free account at [Replicate](https://replicate.com)
   - Get your API token from [Replicate Account Settings](https://replicate.com/account/api-tokens)
   - Set the environment variable:
     ```bash
     export REPLICATE_API_TOKEN=your_token_here
     ```
   - Or create a `.env` file with:
     ```
     REPLICATE_API_TOKEN=your_token_here
     ```

4. **Verify model file:**
   - Ensure `model/best.pt` exists and is a valid YOLOv9 model

## 💻 Usage

### Quick Start (Notebook)

The easiest way to run the complete pipeline is through the Jupyter notebook:

```bash
jupyter notebook end_to_end_demo.ipynb
```

Then run all cells sequentially. The notebook will guide you through each stage.

### Command Line Usage

You can also run each stage independently:

#### Stage 1: Detection
```bash
python detect.py \
    --model model/best.pt \
    --images squirrel_data/test/images \
    --output outputs/detections \
    --conf 0.25
```

#### Stage 2: Mask Generation
```bash
python generate_masks.py \
    --detections outputs/detections/detections.json \
    --images squirrel_data/test/images \
    --output outputs/masks \
    --use-existing masks2  # Optional: use existing masks
```

#### Stage 3: Inpainting

First, test the backend:
```bash
python inpaint.py \
    --test \
    --test-image squirrel_data/test/images/image1.jpg \
    --test-mask outputs/masks/image1_mask.png
```

Then run batch inpainting:
```bash
python inpaint.py \
    --masks outputs/masks/mask_paths.json \
    --images squirrel_data/test/images \
    --output outputs/inpainted \
    --backend stable_diffusion
```

#### Stage 4: Perturbation Analysis

First, train the classifier:
```bash
python perturbation_analysis.py \
    --train \
    --train-images squirrel_data/train/images \
    --val-images squirrel_data/valid/images \
    --model outputs/classifier_model.pt
```

Then analyze perturbations:
```bash
python perturbation_analysis.py \
    --detections outputs/detections/detections.json \
    --images squirrel_data/test/images \
    --inpainted outputs/inpainted/stable_diffusion/inpainted_paths.json \
    --model outputs/classifier_model.pt \
    --output outputs/analysis
```

#### Stage 5: Visualizations
```bash
python visualize.py \
    --detections outputs/detections/detections.json \
    --masks outputs/masks/mask_paths.json \
    --inpainted outputs/inpainted/stable_diffusion/inpainted_paths.json \
    --analysis outputs/analysis/perturbation_analysis.csv \
    --images squirrel_data/test/images \
    --output outputs/final_visuals
```

## 🔄 Pipeline Stages

### Stage 1: YOLOv9 Detection
- **Input**: Test images + `best.pt` model
- **Output**: `detections.json` with bounding boxes, scores, and labels
- **Location**: `outputs/detections/`

### Stage 2: Mask Generation
- **Input**: Detection results
- **Output**: Binary masks (either from bounding boxes or existing masks)
- **Location**: `outputs/masks/`
- **Options**: Can use existing masks from `masks2/` directory

### Stage 3: Inpainting
- **Input**: Original images + masks
- **Output**: Inpainted images using Stable Diffusion
- **Location**: `outputs/inpainted/stable_diffusion/`
- **Backend**: Replicate API (free tier available)
- **Verification**: Tests backend before batch processing

### Stage 4: Perturbation Analysis
- **Input**: Original images, inpainted images, detections
- **Output**: 
  - `perturbation_analysis.csv`: Per-image metrics
  - `summary.json`: Aggregate statistics
- **Metrics**:
  - **Confidence Drop (Δ)**: `confidence(original) - confidence(perturbed)`
  - **Flip Rate**: Proportion of images where prediction changed
- **Location**: `outputs/analysis/`

### Stage 5: Visualizations
- **Input**: All previous outputs
- **Output**: 
  - Individual visualizations (side-by-side grids)
  - Summary plots (confidence distributions, scatter plots)
- **Location**: `outputs/final_visuals/`

## 📊 Outputs

### Detection Results (`outputs/detections/detections.json`)
```json
{
  "image1.jpg": {
    "image_path": "...",
    "num_detections": 2,
    "detections": [
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.95,
        "class_id": 0,
        "class_name": "Squirrel"
      }
    ]
  }
}
```

### Analysis Results (`outputs/analysis/perturbation_analysis.csv`)
Columns:
- `image_name`: Image filename
- `original_confidence`: Classifier confidence on original image
- `inpainted_confidence`: Classifier confidence on inpainted image
- `confidence_drop`: Difference (original - inpainted)
- `original_prediction`: Predicted class (0 or 1)
- `inpainted_prediction`: Predicted class after inpainting
- `flip`: 1 if prediction changed, 0 otherwise
- `num_detections`: Number of YOLO detections

### Summary Statistics (`outputs/analysis/summary.json`)
```json
{
  "total_images": 65,
  "mean_confidence_drop": 0.1234,
  "std_confidence_drop": 0.0567,
  "flip_rate": 0.15,
  "total_flips": 10,
  "mean_original_confidence": 0.89,
  "mean_inpainted_confidence": 0.77
}
```

## ⚙️ Configuration

### Detection Threshold
Adjust confidence threshold in `detect.py`:
```python
conf_threshold=0.25  # Lower = more detections, higher = fewer but more confident
```

### Inpainting Prompts
Customize prompts in `inpaint.py`:
```python
DEFAULT_REMOVAL_PROMPT = "seamless natural background, consistent lighting"
DEFAULT_NEGATIVE_PROMPT = "duplicate, distortion, artifacts, low quality"
```

### Classifier Training
Modify training parameters in `perturbation_analysis.py`:
```python
epochs=10
batch_size=16
lr=0.001
```

## 🔧 Troubleshooting

### Replicate API Issues
- **Error**: "API token not found"
  - **Solution**: Set `REPLICATE_API_TOKEN` environment variable
- **Error**: Rate limiting
  - **Solution**: The script includes delays between requests. For large batches, consider processing in smaller chunks.

### Model Loading Issues
- **Error**: "Cannot load model"
  - **Solution**: Verify `model/best.pt` is a valid YOLOv9 model file
  - Ensure ultralytics is installed: `pip install ultralytics`

### Memory Issues
- **Error**: Out of memory during classifier training
  - **Solution**: Reduce `batch_size` in `perturbation_analysis.py`

### Mask Generation
- **Warning**: "Mask not found"
  - **Solution**: Ensure masks exist in `masks2/` or set `--use-existing` to `None` to generate from bounding boxes

## 📝 Notes

- **Free Tier**: Replicate API offers free credits. Monitor usage at [replicate.com/account](https://replicate.com/account)
- **Processing Time**: Batch inpainting can take time due to API rate limits. Plan accordingly.
- **GPU**: While not required, GPU significantly speeds up classifier training and inference.

## 🎓 Methodology

This project implements perturbation-based explainability:

1. **Detection**: Identify objects of interest (squirrels)
2. **Perturbation**: Remove objects via inpainting
3. **Measurement**: Compare model confidence before/after
4. **Analysis**: Quantify importance via confidence drop and flip rate

Objects that cause larger confidence drops when removed are considered more important for the model's predictions.

## 📄 License

This project is for educational/research purposes. Please refer to the original dataset license (CC BY 4.0 as specified in `data.yaml`).

## 🙏 Acknowledgments

- YOLOv9 by Ultralytics
- Stable Diffusion by Stability AI
- Replicate for API access
- Duke Squirrel Dataset



**Happy Explaining! 🐿️**
