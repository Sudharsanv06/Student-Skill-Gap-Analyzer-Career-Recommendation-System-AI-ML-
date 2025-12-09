# Source Code Directory

This directory contains the core Python modules for the project:

## Files:

### `preprocess.py`
Text preprocessing and cleaning utilities:
- `lowercase_text()` - Convert text to lowercase
- `remove_special_characters()` - Clean special characters
- `remove_extra_spaces()` - Normalize spacing
- `preprocess_text()` - Complete preprocessing pipeline
- `preprocess_skills_column()` - Batch preprocessing for DataFrames

### `predict.py`
Career prediction and skill gap analysis:
- `CareerPredictor` class - Main prediction engine
  - Load trained models (Random Forest, TF-IDF, Label Encoder)
  - Predict career roles with confidence scores
  - Identify skill gaps
  - Generate career recommendations
  - Display formatted reports

## Usage:

```python
from predict import CareerPredictor

# Initialize
predictor = CareerPredictor()

# Get recommendation
predictor.display_recommendation("python sklearn pandas")
```
