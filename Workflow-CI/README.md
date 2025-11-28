# Workflow CI - David kristian silalahi  
MSML Dicoding Submission - Kriteria 3: CI/CD Pipeline with MLflow

## Author: David kristian silalahi
## Level: Skilled (3 points)
## Date: November 2025

## Features:
- ✅ MLflow Project configuration
- ✅ GitHub Actions CI/CD
- ✅ Automated model training
- ✅ Artifact management

## Usage:
```bash
# Local testing
cd MLProject  
mlflow run . -P n_estimators=100

# CI/CD triggers automatically on push
git push origin main
```

## Project Structure:
```
Workflow-CI/
├── README.md
├── .github/
│   └── workflows/
│       └── mlflow-training.yml
└── MLProject/
    ├── modelling.py
    ├── conda.yaml
    ├── MLProject
    ├── namadataset_preprocessing/
    └── artifacts_ci/
```

## Workflow Features:
- **Automated Training**: Triggers on every push to main branch
- **Model Validation**: Runs tests and validation checks
- **Artifact Storage**: Saves trained models and metrics
- **Performance Reporting**: Generates training reports

## MLflow Project:
- **Entry Points**: Configurable model training parameters
- **Environment**: Conda environment with all dependencies
- **Reproducibility**: Consistent results across different environments

## GitHub Actions:
- **CI Pipeline**: Automated testing and training
- **Artifact Management**: Stores models and metrics
- **Quality Checks**: Code quality and model performance validation

---
**MSML Dicoding Submission - Kriteria 3: Skilled Level (3 points)**