# AI-Driven Early Detection of Misinformation in Digital Media

A comprehensive machine learning system for detecting fake news and misinformation using state-of-the-art transformer models. This project implements multiple approaches including DistilGPT2 with LoRA fine-tuning and RoBERTa-based classification for robust misinformation detection.

## Project Overview

This project implements an advanced fake news detection system using multiple state-of-the-art approaches:

1. **DistilGPT2 with LoRA**: Efficient fine-tuning for text generation-based classification
2. **RoBERTa Classification**: Robust transformer-based sequence classification
3. **Comprehensive Data Processing**: Advanced preprocessing and augmentation techniques
4. **Extensive Evaluation**: Multi-metric evaluation with detailed visualizations

### Key Features

* **Multiple Model Architectures**:
  - DistilGPT2 with LoRA (Low-Rank Adaptation) fine-tuning
  - RoBERTa for sequence classification with hate speech detection capabilities
* **Advanced Data Balancing**:
  - Oversampling techniques
  - Text augmentation
  - Class weighting
  - Stratified sampling
* **Comprehensive Evaluation**:
  - Confusion matrices with normalization
  - Per-class performance metrics
  - ROC-AUC analysis
  - Error analysis and visualization
* **Training Optimizations**:
  - Gradient checkpointing
  - Mixed precision training (when available)
  - Early stopping with patience
  - Learning rate scheduling
* **Visualization Suite**:
  - Training progress tracking
  - Performance analysis charts
  - Error distribution analysis
  - Class-wise performance radar charts

## Project Structure

```
AI-Driven-Early-Detection-of-Misinformation-in-Digital-Media/
├── fake_news_model/
│   ├── data/
│   │   ├── balance_data.py          # Data balancing utilities
│   │   └── visualizations/          # Data analysis visualizations
│   ├── improved_training.py         # Advanced RoBERTa training pipeline
│   ├── targeted_tuning.py          # DistilGPT2 + LoRA training
│   ├── enhanced_visualizations.py  # Comprehensive visualization suite
│   ├── analyze_confusion_matrix.py # Detailed confusion matrix analysis
│   ├── visualization_analysis.py   # Performance analysis tools
│   ├── run_analysis.py             # Automated analysis runner
│   ├── enhanced_visuals/           # Generated visualization outputs
│   └── improved_results/           # Training results and models
├── main.py                         # Main application entry point
├── requirements.txt               # Project dependencies
└── README.md                     # This file
```

## Dataset Structure

The system works with datasets containing:
* **Training set**: 8,873+ examples
* **Validation set**: 986+ examples  
* **Test set**: 2,465+ examples

**Label Categories**:
* `TRUE`: Factually accurate claims
* `FALSE`: Factually incorrect claims  
* `PARTIALLY TRUE`: Claims with mixed accuracy

**Data Sources**:
* LIAR dataset
* PolitiFact fact-checking database
* Custom curated datasets

## Model Architectures

### 1. RoBERTa Classification (Improved Training)
- **Base Model**: `facebook/roberta-hate-speech-dynabench-r4-target`
- **Architecture**: Sequence classification with 3-class output
- **Features**:
  - Transfer learning from hate speech detection
  - Frozen embeddings and lower layers
  - Gradient checkpointing for memory efficiency
  - Label smoothing and class weighting
  - Cosine learning rate scheduling

### 2. DistilGPT2 + LoRA (Targeted Tuning)
- **Base Model**: DistilGPT2
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Configuration**:
  - r: 8, alpha: 32, dropout: 0.05
  - Target modules: ["c_attn", "c_proj"]
  - Parameter-efficient training

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/mohssinehamada/AI-Driven-Early-Detection-of-Misinformation-in-Digital-Media.git
cd AI-Driven-Early-Detection-of-Misinformation-in-Digital-Media
```

2. **Create and activate virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python main.py
```

### Advanced Training

#### RoBERTa Improved Training:
```bash
python fake_news_model/improved_training.py
```

#### DistilGPT2 + LoRA Training:
```bash
python fake_news_model/targeted_tuning.py
```

#### Generate Comprehensive Analysis:
```bash
python fake_news_model/run_analysis.py
```

### Data Format
Input data should be in JSONL format:
```json
{
    "claim": "Your news claim text here",
    "evidence": ["Supporting evidence 1", "Supporting evidence 2"],
    "label": "SUPPORTS/REFUTES/NOT ENOUGH INFO"
}
```

## Evaluation Metrics

The system provides comprehensive evaluation including:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted averages
- **ROC-AUC**: Multi-class area under curve
- **Confusion Matrix**: Detailed error analysis
- **Class-wise Performance**: Individual class metrics
- **Error Distribution**: Analysis of misclassification patterns

## Visualization Outputs

The system generates various visualizations:

1. **Training Progress**: Loss curves and metric tracking
2. **Confusion Matrices**: Both raw counts and normalized
3. **Performance Radar**: Multi-metric class comparison
4. **Error Analysis**: Misclassification pattern analysis
5. **Class Distribution**: Dataset balance visualization

## Performance Optimization

### Hardware Support
- **MPS (Metal Performance Shaders)**: Optimized for Apple Silicon
- **CPU Fallback**: Automatic fallback for compatibility
- **Memory Optimization**: Gradient checkpointing and efficient batching

### Training Optimizations
- **Mixed Precision**: FP16 training when supported
- **Gradient Accumulation**: Effective larger batch sizes
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Cosine annealing with restarts

## Dependencies

Core requirements:
```
torch>=1.13.0
transformers==4.36.0
accelerate==0.25.0
peft==0.7.1
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
nltk
```

Optional:
```
wandb  # For experiment tracking
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Research Context

This project is part of ongoing research in:
- **Misinformation Detection**: Early identification of false information
- **Transfer Learning**: Leveraging pre-trained models for fact-checking
- **Multi-modal Analysis**: Combining text and metadata features
- **Real-time Detection**: Scalable systems for social media monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Hugging Face**: For the transformers library and pre-trained models
- **Facebook AI**: For the RoBERTa model architecture
- **Microsoft**: For the LoRA adaptation technique
- **Research Community**: For datasets and evaluation frameworks

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{hamada2024ai_misinformation_detection,
  title={AI-Driven Early Detection of Misinformation in Digital Media},
  author={Mohssine Hamada},
  year={2024},
  url={https://github.com/mohssinehamada/AI-Driven-Early-Detection-of-Misinformation-in-Digital-Media}
}
```
