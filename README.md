# 🧠 Brain Tumor Classification using Neural Networks

A sophisticated Streamlit web application for automated brain tumor classification from MRI scans using state-of-the-art deep learning techniques. This application combines computer vision, explainable AI, and medical imaging analysis to provide accurate tumor classification with interpretable results.

![Brain Tumor Classification](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)

![demovideo](https://github.com/user-attachments/assets/8a0cba37-cb7d-4a09-874a-9401faa7c526)

## 🚀 Features

### 🎯 Core Functionality

- **Multi-Model Classification**: Choose between Transfer Learning (Xception) and Custom CNN architectures
- **Real-time Processing**: Instant MRI scan analysis with confidence scores
- **Interactive Interface**: User-friendly Streamlit web interface with drag-and-drop file upload
- **Medical Grade Accuracy**: Trained on comprehensive brain tumor MRI datasets

### 🔍 Explainable AI Features

- **Saliency Map Generation**: Visual heatmaps showing which brain regions influenced the classification
- **AI-Powered Explanations**: Medical explanations using Google's Generative AI (Gemini)
- **Probability Visualization**: Interactive bar charts showing confidence levels for each tumor type
- **Region Highlighting**: Focus on specific brain areas that contributed to the diagnosis

### 📊 Tumor Types Classified

- **Glioma**: Primary brain tumors originating from glial cells
- **Meningioma**: Tumors arising from the meninges (brain covering)
- **Pituitary**: Tumors affecting the pituitary gland
- **No Tumor**: Normal brain scans

## 🛠 Technical Architecture

### 🧠 Deep Learning Models

#### 1. Transfer Learning Model (Xception)

```python
# Architecture Overview
Xception Base Model (ImageNet weights)
├── Global Max Pooling
├── Flatten Layer
├── Dropout (0.3)
├── Dense (128, ReLU)
├── Dropout (0.25)
└── Output Layer (4 classes, Softmax)
```

**Technical Specifications:**

- **Base Model**: Xception with ImageNet pre-trained weights
- **Input Size**: 299×299×3 pixels
- **Optimizer**: Adamax (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall

#### 2. Custom CNN Model

```python
# Architecture Overview
Custom Convolutional Neural Network
├── Multiple Convolutional Layers
├── Pooling Layers
├── Dense Layers
└── Output Layer (4 classes, Softmax)
```

**Technical Specifications:**

- **Input Size**: 224×224×3 pixels
- **Optimizer**: Adamax
- **Loss Function**: Categorical Crossentropy

### 🔬 Saliency Map Implementation

The application uses **Guided Backpropagation** for generating interpretable saliency maps:

```python
# Technical Process
1. Gradient Computation using TensorFlow GradientTape
2. Guided Backpropagation (ReLU on gradients)
3. Channel-wise Maximum Aggregation
4. Circular Brain Mask Application
5. Gaussian Smoothing (11×11 kernel)
6. Hot Colormap Visualization
7. Alpha Blending with Original Image
```

**Key Technical Features:**

- **Gradient-based Visualization**: Uses TensorFlow's automatic differentiation
- **Brain-specific Masking**: Circular mask to focus on brain regions
- **Smoothing Techniques**: Gaussian blur for cleaner visualizations
- **Threshold-based Filtering**: Focuses on most important regions (75th percentile)

### 🤖 AI Explanation System

Powered by Google's Generative AI (Gemini 1.5 Flash):

```python
# Explanation Generation Process
1. Saliency Map Analysis
2. Medical Context Integration
3. Scientific Terminology Usage
4. Region-specific Explanations
5. Confidence-based Reasoning
```

**Technical Implementation:**

- **Model**: Gemini 1.5 Flash
- **Prompt Engineering**: Structured medical prompts
- **Image Analysis**: Direct saliency map interpretation
- **Output Format**: Scientific medical explanations

## 📦 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Google Generative AI API key

### Step-by-Step Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**

   ```bash
   # Copy environment template
   cp env_template.txt .env

   # Edit .env file with your API key
   GOOGLE_API_KEY=your_google_api_key_here
   ```

4. **Model Files Setup**
   Ensure these files are in the project root:
   - `xception_model.weights.h5` - Transfer learning model weights
   - `cnn_model.h5` - Custom CNN model

## 🚀 Usage

### Local Development

```bash
streamlit run app.py
```

### Web Interface Workflow

1. **Upload MRI Scan**: Drag and drop or select an MRI image file
2. **Model Selection**: Choose between Xception or Custom CNN
3. **Analysis**: View classification results and confidence scores
4. **Interpretation**: Examine saliency maps and AI explanations
5. **Visualization**: Explore probability distributions

### Core Libraries

```txt
streamlit==1.28+          # Web application framework
tensorflow==2.0+          # Deep learning framework
opencv-python-headless    # Computer vision (headless for deployment)
numpy                     # Numerical computing
Pillow                    # Image processing
plotly                    # Interactive visualizations
matplotlib                # Scientific plotting
python-dotenv             # Environment variable management
google-generativeai       # AI explanation generation
```

### References

- [Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [What is neural network?](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown)
- [How neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w&ab_channel=3Blue1Brown)
- [Convolution](https://www.youtube.com/watch?v=KuXjwB4LzSA&ab_channel=3Blue1Brown)
- [CNN explainer](https://poloclub.github.io/cnn-explainer/)
- [How Tesla uses Neural Network to power self-driving](https://www.youtube.com/watch?v=FnFksQo-yEY&ab_channel=PreserveKnowledge)
- [Transfer Learning](https://builtin.com/data-science/transfer-learning)
- [Saliency Map implementation](https://medium.com/@bijil.subhash/explainable-ai-saliency-maps-89098e230100)

**⚠️ Medical Disclaimer**: This application is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and clinical oversight.

