# NVLabs C-RADIO Models for FiftyOne

![RADIO Models in FiftyOne](cradio-fiftyone.gif)

This repository provides FiftyOne integration for C-RADIO models from NVIDIA Labs. RADIO models are state-of-the-art models that produce rich spatial features and global summaries, making them excellent for image embeddings, similarity search, attention visualization, and downstream computer vision tasks.

## 🚀 Features

- **Multiple Model Variants**: Support for all RADIO model sizes (C-RADIOv3-B/L/H/g, E-RADIO v2)
- **Dual Output Types**: Extract global summary embeddings or spatial attention features
- **Attention Heatmaps**: Visualize what regions the model focuses on
- **FiftyOne Integration**: Seamless integration with FiftyOne's computer vision workflows
- **GPU Acceleration**: Optimized CUDA support with automatic mixed precision (bfloat16)
- **Production Ready**: Robust error handling and device management


## 🛠️ Installation

```bash
# Install FiftyOne
pip install fiftyone

# Register the RADIO model source
import fiftyone.zoo as foz
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/NVLabs_CRADIOV3",
)
```

## 🚀 Quick Start

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load a dataset
dataset = foz.load_zoo_dataset("quickstart", shuffle=True)

# Load RADIO model for embeddings
model = foz.load_zoo_model("nv_labs/c-radio_v3-h")

# Compute embeddings
dataset.compute_embeddings(
    model=model,
    embeddings_field="radio_embeddings",
)

# Launch FiftyOne App
session = fo.launch_app(dataset)
```

## 📊 Available Models

| Model Name | Description | Architecture | Patch Size | Best For |
|------------|-------------|--------------|------------|----------|
| `nv_labs/c-radio_v3-b` | C-RADIOv3-B | ViT-B/16 | 16×16 | Fast inference, moderate accuracy |
| `nv_labs/c-radio_v3-l` | C-RADIOv3-L | ViT-L/16 | 16×16 | Balanced performance |
| `nv_labs/c-radio_v3-h` | C-RADIOv3-H | ViT-H/16 | 16×16 | High accuracy, recommended |
| `nv_labs/c-radio_v3-g` | C-RADIOv3-g | ViT-H/14 | 14×14 | Maximum performance |

## ⚙️ Model Configuration

### Output Types

```python
# Global image embeddings (default)
model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-h",
    output_type="summary"  # Global semantic features
)

# Spatial attention features
spatial_model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-h", 
    output_type="spatial"  # Patch-level spatial features
)
```

### Feature Formats

```python
# Computer vision format (recommended for spatial features)
model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-h",
    output_type="summary",
    feature_format="NCHW"  # "NCHW": [Batch, Channels, Height, Width] , or you can use "NLC":[Batch, Num_patches, Channels]
)

# Transformer format
model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-h",
    output_type="spatial", 
    feature_format="NCHW" # can only use this format for spatial features
)
```

### Complete Configuration Options

```python
model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-h",
    
    # Core settings
    output_type="spatial",              # "summary" or "spatial"
    feature_format="NCHW",             # "NCHW" or "NLC" (NCHW only for spatial)
    
    # Performance options
    use_mixed_precision=True,          # Auto-detected, bfloat16 on Ampere+
    use_external_preprocessor=False,   # Advanced preprocessing
    
    # Spatial heatmap options (when output_type="spatial")
    apply_smoothing=True,              # Smooth attention heatmaps
    smoothing_sigma=1.51,              # Gaussian smoothing strength
    
)
```

## 🔥 Use Cases & Examples

### 1. Global Image Embeddings

Extract high-level semantic representations for similarity search and clustering:

```python
# Compute embeddings
dataset.compute_embeddings(
    model=model,
    embeddings_field="radio_embeddings"
)

```

### 2. Spatial Attention Heatmaps

Visualize what regions the model pays attention to:

```python
# Load spatial model with smoothing
spatial_model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-h",
    output_type="spatial",
    apply_smoothing=True, # or False
    smoothing_sigma=0.51, # used only when apply_smoothing=True
    feature_format="NCHW"
)

# Generate attention heatmaps
dataset.apply_model(spatial_model, "radio_heatmap")

# View heatmaps in FiftyOne App
session = fo.launch_app(dataset)
```

### 3. Embedding Visualization with UMAP

Create 2D visualizations of your image embeddings:

```python
import fiftyone.brain as fob

# First compute embeddings
dataset.compute_embeddings(
    model=model,
    embeddings_field="radio_embeddings"
)

# Create UMAP visualization
results = fob.compute_visualization(
    dataset,
    method="umap",  # Also supports "tsne", "pca"
    brain_key="radio_viz",
    embeddings="radio_embeddings"
)

# Explore in the App
session = fo.launch_app(dataset)
```

### 4. Similarity Search

Build powerful similarity search with RADIO embeddings:

```python
import fiftyone.brain as fob

# Build similarity index
results = fob.compute_similarity(
    dataset,
    backend="sklearn",  # Fast sklearn backend
    brain_key="radio_sim", 
    embeddings="radio_embeddings"
)

# Find similar images
sample_id = dataset.first().id
similar_samples = dataset.sort_by_similarity(
    sample_id,
    brain_key="radio_sim",
    k=10  # Top 10 most similar
)

# View results
session = fo.launch_app(similar_samples)
```

### 5. Dataset Representativeness

Score how representative each sample is of your dataset:

```python
import fiftyone.brain as fob

# Compute representativeness scores
fob.compute_representativeness(
    dataset,
    representativeness_field="radio_represent",
    method="cluster-center",
    embeddings="radio_embeddings"
)

# Find most representative samples
representative_view = dataset.sort_by("radio_represent", reverse=True)
```

### 6. Duplicate Detection

Find and remove near-duplicate images:

```python
import fiftyone.brain as fob

# Detect duplicates using embeddings
results = fob.compute_uniqueness(
    dataset,
    embeddings="radio_embeddings"
)

# Filter to most unique samples
unique_view = dataset.sort_by("uniqueness", reverse=True)

```

### 7. Advanced: Custom Analysis Pipeline

Combine multiple RADIO outputs for comprehensive analysis:

```python
# Step 1: Global embeddings for similarity
embedding_model = foz.load_zoo_model("nv_labs/c-radio_v3-h")
dataset.compute_embeddings(embedding_model, "radio_embeddings")

# Step 2: Spatial heatmaps for attention analysis
spatial_model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-h",
    output_type="spatial",
    apply_smoothing=True,
    smoothing_sigma=0.8
)
dataset.apply_model(spatial_model, "radio_heatmap")

# Step 3: Build similarity index
import fiftyone.brain as fob
fob.compute_similarity(dataset, embeddings="radio_embeddings", brain_key="radio_sim")

# Step 4: Comprehensive analysis
session = fo.launch_app(dataset)
```

## 🔧 Model Architecture Details

### RADIO Foundation Models
- **Architecture**: Vision Transformer with rich multi-scale feature extraction
- **Training**: Large-scale self-supervised and multi-modal training
- **Capabilities**: Both global semantic understanding and fine-grained spatial attention
- **Resolution**: Adaptive resolution support with dynamic resizing

### Feature Specifications
- **Spatial Features**: Rich channel features at multiple spatial scales
- **Preprocessing**: Automatic RGB normalization to [0,1] range
- **Device Management**: Automatic GPU/CPU placement with mixed precision support

## 🛠️ Troubleshooting

### Common Issues

**GPU Memory Errors**
```python
# Use smaller models or disable mixed precision
model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-b",  # Use smaller model
    use_mixed_precision=False
)
```

**Mixed Precision Issues**
```python
# Disable mixed precision on older GPUs
model = foz.load_zoo_model(
    "nv_labs/c-radio_v3-h",
    use_mixed_precision=False
)
```

## 📖 Citation

```bibtex
@misc{heinrich2025radiov25improvedbaselinesagglomerative,
      title={RADIOv2.5: Improved Baselines for Agglomerative Vision Foundation Models},
      author={Greg Heinrich and Mike Ranzinger and Hongxu and Yin and Yao Lu and Jan Kautz and Andrew Tao and Bryan Catanzaro and Pavlo Molchanov},
      year={2024},
      eprint={2412.07679},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.07679},
}
```

## 📄 License

This implementation follows the original RADIO model license. Please refer to the [NVIDIA RADIO repository](https://github.com/NVlabs/RADIO) for complete license details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit:
- 🐛 Bug reports and issues
- 💡 Feature requests and suggestions  
- 🔧 Pull requests with improvements
- 📖 Documentation enhancements

## 🔗 Related Resources

- [NVIDIA RADIO GitHub](https://github.com/NVlabs/RADIO) - Original implementation
- [FiftyOne Documentation](https://docs.voxel51.com/) - Computer vision workflows
- [FiftyOne Model Zoo](https://docs.voxel51.com/user_guide/model_zoo/index.html) - Model ecosystem
- [FiftyOne Brain](https://docs.voxel51.com/user_guide/brain.html) - ML-powered dataset curation
- [RADIO Paper](https://arxiv.org/abs/2412.07679) - Technical details and benchmarks