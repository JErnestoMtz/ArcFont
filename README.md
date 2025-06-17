# ArcFont - Font Embedding Model

Arcfont is an experimental model that generates 1024-dimensional embeddings from text images, focusing on encoding typographic and styling features. The model was developed and trained over a weekend as a proof of concept, so expect some rough edges.

ArcFont draws inspiration from the [ArcFace](https://arxiv.org/abs/1801.07698) architecture commonly used in face recognition, adapting it for font recognition. Initial testing shows promising results compared to other alternatives, though more rigorous evaluation is needed.

The synthetic data generation pipeline and training methodology will be documented in future updates.

## Try it Online

Want to quickly test the model without setting up anything locally? You can try ArcFont through our interactive website and API at [https://caoslabs.com/arcfont](https://caoslabs.com/arcfont).
This also helps support the development of the project.

# Installation

```bash
git clone https://github.com/JErnestoMtz/ArcFont.git
cd ArcFont
pip install -e .
```

**Note**: The model weights (`arcfont.pt`) are available as a release download due to file size constraints. Download the latest `arcfont.pt` from the [Releases page](https://github.com/JErnestoMtz/ArcFont/releases) and place it in the project root directory.

## Quick Start

```python
import torch
from arcfont import load_model, get_embedding
from PIL import Image

# Load model
model_path = "arcfont.pt"  # Path to the model file
model = load_model(model_path)

# Process a single image
image_path = "path/to/your/text/image.jpg"
embedding = get_embedding(image_path, model)

print(f"Embedding shape: {embedding.shape}")
print(f"First 5 values: {embedding[:5]}")
```

## API Reference

### Loading the Model

```python
from arcfont import load_model

model = load_model('arcfont.pt')
```

### Processing Images

```python
from arcfont import get_embedding, get_embeddings_batch

# Single image
embedding = get_embedding('font1.jpg', model)

# Batch processing
image_paths = ['font1.jpg', 'font2.jpg', 'font3.jpg']
embeddings = get_embeddings_batch(image_paths, model, batch_size=32)
```

### Using the ArcFont Class Directly

```python
from arcfont import ArcFont

# Create model instance
model = ArcFont('arcfont.pt')

# Use with preprocessed tensors
import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('font.jpg').convert('RGB')
img_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    embedding_tensor = model(img_tensor)
```

## Font Similarity

Here's how to compute similarity between fonts using cosine similarity:

```python
import numpy as np
from arcfont import get_embedding, load_model

# Load model
model = load_model('arcfont.pt')

# Get embeddings for two fonts
emb1 = get_embedding('font1.jpg', model)
emb2 = get_embedding('font2.jpg', model)

# Compute cosine similarity
similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Similarity: {similarity:.4f}")
```

## Technical Details

- Input: RGB images of text (resized to 240x240)
- Output: 1024-dimensional embedding vector
- Model Format: TorchScript exported model with included weights
- Preprocessing: Standard ImageNet normalization (after resize)

## Use Cases

Here are some practical applications for ArcFont:

1. **Font Identification**: Match fonts in images against a known font database using embedding comparisons.

2. **Font Clustering**: Group similar fonts using clustering algorithms like [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) or [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/).

3. **Font Similarity Search**: Find visually similar fonts by comparing their embeddings.

4. **Document Analysis**: Analyze typographic patterns in documents.

Example of basic font clustering:
```python
from sklearn.cluster import DBSCAN
import numpy as np

# Get embeddings for a set of font images
embeddings = get_embeddings_batch(font_images, model, device)

# Perform clustering
clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
labels = clustering.labels_

# Print cluster assignments
for i, label in enumerate(labels):
    print(f"Font {i} belongs to cluster {label}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

Future updates will include:
- Documentation of the synthetic data generation pipeline
- Training code and methodology
- Model improvements and variants
- Evaluation benchmarks

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{arcfont2024,
  author = {J. Ernesto Martínez},
  title = {ArcFont: A Deep Learning Model for Font Embeddings},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/JErnestoMtz/ArcFont}
}
``` 