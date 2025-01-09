# Image Search in Natural Language

[![GitHub stars](https://img.shields.io/github/stars/asiff00/Image-Search-in-Natural-Language?style=social)](https://github.com/asiff00/Image-Search-in-Natural-Language)
[![LinkedIn](https://img.shields.io/badge/-LinkedIn-blue?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdullahalasif-bd)
[![YouTube](https://img.shields.io/badge/YouTube-red?style=flat&logo=youtube)](https://youtu.be/NScTko_54uA)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--L--14-orange.svg)](https://github.com/openai/CLIP)

This is a web application that lets you search for images using natural language, powered by modern Vision Transformer (ViT) models. It works by processing and indexing the images you upload, then allowing you to find them by typing in what you're looking for.

## Demo

Watch the demo video to see the application in action:

<video controls src="docs/assets/demo.mp4" title="AI Image Search Tool"></video>


## System Architecture

### Core Components

1. **AIPhotoGallery (Main Controller)**
   - Manages the entire image processing and search pipeline
   - Initializes CLIP model on available device (CPU/CUDA)
   - Handles index management and search operations
   - Components:
     ```python
     class AIPhotoGallery:
         def __init__(self):
             self.model = SentenceTransformer("clip-ViT-L-14")
             self.indexing_manager = IndexingManager()
             self.image_processor = ImageProcessor()
     ```

2. **Image Processing System**
   - Handles image validation and deduplication
   - Supports formats: .jpg, .jpeg, .png, .gif, .bmp, .webp
   - MD5 hash-based duplicate detection
   - Image verification using PIL
   ```python
   class ImageProcessor:
       SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', .gif', '.bmp', '.webp'}
       def process_image(self, image_path: Path):
           image = Image.open(image_path).convert("RGB")
           return self.model.encode(image)
   ```

3. **Indexing System**
   - FAISS-based vector similarity search
   - Asynchronous background indexing
   - Incremental index updates
   - Index caching with 5-minute TTL
   ```
   [Index Structure]
   /Index/
   ├── vector.index      # FAISS vector index
   └── vector.index.paths # Mapped image paths
   ```

### Data Flow

```
[Image Upload Flow]
1. Upload Request → Duplicate Check (MD5) → Save to /images/
2. Background Indexing:
   Image → CLIP Embedding → FAISS Index Update

[Search Flow]
1. Text Query → CLIP Text Embedding
2. FAISS Similarity Search → Top K Similar Images
3. Dynamic HTML Gallery Generation
```

### Directory Organization
```
/
├── app/
│   ├── models/
│   │   ├── gallery.py     # Main controller
│   │   └── indexing.py    # Index management
│   ├── utils/
│   │   ├── image_processor.py  # Image handling
│   │   └── search.py      # FAISS operations
│   ├── routes.py          # FastAPI endpoints
│   └── __init__.py        # App initialization
├── images/                # Image storage
│   └── [uploaded images]
├── Index/                 # FAISS indexes
│   ├── vector.index
│   └── vector.index.paths
└── templates/             # Jinja2 templates
```

### API Implementation

```python
@app.post("/upload")
async def upload(files: List[UploadFile]):
    # 1. Validate and save images
    # 2. Trigger background indexing
    # 3. Return upload status

@app.post("/search")
async def search(query: SearchQuery):
    # 1. Process text query
    # 2. Perform FAISS search
    # 3. Generate gallery HTML
```

### Technical Stack Details

1. **ML Framework**
   - CLIP (ViT-L-14) for text-image embeddings
   - PyTorch backend with CUDA support
   - FAISS for efficient similarity search
   ```python
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = SentenceTransformer("clip-ViT-L-14", device=device)
   ```

2. **Web Framework**
   - FastAPI with async support
   - Jinja2 templating
   - Static file serving
   - Multipart file upload handling

3. **Storage System**
   - File-based image storage
   - FAISS vector index
   - Path mapping for image retrieval

### Performance Features

1. **Efficient Image Processing**
   - Duplicate detection before processing
   - Async background indexing
   - Incremental index updates

2. **Search Optimization**
   - FAISS index caching
   - Lazy loading of gallery images
   - Async search operations

3. **Resource Management**
   - CUDA acceleration when available
   - Background task executor
   - Memory-efficient index loading

## Setup and Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python run.py
   ```

The server will start at `http://0.0.0.0:3000` with hot-reload enabled for development. 

