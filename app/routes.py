"""
FastAPI routes for the Image Search application.

Handles endpoints for image upload, search, and gallery management.
"""
from fastapi import File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from pathlib import Path
from . import app, templates
from .models.gallery import AIPhotoGallery

gallery = AIPhotoGallery()

class SearchQuery(BaseModel):
    """Data model for search queries.

    Args:
        query (str): The search query string.
    """
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page of the application.

    Args:
        request (Request): The incoming request object.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/init")
async def init_gallery():
    """Initialize the image gallery and start indexing if needed.

    Returns:
        dict: A dictionary containing the status of the initialization.
    
    Raises:
        HTTPException: If an error occurs during initialization.
    """
    try:
        status = gallery.indexing_manager.get_status()
        
        if gallery.indexing_manager.needs_indexing():
            gallery.start_indexing()
            return {"status": "initialization started"}
        
        return {"status": "already initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexing-status")
async def get_indexing_status():
    """Get the current status of the image indexing process.

    Returns:
        dict: A dictionary containing the indexing status.
    """
    return gallery.indexing_manager.get_status()

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Handle image file uploads.
    Checks for duplicates and starts indexing process for new images.

    Args:
        files (List[UploadFile]): A list of uploaded files.

    Returns:
        dict: A dictionary containing the upload message, uploaded files, skipped files, and indexing status.

    Raises:
        HTTPException: If no valid files are uploaded or if an error occurs during the upload process.
    """
    uploaded_files = []
    skipped_files = []
    
    try:
        for file in files:
            try:
                if gallery.image_processor.is_duplicate(file):
                    skipped_files.append(file.filename)
                    continue
                
                file_path = gallery.image_processor.save_uploaded_file(file)
                uploaded_files.append(str(file_path))
            except ValueError as e:
                if "already been uploaded" in str(e):
                    skipped_files.append(file.filename)
                else:
                    raise
        
        message_parts = []
        if uploaded_files:
            message_parts.append(f'Successfully uploaded {len(uploaded_files)} files')
            gallery.indexing_manager.add_new_images(len(uploaded_files))
            gallery.mark_new_images()
            gallery.start_indexing()
        if skipped_files:
            message_parts.append(f'Skipped {len(skipped_files)} duplicate files')
        
        if not uploaded_files and not skipped_files:
            raise HTTPException(status_code=400, detail="No valid files were uploaded")
        
        return {
            'message': '. '.join(message_parts),
            'files': uploaded_files,
            'skipped': skipped_files,
            'status': 'indexing_started'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(query: SearchQuery):
    """
    Search for images using natural language queries.
    Returns HTML markup for displaying matching images in the gallery.

    Args:
        query (SearchQuery): The search query object.

    Returns:
        dict: A dictionary containing the HTML markup for the gallery.

    Raises:
        HTTPException: If an error occurs during the search process.
    """
    try:
        print("Starting search...")
        if not gallery.index_path.exists():
            return {'html': '<div class="error">No index found. Please upload some images first.</div>'}
            
        _, retrieved_images = gallery.retrieve_similar_images(query.query, top_k=12)
        
        print(f"Retrieved images count: {len(retrieved_images) if retrieved_images else 0}")
        
        gallery_html = []
        for i, img_path in enumerate(retrieved_images):
            try:
                relative_path = gallery.image_processor.get_relative_path(img_path)
                url_path = f"/images/{relative_path.as_posix()}"
                
                gallery_html.append(f"""
                    <div class="gallery-item" data-path="{url_path}">
                        <div class="gallery-item-content" style="opacity: 0">
                            <img src="{url_path}" 
                                 alt="Gallery image {i+1}" 
                                 loading="lazy"
                                 onload="this.parentElement.style.opacity = '1';">
                            <div class="gallery-item-overlay">
                                <span class="mdi mdi-eye"></span>
                            </div>
                        </div>
                        <div class="gallery-item-loading">
                            <div class="loading-spinner"></div>
                        </div>
                    </div>
                """)
            except Exception as e:
                print(f"Error processing image {i+1} ({img_path}): {e}")
                continue
        
        final_html = "".join(gallery_html)
        return {'html': final_html}
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
