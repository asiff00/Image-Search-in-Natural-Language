document.addEventListener('DOMContentLoaded', function() {
    fetch('/init', {
        method: 'POST'
    }).then(response => response.json())
      .then(data => {
        if (data.status === "initialization started") {
            updateIndexingStatus();
        } else {
            document.getElementById('gallery').classList.add('loaded');
            searchImages('');
        }
    }).catch(error => {
        showToast('Error initializing gallery', 'error');
    });
});

function showToast(message, type = 'success', details = null) {
    const toast = document.getElementById('toast');
    toast.className = `toast toast-${type} show`;
    toast.querySelector('.mdi').className = `mdi mdi-${
        type === 'success' ? 'check-circle' : 
        type === 'warning' ? 'alert' : 'alert-circle'
    }`;
    
    const messageHtml = details 
        ? `<p>${message}<br><span class="details">${details}</span></p>`
        : `<p>${message}</p>`;
    
    toast.querySelector('p').innerHTML = messageHtml;
    
    setTimeout(() => {
        toast.className = 'toast';
    }, 5000);
}

function updateIndexingStatus() {
    fetch('/indexing-status')
        .then(response => response.json())
        .then(data => {
            const statusDiv = document.getElementById('indexing-status');
            const progressFill = document.getElementById('progress-fill');
            const message = document.getElementById('indexing-message');
            const gallery = document.getElementById('gallery');
            
            if (data.status !== 'done') {
                statusDiv.classList.add('active');
                const progress = (data.processed_images / data.total_images) * 100;
                progressFill.style.width = `${progress}%`;
                
                if (data.status === 'indexing') {
                    const type = data.indexing_type === 'full' ? 'Full Index' : 'Updating Index';
                    message.textContent = `${type}: ${data.processed_images}/${data.total_images}`;
                    
                    gallery.querySelectorAll('.gallery-item').forEach((item, index) => {
                        const itemIndex = parseInt(item.dataset.index);
                        if (itemIndex < data.processed_images) {
                            item.classList.remove('indexing', 'processing');
                            item.classList.add('indexed');
                        } else if (itemIndex === data.processed_images) {
                            item.classList.add('processing');
                            item.classList.remove('indexing', 'indexed');
                        } else {
                            item.classList.add('indexing');
                            item.classList.remove('processing', 'indexed');
                        }
                    });
                }
                
                setTimeout(updateIndexingStatus, 500);
            } else {
                statusDiv.classList.remove('active');
                searchImages('');
            }
        });
}

function handleFileUpload(files) {
    const formData = new FormData();
    const gallery = document.getElementById('gallery');
    
    const currentItems = gallery.querySelectorAll('.gallery-item').length;
    const placeholderHtml = Array.from(files).map((_, index) => `
        <div class="gallery-item uploading" data-index="${currentItems + index}" 
             style="opacity: 0; transform: translateY(20px)">
            <div class="gallery-item-content">
                <div class="placeholder-content">
                    <div class="loading-spinner"></div>
                </div>
            </div>
        </div>
    `).join('');
    
    if (gallery.innerHTML) {
        gallery.insertAdjacentHTML('afterbegin', placeholderHtml);
        setTimeout(() => {
            gallery.querySelectorAll('.uploading').forEach(item => {
                item.style.opacity = '1';
                item.style.transform = 'translateY(0)';
            });
        }, 50);
    } else {
        gallery.innerHTML = placeholderHtml;
    }

    for (let file of files) {
        formData.append('files', file);
    }

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.skipped && data.skipped.length > 0) {
            const details = `Skipped files: ${data.skipped.join(', ')}`;
            showToast(data.message, 'warning', details);
        } else {
            showToast(data.message, 'success');
        }
        
        if (data.files && data.files.length > 0) {
            updateIndexingStatus();
            monitorNewImages();
        }
    })
    .catch(error => {
        showToast('Upload failed: ' + error, 'error');
        gallery.querySelectorAll('.uploading').forEach(item => {
            item.style.opacity = '0';
            setTimeout(() => item.remove(), 300);
        });
    });
}

function monitorNewImages() {
    let checkInterval = setInterval(() => {
        fetch('/indexing-status')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'done') {
                    clearInterval(checkInterval);
                    refreshGallery();
                }
            })
            .catch(error => {
                console.error('Error checking indexing status:', error);
                clearInterval(checkInterval);
            });
    }, 1000);
}

function refreshGallery() {
    const gallery = document.getElementById('gallery');
    const currentItems = gallery.querySelectorAll('.gallery-item:not(.uploading)');
    
    currentItems.forEach(item => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(20px)';
    });

    setTimeout(() => {
        searchImages('');
    }, 300);
}

function searchImages(query) {
    const gallery = document.getElementById('gallery');
    console.log('Starting search with query:', query);
    
    gallery.classList.add('loading');
    
    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Search response:', data);
        
        if (data.html) {
            gallery.innerHTML = data.html;
            gallery.classList.add('loaded');
            
            const items = gallery.querySelectorAll('.gallery-item');
            console.log(`Found ${items.length} gallery items`);
            
            items.forEach((item, index) => {
                const img = item.querySelector('img');
                const content = item.querySelector('.gallery-item-content');
                
                if (img) {
                    console.log(`Image ${index + 1}:`, {
                        src: img.src,
                        path: item.dataset.path,
                        complete: img.complete,
                        naturalWidth: img.naturalWidth,
                        naturalHeight: img.naturalHeight
                    });
                    
                    item.classList.remove('loaded');
                    content.style.opacity = '0';
                    
                    img.onload = () => {
                        console.log(`Image ${index + 1} loaded:`, img.src);
                        item.classList.add('loaded');
                        content.style.opacity = '1';
                        item.querySelector('.gallery-item-loading').style.display = 'none';
                    };
                    
                    img.onerror = (e) => {
                        console.error(`Image ${index + 1} failed:`, img.src, e);
                        item.classList.add('error');
                    };
                    
                    if (img.complete) {
                        if (img.naturalWidth) {
                            img.onload();
                        } else {
                            img.onerror();
                        }
                    }
                }
            });
        } else {
            console.warn('No HTML content in response');
            gallery.innerHTML = '<div class="no-results">No images found</div>';
        }
    })
    .catch(error => {
        console.error('Search failed:', error);
        showToast('Search failed: ' + error.message, 'error');
    })
    .finally(() => {
        gallery.classList.remove('loading');
    });
}

const uploadZone = document.getElementById('upload-zone');

['dragenter', 'dragover'].forEach(eventName => {
    uploadZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragging');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    uploadZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragging');
        if (eventName === 'drop') {
            handleFileUpload(e.dataTransfer.files);
        }
    });
});

uploadZone.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = 'image/*';
    input.onchange = (e) => handleFileUpload(e.target.files);
    input.click();
});

const searchBox = document.querySelector('.search-box');
let searchTimeout;

searchBox.addEventListener('input', (e) => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
        searchImages(e.target.value);
    }, 300);
});

document.addEventListener('click', function(e) {
    const galleryItem = e.target.closest('.gallery-item');
    if (galleryItem) {
        const img = galleryItem.querySelector('img');
        if (img) {
            const modal = document.createElement('div');
            modal.className = 'image-preview-modal';
            modal.innerHTML = `
                <div class="modal-content">
                    <img src="${img.src}" alt="Preview">
                    <button class="close-button">
                        <span class="mdi mdi-close"></span>
                    </button>
                </div>
            `;
            
            document.body.appendChild(modal);
            
            modal.addEventListener('click', function(e) {
                if (e.target === modal || e.target.closest('.close-button')) {
                    modal.remove();
                }
            });
        }
    }
});
