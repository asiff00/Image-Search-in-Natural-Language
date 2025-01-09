"""
Main entry point for the Image Search application.
Starts the FastAPI server using uvicorn on host 0.0.0.0:3000 with hot reload enabled.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)
