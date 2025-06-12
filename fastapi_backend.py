import subprocess
import threading
import time
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
process = None

@app.on_event("startup")
def start_subprocess():
    global process
    # Start the subprocess in the background
    process = subprocess.Popen([
        "python", "realtime.py", "dev"
    ])

@app.on_event("shutdown")
def stop_subprocess():
    global process
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

@app.get("/health")
def health():
    global process
    running = process is not None and process.poll() is None
    return JSONResponse({"backend_running": running}) 
