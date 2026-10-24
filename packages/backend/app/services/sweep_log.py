import logging

# Child of uvicorn.error so records reach the console handler uvicorn installs
# (root has none) and inherit main.py's timestamped formatter.
progress_log = logging.getLogger("uvicorn.error.sweep")
