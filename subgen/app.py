"""
FastAPI application assembly.

Imports the shared ``app`` instance from :mod:`subgen.config`, configures
logging, includes every route router, and starts the background worker
threads.
"""

from subgen.config import app  # The FastAPI instance from config.py
from subgen.logging_setup import configure_logging, log_startup_config

# Configure logging (reads ``debug`` internally)
configure_logging()
log_startup_config()

# ---------------------------------------------------------------------------
# Import and include all routers
# ---------------------------------------------------------------------------
from subgen.routes.status import router as status_router
from subgen.routes.tautulli import router as tautulli_router
from subgen.routes.plex import router as plex_router
from subgen.routes.jellyfin import router as jellyfin_router
from subgen.routes.emby import router as emby_router
from subgen.routes.batch import router as batch_router
from subgen.routes.asr import router as asr_router
from subgen.routes.detect_language import router as detect_language_router

app.include_router(status_router)
app.include_router(tautulli_router)
app.include_router(plex_router)
app.include_router(jellyfin_router)
app.include_router(emby_router)
app.include_router(batch_router)
app.include_router(asr_router)
app.include_router(detect_language_router)

# ---------------------------------------------------------------------------
# Start worker threads
# ---------------------------------------------------------------------------
from subgen.queue.worker import start_workers

start_workers()
