"""Audio backends: the only code that touches a concrete audio device / SDK."""

from .base import AudioBackend

__all__ = ["AudioBackend"]
