"""Worker process: consumes the job stream and runs the transcription pipeline."""

from transcription.worker.processor import ShutdownRequested, Worker

__all__ = ["ShutdownRequested", "Worker"]
