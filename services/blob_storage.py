import mimetypes
import time
import logging
from typing import Optional

from azure.storage.blob import BlobServiceClient, ContentSettings

from services.config import Config

logger = logging.getLogger("blob_storage")


class BlobStorageService:
    def __init__(self, connection_string: Optional[str] = None, container_name: Optional[str] = None):
        conn_str = connection_string or Config.azure_storage_connection_string
        container = container_name or Config.azure_storage_container
        if not conn_str or not container:
            raise ValueError("Azure storage connection string and container name are required.")

        self._client = BlobServiceClient.from_connection_string(conn_str)
        self._container = self._client.get_container_client(container)

    @staticmethod
    def guess_extension(mime_type: Optional[str]) -> str:
        if not mime_type:
            return ".bin"
        ext = mimetypes.guess_extension(mime_type) or ".bin"
        if ext == ".jpe":
            return ".jpg"
        return ext

    @staticmethod
    def _guess_content_type_from_name(blob_name: str) -> Optional[str]:
        # mimetypes guesses based on filename extension
        ctype, _ = mimetypes.guess_type(blob_name)
        return ctype

    def upload_bytes(self, blob_name: str, data: bytes, content_type: Optional[str] = None) -> str:
        # If caller didn't provide content type, try guessing from blob_name
        content_type = content_type or self._guess_content_type_from_name(blob_name)

        settings = ContentSettings(content_type=content_type) if content_type else None

        blob_client = self._container.get_blob_client(blob_name)

        start = time.perf_counter()
        try:
            blob_client.upload_blob(
                data=data,
                overwrite=True,
                content_settings=settings,
            )
        finally:
            ms = (time.perf_counter() - start) * 1000.0
            size_bytes = len(data) if data is not None else 0
            logger.info(
                "[timing] step=azure_blob.upload ms=%.2f blob=%s bytes=%d content_type=%s",
                ms,
                blob_name,
                size_bytes,
                content_type,
            )

        return f"{self._container.url}/{blob_name}"
