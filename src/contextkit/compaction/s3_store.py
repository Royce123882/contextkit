"""S3-backed compaction store.

Requires the ``s3`` optional dependency group::

    pip install contextkit[s3]
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from types import ModuleType

logger = logging.getLogger("contextkit")


def _import_aiobotocore() -> ModuleType:
    """Import aiobotocore at runtime, raising a clear error if missing."""
    try:
        import aiobotocore as _aiobotocore

        return _aiobotocore
    except ImportError as exc:
        raise ImportError(
            "aiobotocore is required for S3CompactionStore. "
            "Install it with: pip install contextkit[s3]"
        ) from exc


class S3CompactionStore:
    """S3-backed compaction store.

    Stores compaction artifacts as markdown objects in an S3 bucket.

    Requires::

        pip install contextkit[s3]

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix within the bucket.
        region: AWS region (default: from environment).
        session: Optional aiobotocore ``AioSession`` for custom credentials.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "contextkit/compacted/",
        region: str | None = None,
        session: Any = None,
    ) -> None:
        _import_aiobotocore()  # Fail fast if not installed
        self._bucket = bucket
        self._prefix = prefix
        self._region = region
        self._session = session

    def _s3_key(self, key: str) -> str:
        """Build the full S3 object key for an artifact."""
        return f"{self._prefix}{key}.md"

    def _get_session(self) -> Any:
        """Return the aiobotocore session (create default if needed)."""
        if self._session is not None:
            return self._session
        mod = _import_aiobotocore()
        return mod.get_session()

    @asynccontextmanager
    async def _s3_client(self) -> AsyncIterator[Any]:
        """Yield an S3 client from the session with region configuration."""
        session = self._get_session()
        client_kwargs: Dict[str, Any] = {}
        if self._region:
            client_kwargs["region_name"] = self._region
        async with session.create_client("s3", **client_kwargs) as client:
            yield client

    async def save(
        self,
        key: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """Save content to an S3 object.

        Args:
            key: Unique identifier (used in the object key).
            content: The numbered markdown content.
            metadata: Optional metadata stored as S3 object metadata.

        Returns:
            The ``s3://`` URI of the stored object.
        """
        async with self._s3_client() as client:
            put_kwargs: Dict[str, Any] = {
                "Bucket": self._bucket,
                "Key": self._s3_key(key),
                "Body": content.encode("utf-8"),
                "ContentType": "text/markdown; charset=utf-8",
            }
            if metadata:
                put_kwargs["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
            await client.put_object(**put_kwargs)

        return f"s3://{self._bucket}/{self._s3_key(key)}"

    async def load(self, key: str) -> str | None:
        """Load content from an S3 object.

        Args:
            key: The artifact identifier.

        Returns:
            Object contents as a string, or None if not found.
        """
        async with self._s3_client() as client:
            try:
                response = await client.get_object(
                    Bucket=self._bucket,
                    Key=self._s3_key(key),
                )
                body = await response["Body"].read()
                return body.decode("utf-8")
            except client.exceptions.NoSuchKey:
                return None

    async def delete(self, key: str) -> bool:
        """Delete an S3 object.

        Args:
            key: The artifact identifier.

        Returns:
            True if deletion was attempted (S3 DeleteObject is idempotent).
        """
        async with self._s3_client() as client:
            await client.delete_object(
                Bucket=self._bucket,
                Key=self._s3_key(key),
            )
        return True

    async def list_keys(self) -> List[str]:
        """List all artifact keys in the S3 prefix.

        Returns:
            A list of keys (object key stems without prefix and ``.md`` suffix).
        """
        keys: List[str] = []
        async with self._s3_client() as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._bucket,
                Prefix=self._prefix,
            ):
                for obj in page.get("Contents", []):
                    obj_key: str = obj["Key"]
                    if obj_key.endswith(".md"):
                        stem = obj_key[len(self._prefix) : -3]
                        if stem:
                            keys.append(stem)
        return keys
