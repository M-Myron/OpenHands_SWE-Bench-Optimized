"""
Azure Blob Storage image loader for prebuilt Docker images.

This module provides functionality to load prebuilt Docker images from Azure Blob Storage,
which can significantly speed up image preparation in OpenPAI and similar environments.

Thread-safe design:
- Each process gets its own temp directory (using PID)
- Cleanup only removes files created by the current process
- Docker image removal is handled separately per instance
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from openhands.core.logger import openhands_logger as logger


def get_blob_mount_dir() -> str | None:
    """Get the blob mount directory from environment variable.

    Returns:
        str | None: The blob mount directory path if set, None otherwise.
    """
    return os.environ.get('BLOB_MOUNT_DIR', None)


def get_blob_images_subdir() -> str:
    """Get the subdirectory within blob mount where images are stored.

    Returns:
        str: The subdirectory path, defaults to 'images'.
    """
    return os.environ.get('BLOB_IMAGES_SUBDIR', 'images')


def get_blob_load_timeout() -> int:
    """Get the timeout for loading Docker images from blob storage.

    Returns:
        int: Timeout in seconds, defaults to 600 (10 minutes).
    """
    return int(os.environ.get('BLOB_LOAD_TIMEOUT', '1200'))


def is_blob_image_loading_enabled() -> bool:
    """Check if blob image loading is enabled.

    Returns:
        bool: True if blob image loading is enabled, False otherwise.
    """
    blob_mount_dir = get_blob_mount_dir()
    if not blob_mount_dir:
        return False

    images_dir = os.path.join(blob_mount_dir, get_blob_images_subdir())
    if not os.path.exists(images_dir):
        logger.warning(
            f'Blob mount directory configured ({blob_mount_dir}) but images '
            f'subdirectory does not exist: {images_dir}'
        )
        return False

    return True


def get_image_tarball_name(image_name: str) -> str:
    """Convert a Docker image name to a tarball filename.

    Args:
        image_name: Docker image name (e.g., 'myrepo/myimage:tag')

    Returns:
        str: Tarball filename (e.g., 'myrepo__myimage__tag.tar.gz')
    """
    # Replace special characters with underscores
    safe_name = image_name.replace('/', '__').replace(':', '__')
    return f'{safe_name}.tar.gz'


def find_blob_image_tarball(image_name: str) -> str | None:
    """Find the tarball for a given image in blob storage.

    Args:
        image_name: Docker image name to search for

    Returns:
        str | None: Full path to the tarball if found, None otherwise.
    """
    if not is_blob_image_loading_enabled():
        return None

    blob_mount_dir = get_blob_mount_dir()
    images_dir = os.path.join(blob_mount_dir, get_blob_images_subdir())
    tarball_name = get_image_tarball_name(image_name)
    logger.info(f'Looking for prebuilt image tarball ({tarball_name}) in blob storage at {images_dir}')
    tarball_path = os.path.join(images_dir, tarball_name)

    if os.path.exists(tarball_path):
        logger.info(f'Found prebuilt image tarball in blob storage: {tarball_path}')
        return tarball_path

    logger.debug(f'No prebuilt image tarball found at: {tarball_path}')
    return None


def get_process_temp_dir() -> str:
    """Get a process-specific temp directory for blob images.

    Each process gets its own directory to avoid conflicts in parallel execution.

    Returns:
        str: Process-specific temporary directory path.
    """
    base_dir = os.environ.get('BLOB_TEMP_DIR', '/tmp/openhands_blob_images')
    # Use PID to make it process-specific
    process_dir = os.path.join(base_dir, f'pid_{os.getpid()}')
    return process_dir


def load_image_from_blob(image_name: str) -> tuple[bool, str | None]:
    """Load a Docker image from blob storage tarball.

    This function:
    1. Checks if the image tarball exists in blob storage
    2. Copies it to a process-specific local directory (for better performance)
    3. Loads it into the Docker daemon
    4. Returns the local path for later cleanup

    Args:
        image_name: Docker image name to load

    Returns:
        tuple[bool, str | None]: (success, local_tarball_path or None)
            - success: True if image was successfully loaded
            - local_tarball_path: Path to the local tarball copy (for cleanup), or None if failed
    """
    # remove docker.io prefix if present
    if image_name.startswith('docker.io/'):
        image_name = image_name[len('docker.io/'):]
    tarball_path = find_blob_image_tarball(image_name)
    if not tarball_path:
        return False, None

    # Create process-specific temp directory
    local_temp_dir = get_process_temp_dir()
    local_temp_path = None

    try:
        os.makedirs(local_temp_dir, exist_ok=True)

        # Copy tarball to local disk for faster loading
        tarball_name = os.path.basename(tarball_path)
        local_temp_path = os.path.join(local_temp_dir, tarball_name)

        logger.info(
            f'Copying prebuilt image tarball from blob to local disk: '
            f'{tarball_path} -> {local_temp_path}'
        )
        shutil.copy2(tarball_path, local_temp_path)

        # Load the image into Docker
        logger.info(f'Loading Docker image from tarball: {image_name}')

        # Use gzip -dc to decompress and pipe to docker load
        cmd = f'gzip -dc "{local_temp_path}" | docker load'
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=get_blob_load_timeout(),
        )

        if result.returncode != 0:
            logger.error(
                f'Failed to load image from tarball: {result.stderr}'
            )
            # Clean up failed copy
            if os.path.exists(local_temp_path):
                try:
                    os.remove(local_temp_path)
                except Exception:
                    pass
            return False, None

        logger.info(
            f'✅ Successfully loaded prebuilt image from blob storage: {image_name}'
        )
        logger.debug(f'Docker load output: {result.stdout}')

        # Return success and the local path for later cleanup
        return True, local_temp_path

    except subprocess.TimeoutExpired:
        timeout_seconds = get_blob_load_timeout()
        logger.error(
            f'Timeout while loading image from tarball (>{timeout_seconds} seconds): {image_name}'
        )
        # Clean up on timeout
        if local_temp_path and os.path.exists(local_temp_path):
            try:
                os.remove(local_temp_path)
            except Exception:
                pass
        return False, None
    except Exception as e:
        logger.error(
            f'Error loading image from blob storage: {e}'
        )
        # Clean up on error
        if local_temp_path and os.path.exists(local_temp_path):
            try:
                os.remove(local_temp_path)
            except Exception:
                pass
        return False, None


def cleanup_local_temp_dir(local_temp_dir: str | None = None) -> None:
    """Clean up the local temporary directory used for blob images.

    Only cleans up the process-specific directory, safe for parallel execution.

    Args:
        local_temp_dir: The directory to clean up. If None, uses process-specific default.
    """
    if local_temp_dir is None:
        local_temp_dir = get_process_temp_dir()

    if os.path.exists(local_temp_dir):
        try:
            shutil.rmtree(local_temp_dir)
            logger.debug(f'Cleaned up process-specific temp directory: {local_temp_dir}')
        except Exception as e:
            logger.warning(
                f'Failed to clean up local temp directory {local_temp_dir}: {e}'
            )


def cleanup_tarball(tarball_path: str) -> None:
    """Clean up a specific tarball file.

    Args:
        tarball_path: Path to the tarball file to remove.
    """
    if tarball_path and os.path.exists(tarball_path):
        try:
            os.remove(tarball_path)
            logger.debug(f'Cleaned up tarball: {tarball_path}')
        except Exception as e:
            logger.warning(f'Failed to clean up tarball {tarball_path}: {e}')


def remove_docker_image(image_name: str, force: bool = True) -> bool:
    """Remove a Docker image from the local Docker daemon.

    This should be called after processing an instance to free up disk space.
    Safe to call even if image is being used by other processes (will fail gracefully).

    Args:
        image_name: Docker image name to remove
        force: Whether to force removal (default True)

    Returns:
        bool: True if image was removed, False otherwise (may be in use by other process)
    """
    try:
        cmd = ['docker', 'rmi']
        if force:
            cmd.append('-f')
        cmd.append(image_name)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            logger.info(f'Successfully removed Docker image: {image_name}')
            return True
        else:
            # Image might be in use by another process - this is OK in parallel execution
            if 'is being used' in result.stderr or 'conflict' in result.stderr.lower():
                logger.debug(
                    f'Docker image {image_name} is in use by another process, skipping removal'
                )
            else:
                logger.debug(
                    f'Failed to remove Docker image {image_name}: {result.stderr}'
                )
            return False

    except subprocess.TimeoutExpired:
        logger.warning(f'Timeout while removing Docker image: {image_name}')
        return False
    except Exception as e:
        logger.debug(f'Error removing Docker image {image_name}: {e}')
        return False


class BlobImageContext:
    """Context manager for blob image loading with automatic cleanup.

    Usage:
        with BlobImageContext() as ctx:
            success, image_name = ctx.load_image('myimage:tag')
            if success:
                # Use the image
                ...
        # Cleanup happens automatically here
    """

    def __init__(self):
        self.loaded_images: list[str] = []
        self.tarball_paths: list[str] = []
        self.temp_dir = get_process_temp_dir()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up all resources when exiting context."""
        self.cleanup()
        return False

    def load_image(self, image_name: str) -> tuple[bool, str | None]:
        """Load an image and track it for cleanup.

        Args:
            image_name: Docker image name to load

        Returns:
            tuple[bool, str | None]: (success, image_name or None)
        """
        success, tarball_path = load_image_from_blob(image_name)
        if success:
            self.loaded_images.append(image_name)
            if tarball_path:
                self.tarball_paths.append(tarball_path)
            return True, image_name
        return False, None

    def cleanup(self):
        """Clean up all loaded images and temporary files."""
        # Clean up tarball files first
        for tarball_path in self.tarball_paths:
            cleanup_tarball(tarball_path)
        self.tarball_paths.clear()

        # Clean up temp directory
        cleanup_local_temp_dir(self.temp_dir)

        # Remove Docker images (best effort - may fail if in use by other processes)
        for image_name in self.loaded_images:
            remove_docker_image(image_name, force=False)
        self.loaded_images.clear()
