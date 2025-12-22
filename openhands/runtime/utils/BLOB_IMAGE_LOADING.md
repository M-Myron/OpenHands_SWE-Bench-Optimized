# Blob Image Loading for OpenPAI and Similar Environments

This feature enables loading prebuilt Docker images from Azure Blob Storage (or similar mounted storage) instead of pulling from Docker registries or building from scratch. This significantly speeds up evaluation in environments like OpenPAI where network access to Docker registries may be slow or restricted.

## Overview

The blob image loading system:
1. **Checks blob storage first** - Before pulling from registry or building, checks if a prebuilt image tarball exists
2. **Loads efficiently** - Copies tarball to local disk then loads into Docker daemon
3. **Cleans up automatically** - Removes temporary files and optionally Docker images after each instance
4. **Multiprocess-safe** - Each worker process uses its own temp directory and cleanup is non-blocking

## Architecture

### Image Resolution Order

When an image is needed, the system tries (in order):
1. **Local Docker daemon** - Check if image already exists locally
2. **Blob storage** - Load from prebuilt tarball (if enabled)
3. **Remote registry** - Pull from Docker registry
4. **Build locally** - Build from Dockerfile (slowest)

### Cleanup Strategy

For multiprocess safety:
- **Per-process temp dirs**: `/tmp/openhands_blob_images/pid_<PID>/`
- **Immediate tarball cleanup**: Tarballs are removed right after loading into Docker
- **Per-instance Docker cleanup**: Runtime images are removed after each instance completes
- **Shared base images**: Base SWE-bench images are kept (shared across instances)
- **Periodic cleanup**: The run_infer.sh script handles additional Docker cleanup

## Setup

### 1. Prepare Prebuilt Images

On a machine with Docker and the images you want to save:

```bash
# Save a single image
docker save myrepo/myimage:tag | gzip > myimage_tag.tar.gz

# Or use the helper script for batch operations
python openhands/runtime/utils/save_images_to_blob.py \
    --images "image1:tag1,image2:tag2" \
    --output-dir ./image_tarballs
```

### 2. Upload to Azure Blob Storage

```bash
# Using azcopy
azcopy copy ./image_tarballs/* \
    "https://<account>.blob.core.windows.net/<container>/images/?<SAS>"

# Or using Azure CLI
az storage blob upload-batch \
    --account-name <account> \
    --destination <container>/images \
    --source ./image_tarballs
```

### 3. Mount Blob Storage in OpenPAI Job

Configure your OpenPAI job to mount the blob storage:

```yaml
# In your OpenPAI job config
extras:
  virtualCluster: default
taskRoles:
  worker:
    resourcePerInstance:
      cpu: 16
      memoryMB: 65536
      gpu: 0
    commands:
      - export BLOB_MOUNT_DIR=/mnt/blob
      - export BLOB_IMAGES_SUBDIR=images
      - # ... rest of your commands
    dataVolumes:
      - name: blob_images
        mountPath: /mnt/blob
        type: blob
        # Configure your blob connection here
```

### 4. Enable in Evaluation

Set environment variables before running evaluation:

```bash
# Enable blob image loading
export BLOB_MOUNT_DIR=/mnt/blob
export BLOB_IMAGES_SUBDIR=images  # subdirectory within blob mount

# Optional: customize temp directory location
export BLOB_TEMP_DIR=/tmp/openhands_blob_images

# Run evaluation as normal
bash evaluation/benchmarks/swe_bench/scripts/run_infer.sh ...
```

## Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `BLOB_MOUNT_DIR` | Path where blob storage is mounted | None | Yes (to enable) |
| `BLOB_IMAGES_SUBDIR` | Subdirectory within blob mount containing images | `images` | No |
| `BLOB_TEMP_DIR` | Base directory for temporary tarball copies | `/tmp/openhands_blob_images` | No |

## Image Naming Convention

Docker image names are converted to tarball filenames by replacing special characters:

```
Docker Image                          -> Tarball Filename
docker.io/myrepo/myimage:tag         -> docker_io_myrepo_myimage_tag.tar.gz
ghcr.io/openhands/runtime:v1.2.3     -> ghcr_io_openhands_runtime_v1_2_3.tar.gz
```

## Multiprocess Safety

### Process Isolation

Each worker process gets its own temp directory:
```
/tmp/openhands_blob_images/
├── pid_12345/  # Worker 1
│   └── image1.tar.gz
├── pid_12346/  # Worker 2
│   └── image2.tar.gz
└── pid_12347/  # Worker 3
    └── image3.tar.gz
```

### Cleanup Behavior

**Per-instance cleanup (automatic):**
- ✅ Remove process-specific temp directory
- ✅ Remove runtime wrapper images (built on base)
- ❌ Keep base SWE-bench images (shared across instances)

**Periodic cleanup (via run_infer.sh):**
- Every 30 minutes: Docker container prune, image prune, builder prune
- Aggressive cleanup for long-running evaluations

### Why Not Remove Base Images?

Base images (e.g., SWE-bench instance images) are:
1. **Shared across multiple instances** - May be in use by other parallel workers
2. **Large** - Loading them repeatedly is expensive
3. **Read-only** - Don't accumulate modifications

Runtime wrapper images are removed because they:
1. **Instance-specific** - Built uniquely for each evaluation
2. **Accumulate quickly** - New image per instance
3. **Safe to remove** - Only used by one instance

## Troubleshooting

### Images not being loaded from blob

```bash
# Check if blob loading is enabled
echo $BLOB_MOUNT_DIR

# Check if images directory exists
ls -la $BLOB_MOUNT_DIR/$BLOB_IMAGES_SUBDIR

# Check image naming
python -c "
from openhands.runtime.utils.blob_image_loader import get_image_tarball_name
print(get_image_tarball_name('docker.io/myrepo/myimage:tag'))
"
```

### Disk space issues

```bash
# Check Docker disk usage
docker system df

# Manual cleanup (in addition to automatic)
docker system prune -a -f

# Check temp directory size
du -sh /tmp/openhands_blob_images/*
```

### Permission errors

```bash
# Ensure temp directory is writable
chmod 755 /tmp/openhands_blob_images

# Check blob mount permissions
ls -la $BLOB_MOUNT_DIR
```

## Performance Comparison

Typical times for a single SWE-bench instance image:

| Method | Time | Notes |
|--------|------|-------|
| Blob load | ~30-60s | Depends on image size and disk I/O |
| Registry pull | 2-10min | Depends on network speed |
| Build from scratch | 10-30min | Full dependency installation |

For a full evaluation with 300 instances:
- **Without blob loading**: ~10-20 hours (network bottleneck)
- **With blob loading**: ~2-5 hours (disk I/O only)

## Best Practices

1. **Prebuild all images**: Build and upload images for all instances before evaluation
2. **Use fast local storage**: Mount blob to local SSD if possible
3. **Monitor disk space**: Ensure sufficient space for tarballs + Docker images
4. **Regular cleanup**: The periodic cleanup in run_infer.sh helps prevent accumulation
5. **Parallel workers**: Tune `NUM_WORKERS` based on available CPU and disk I/O

## Example: Full Workflow

```bash
# 1. Prebuild images (on a machine with good network)
python evaluation/benchmarks/swe_bench/prebuild_images.py \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --num-workers 4

# 2. Save images to tarballs
for img in $(docker images --format "{{.Repository}}:{{.Tag}}" | grep sweb.eval); do
    name=$(echo $img | tr '/:.' '_')
    docker save $img | gzip > "${name}.tar.gz"
done

# 3. Upload to blob storage
azcopy copy ./*.tar.gz "https://account.blob.core.windows.net/container/images/?SAS"

# 4. In OpenPAI job
export BLOB_MOUNT_DIR=/mnt/blob
export BLOB_IMAGES_SUBDIR=images
bash evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
    config.toml \
    HEAD \
    CodeActAgent \
    10 \
    100 \
    4
```

## Implementation Details

### Key Files

- `openhands/runtime/utils/blob_image_loader.py` - Core blob loading logic
- `openhands/runtime/utils/runtime_build.py` - Integration with runtime building
- `openhands/runtime/builder/docker.py` - Integration with Docker builder
- `evaluation/benchmarks/swe_bench/run_infer.py` - Per-instance cleanup

### Load Flow

```
1. build_runtime_image_in_folder()
   ├─> Check local Docker daemon
   ├─> is_blob_image_loading_enabled()?
   │   ├─> find_blob_image_tarball()
   │   ├─> load_image_from_blob()  # Copy to /tmp/openhands_blob_images/pid_XXX/
   │   ├─> gzip -dc | docker load
   │   └─> cleanup tarball immediately
   ├─> Try pull from registry
   └─> Build from Dockerfile

2. After instance completes (process_instance finally block)
   ├─> cleanup_local_temp_dir()  # Remove /tmp/openhands_blob_images/pid_XXX/
   └─> remove_docker_image(runtime_image)  # Remove wrapper image
```

## Security Considerations

- **SAS tokens**: Use read-only SAS tokens with appropriate expiration
- **Network isolation**: Blob access doesn't require Docker registry access
- **Image verification**: Images are verified after loading (docker images.get)
- **Cleanup**: Temporary files are removed to prevent leakage

## Future Enhancements

- Support for other storage backends (S3, NFS, etc.)
- Image signature verification
- Differential image updates
- Compression format options (zstd, xz)
- Caching layer for frequently used images
