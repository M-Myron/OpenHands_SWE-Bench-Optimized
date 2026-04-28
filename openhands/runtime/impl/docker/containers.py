import docker


def stop_all_containers(prefix: str) -> None:
    docker_client = docker.from_env(timeout=120)
    try:
        # Use Docker's name filter to avoid listing ALL containers on the system.
        # This is critical when running with many concurrent workers — listing all
        # containers hammers the Docker daemon and causes ReadTimeout errors.
        containers = docker_client.containers.list(
            all=True, filters={'name': prefix}
        )
        for container in containers:
            try:
                if container.name.startswith(prefix):
                    container.stop()
            except docker.errors.APIError:
                pass
            except docker.errors.NotFound:
                pass
    except docker.errors.NotFound:  # yes, this can happen!
        pass
    except Exception:
        # Don't let cleanup errors propagate — the container may already be gone
        pass
    finally:
        docker_client.close()
