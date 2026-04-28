"""ThinDockerRuntime: A lightweight Docker runtime for SWE-bench evaluation.

Instead of building a heavy OpenHands overlay image on top of SWE-bench images
(which installs micromamba, poetry, Node.js, Python 3.12, tmux, and 800+ pip
packages), this runtime:

1. Starts the raw SWE-bench base image directly (no build step)
2. Copies a single thin_executor.py (~500 lines, pure Python stdlib) into the container
3. Starts it as an HTTP server for action execution

This eliminates the 8-15 minute image build step per unique base image.
"""

import os
import platform
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

import docker
import httpx
import tenacity

from openhands.core.config import OpenHandsConfig
from openhands.core.exceptions import (
    AgentRuntimeDisconnectedError,
    AgentRuntimeNotFoundError,
)
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action import (
    Action,
    ActionConfirmationStatus,
    AgentThinkAction,
    CmdRunAction,
    FileEditAction,
    FileReadAction,
    FileWriteAction,
)
from openhands.events.event import FileEditSource
from openhands.events.observation import (
    AgentThinkObservation,
    ErrorObservation,
    NullObservation,
    Observation,
    UserRejectObservation,
)
from openhands.events.serialization import event_to_dict, observation_from_dict
from openhands.events.serialization.action import ACTION_TYPE_TO_CLASS
from openhands.integrations.provider import PROVIDER_TOKEN_TYPE
from openhands.llm.llm_registry import LLMRegistry
from openhands.runtime.base import Runtime
from openhands.runtime.impl.docker.containers import stop_all_containers
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime_status import RuntimeStatus
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.port_lock import PortLock, find_available_port_with_lock
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.http_session import HttpSession
from openhands.utils.shutdown_listener import add_shutdown_listener
from openhands.utils.tenacity_stop import stop_if_should_exit

CONTAINER_NAME_PREFIX = 'openhands-thin-'

EXECUTION_SERVER_PORT_RANGE = (30000, 39999)

if os.name == 'nt' or platform.release().endswith('microsoft-standard-WSL2'):
    EXECUTION_SERVER_PORT_RANGE = (30000, 34999)

# Path to the thin_executor.py file that gets copied into containers
THIN_EXECUTOR_PATH = os.path.join(os.path.dirname(__file__), 'thin_executor.py')


def _is_retryable_error(exception):
    if isinstance(exception, tenacity.RetryError):
        cause = exception.last_attempt.exception()
        return _is_retryable_error(cause)
    return isinstance(
        exception,
        (
            ConnectionError,
            httpx.ConnectTimeout,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
            httpx.ReadTimeout,
        ),
    )


class ThinDockerRuntime(Runtime):
    """Lightweight Docker runtime that skips the OpenHands image build step.

    Uses raw SWE-bench Docker images directly and injects a minimal Python
    HTTP server (thin_executor.py) for action execution. The agent, controller,
    and LLM inference all run on the host.

    Only supports:
    - CmdRunAction (bash commands)
    - FileReadAction
    - FileWriteAction
    - FileEditAction (str_replace, view, create)
    - AgentThinkAction
    - AgentFinishAction
    """

    def __init__(
        self,
        config: OpenHandsConfig,
        event_stream: EventStream,
        llm_registry: LLMRegistry,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_callback: Callable | None = None,
        attach_to_existing: bool = False,
        headless_mode: bool = True,
        user_id: str | None = None,
        git_provider_tokens: PROVIDER_TOKEN_TYPE | None = None,
    ):
        self.config = config
        self.status_callback = status_callback

        self._host_port = -1
        self._container_port = -1
        self._host_port_lock: PortLock | None = None

        self.docker_client: docker.DockerClient = self._init_docker_client()
        self.base_container_image = self.config.sandbox.base_container_image
        self.container_name = CONTAINER_NAME_PREFIX + sid
        self.container = None

        self.session = HttpSession()
        self.action_semaphore = threading.Semaphore(1)
        self._runtime_closed = False

        super().__init__(
            config,
            event_stream,
            llm_registry,
            sid,
            plugins,
            env_vars,
            status_callback,
            attach_to_existing,
            headless_mode,
            user_id,
            git_provider_tokens,
        )

    @staticmethod
    def _init_docker_client() -> docker.DockerClient:
        try:
            return docker.from_env(timeout=300)
        except Exception as ex:
            logger.error(
                'Launch docker client failed. Please make sure you have installed docker and started docker desktop/daemon.',
            )
            raise ex

    @property
    def action_execution_server_url(self) -> str:
        return f'{self.config.sandbox.local_runtime_url}:{self._host_port}'

    async def connect(self) -> None:
        self.set_runtime_status(RuntimeStatus.STARTING_RUNTIME)

        if self.attach_to_existing:
            try:
                await call_sync_from_async(self._attach_to_container)
            except docker.errors.NotFound as e:
                raise AgentRuntimeDisconnectedError from e
        else:
            self.log(
                'info',
                f'[ThinDocker] Starting raw container from: {self.base_container_image}',
            )
            await call_sync_from_async(self._start_container)
            self.log('info', f'[ThinDocker] Container started: {self.container_name}')

        self.log(
            'info',
            f'[ThinDocker] Waiting for thin executor at {self.action_execution_server_url}...',
        )
        await call_sync_from_async(self._wait_until_alive)
        self.log('info', '[ThinDocker] Runtime is ready.')

        if not self.attach_to_existing:
            await call_sync_from_async(self.setup_initial_env)

        self.set_runtime_status(RuntimeStatus.READY)
        self._runtime_initialized = True

    def _start_container(self) -> None:
        """Start a raw SWE-bench container and inject thin_executor."""
        # Allocate port
        self._host_port, self._host_port_lock = self._find_available_port_with_lock(
            EXECUTION_SERVER_PORT_RANGE
        )
        self._container_port = self._host_port

        use_host_network = self.config.sandbox.use_host_network
        network_mode = 'host' if use_host_network else None

        port_mapping = None
        if not use_host_network:
            port_mapping = {
                f'{self._container_port}/tcp': [
                    {
                        'HostPort': str(self._host_port),
                        'HostIp': self.config.sandbox.runtime_binding_address,
                    }
                ],
            }

        # Environment variables
        environment = dict(**self.initial_env_vars)
        environment.update({
            'PYTHONUNBUFFERED': '1',
            'PAGER': 'cat',
            'GIT_PAGER': 'cat',
        })
        environment.update(self.config.sandbox.runtime_startup_env_vars)

        if self.base_container_image is None:
            raise ValueError('base_container_image is not set')

        # Ensure the image is available locally (pull if needed)
        self._ensure_image_available(self.base_container_image)

        # Start container with sleep infinity (keep it alive)
        try:
            self.container = self.docker_client.containers.run(
                self.base_container_image,
                command=['bash', '-c', 'sleep infinity'],
                entrypoint=[],
                network_mode=network_mode,
                ports=port_mapping,
                working_dir='/workspace',
                name=self.container_name,
                detach=True,
                environment=environment,
                **(self.config.sandbox.docker_runtime_kwargs or {}),
            )
        except Exception as e:
            self.log('error', f'[ThinDocker] Failed to start container: {e}')
            self.close()
            raise

        # Copy thin_executor.py into the container
        self._copy_file_to_container(THIN_EXECUTOR_PATH, '/tmp/thin_executor.py')

        # Find a working Python in the container
        python_path = self._find_python_in_container()

        # Copy the Python binary to a hidden name so that agent commands like
        # `killall python` or `killall -9 python3` won't kill the executor.
        # `killall` matches on /proc/PID/comm which is the binary basename.
        hidden_python = '/tmp/.thin_exec_py'
        exit_code, _ = self.container.exec_run(
            ['bash', '-c', f'cp {python_path} {hidden_python} && chmod +x {hidden_python}'],
        )
        if exit_code != 0:
            self.log('warning', f'[ThinDocker] Failed to copy python to {hidden_python}, using original')
            hidden_python = python_path

        # Start the thin executor server inside the container
        exec_cmd = f'{hidden_python} /tmp/thin_executor.py {self._container_port} --working-dir /workspace'
        self.log('info', f'[ThinDocker] Starting thin executor: {exec_cmd}')

        # Use docker exec in detached mode
        self.container.exec_run(
            ['bash', '-c', f'nohup {exec_cmd} > /tmp/thin_executor.log 2>&1 &'],
            detach=True,
        )

        self.log(
            'info',
            f'[ThinDocker] Container ready. Server URL: {self.action_execution_server_url}',
        )

    def _find_python_in_container(self) -> str:
        """Find a working Python interpreter in the container."""
        candidates = [
            '/opt/miniconda3/envs/testbed/bin/python',
            '/opt/conda/envs/testbed/bin/python',
            '/opt/miniconda3/bin/python',
            '/usr/bin/python3',
            '/usr/bin/python',
            'python3',
            'python',
        ]
        for python_path in candidates:
            exit_code, output = self.container.exec_run(
                ['bash', '-c', f'{python_path} --version 2>/dev/null && echo OK'],
            )
            if exit_code == 0 and b'OK' in output:
                self.log('info', f'[ThinDocker] Using Python: {python_path}')
                return python_path

        raise RuntimeError(
            'Could not find a working Python interpreter in the container'
        )

    def _ensure_image_available(self, image: str) -> None:
        """Ensure a Docker image is available locally, pulling if needed.

        The Docker SDK's containers.run() auto-pull can fail silently under
        rate limits or network issues, then report a confusing 404 on inspect.
        This method does an explicit pull with retries when the image is missing.
        """
        try:
            self.docker_client.images.get(image)
            self.log('debug', f'[ThinDocker] Image already available locally: {image}')
            return
        except docker.errors.ImageNotFound:
            pass

        self.log('info', f'[ThinDocker] Pulling image: {image}')
        max_retries = 3
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                self.docker_client.images.pull(image, platform=self.config.sandbox.platform or 'linux/amd64')
                self.log('info', f'[ThinDocker] Successfully pulled image: {image}')
                return
            except Exception as e:
                last_err = e
                self.log(
                    'warning',
                    f'[ThinDocker] Pull attempt {attempt}/{max_retries} failed for {image}: {e}',
                )
                if attempt < max_retries:
                    time.sleep(5 * attempt)

        raise docker.errors.ImageNotFound(
            f'Failed to pull image {image} after {max_retries} attempts: {last_err}'
        )

    def _copy_file_to_container(self, host_path: str, container_path: str) -> None:
        """Copy a file from host to container using docker cp."""
        result = subprocess.run(
            ['docker', 'cp', host_path, f'{self.container_name}:{container_path}'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f'Failed to copy {host_path} to container: {result.stderr}'
            )

    def _attach_to_container(self) -> None:
        """Attach to an existing container."""
        self.container = self.docker_client.containers.get(self.container_name)
        if self.container.status == 'exited':
            self.container.start()

        config = self.container.attrs['Config']
        for env_var in config['Env']:
            if env_var.startswith('port='):
                self._host_port = int(env_var.split('port=')[1])
                self._container_port = self._host_port

    @tenacity.retry(
        stop=tenacity.stop_after_delay(120) | stop_if_should_exit(),
        retry=tenacity.retry_if_exception(_is_retryable_error),
        reraise=True,
        wait=tenacity.wait_fixed(2),
    )
    def _wait_until_alive(self) -> None:
        """Wait for the thin executor to become available."""
        try:
            container = self.docker_client.containers.get(self.container_name)
            if container.status == 'exited':
                raise AgentRuntimeDisconnectedError(
                    f'Container {self.container_name} has exited.'
                )
        except docker.errors.NotFound:
            raise AgentRuntimeNotFoundError(
                f'Container {self.container_name} not found.'
            )

        response = self._send_request('GET', f'{self.action_execution_server_url}/alive', timeout=5)
        if response.status_code != 200:
            raise ConnectionError(f'Thin executor not ready: {response.status_code}')

    def _send_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send HTTP request to the thin executor."""
        from openhands.runtime.utils.request import send_request
        return send_request(self.session, method, url, **kwargs)

    @tenacity.retry(
        retry=tenacity.retry_if_exception(
            lambda e: isinstance(e, (httpx.RemoteProtocolError,))
        ),
        stop=tenacity.stop_after_attempt(5) | stop_if_should_exit(),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
    )
    def _send_action_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Send request to the thin executor with retry logic."""
        return self._send_request(method, url, **kwargs)

    # --- Action Execution (Same Interface as ActionExecutionClient) ---

    def send_action_for_execution(self, action: Action) -> Observation:
        """Send an action to the thin executor for execution."""
        # Handle LLM-based edits locally
        if (
            isinstance(action, FileEditAction)
            and action.impl_source == FileEditSource.LLM_BASED_EDIT
        ):
            return self.llm_based_edit(action)

        # Set timeout
        if action.timeout is None:
            if isinstance(action, CmdRunAction) and action.blocking:
                raise RuntimeError('Blocking command with no timeout set')
            action.set_hard_timeout(self.config.sandbox.timeout, blocking=False)

        with self.action_semaphore:
            if not action.runnable:
                if isinstance(action, AgentThinkAction):
                    return AgentThinkObservation('Your thought has been logged.')
                return NullObservation('')

            if (
                hasattr(action, 'confirmation_state')
                and action.confirmation_state == ActionConfirmationStatus.AWAITING_CONFIRMATION
            ):
                return NullObservation('')

            action_type = action.action
            if action_type not in ACTION_TYPE_TO_CLASS:
                raise ValueError(f'Action {action_type} does not exist.')

            if (
                getattr(action, 'confirmation_state', None)
                == ActionConfirmationStatus.REJECTED
            ):
                return UserRejectObservation(
                    'Action has been rejected by the user! Waiting for further user input.'
                )

            assert action.timeout is not None

            try:
                execution_body = {'action': event_to_dict(action)}
                response = self._send_action_request(
                    'POST',
                    f'{self.action_execution_server_url}/execute_action',
                    json=execution_body,
                    timeout=action.timeout + 5,
                )
                output = response.json()
                if getattr(action, 'hidden', False):
                    extras = output.get('extras', {})
                    if extras:
                        extras['hidden'] = True
                obs = observation_from_dict(output)
                obs._cause = action.id
            except httpx.TimeoutException:
                from openhands.core.exceptions import AgentRuntimeTimeoutError
                raise AgentRuntimeTimeoutError(
                    f'Runtime failed to return execute_action before the requested timeout of {action.timeout}s'
                )

            return obs

    def run(self, action: CmdRunAction) -> Observation:
        return self.send_action_for_execution(action)

    def read(self, action: FileReadAction) -> Observation:
        return self.send_action_for_execution(action)

    def write(self, action: FileWriteAction) -> Observation:
        return self.send_action_for_execution(action)

    def edit(self, action: FileEditAction) -> Observation:
        return self.send_action_for_execution(action)

    # --- File Copy Operations ---

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False) -> None:
        """Copy files from host to container using docker cp."""
        if not os.path.exists(host_src):
            raise FileNotFoundError(f'Source file {host_src} does not exist')

        if os.path.isdir(host_src) or recursive:
            # For directories, create a zip and upload via HTTP
            temp_zip_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                    temp_zip_path = tmp.name

                with ZipFile(temp_zip_path, 'w') as zf:
                    if os.path.isdir(host_src):
                        for root, _, files in os.walk(host_src):
                            for f in files:
                                fpath = os.path.join(root, f)
                                arcname = os.path.relpath(fpath, os.path.dirname(host_src))
                                zf.write(fpath, arcname)
                    else:
                        zf.write(host_src, os.path.basename(host_src))

                with open(temp_zip_path, 'rb') as f:
                    self._send_request(
                        'POST',
                        f'{self.action_execution_server_url}/upload_file',
                        files={'file': f},
                        params={'destination': sandbox_dest, 'recursive': 'true'},
                        timeout=300,
                    )
            finally:
                if temp_zip_path and os.path.exists(temp_zip_path):
                    os.unlink(temp_zip_path)
        else:
            # For single files, use docker cp (faster, no HTTP overhead)
            # First ensure destination directory exists
            dest_dir = os.path.dirname(sandbox_dest)
            if dest_dir:
                self.container.exec_run(['mkdir', '-p', dest_dir])

            result = subprocess.run(
                ['docker', 'cp', host_src, f'{self.container_name}:{sandbox_dest}'],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f'Failed to copy to container: {result.stderr}')

    def copy_from(self, path: str) -> Path:
        """Copy files from container to host."""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            params = {'path': path}
            with self.session.stream(
                'GET',
                f'{self.action_execution_server_url}/download_files',
                params=params,
                timeout=30,
            ) as response:
                with open(tmp_path, 'wb') as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            return Path(tmp_path)
        except httpx.TimeoutException:
            raise TimeoutError('Copy operation timed out')

    def list_files(self, path: str | None = None) -> list[str]:
        """List files in the container."""
        try:
            data = {}
            if path is not None:
                data['path'] = path
            response = self._send_request(
                'POST',
                f'{self.action_execution_server_url}/list_files',
                json=data,
                timeout=10,
            )
            return response.json()
        except httpx.TimeoutException:
            raise TimeoutError('List files operation timed out')

    # --- Port Lock Management ---

    def _find_available_port_with_lock(
        self, port_range: tuple[int, int], max_attempts: int = 5
    ) -> tuple[int, PortLock | None]:
        """Find an available port with race condition protection."""
        result = find_available_port_with_lock(
            min_port=port_range[0],
            max_port=port_range[1],
            max_attempts=max_attempts,
            bind_address='0.0.0.0',
            lock_timeout=1.0,
        )

        if result is None:
            logger.warning(
                f'Port locking failed for range {port_range}, falling back'
            )
            port = find_available_tcp_port(port_range[0], port_range[1])
            return port, None

        port, port_lock = result

        # Check if Docker is using this port.
        # Use the low-level API to avoid the TOCTOU race in the SDK's
        # containers.list() which can fail with NotFound when a container
        # is removed between listing and inspecting.
        try:
            raw_containers = self.docker_client.api.containers()
        except Exception:
            raw_containers = []
        for c in raw_containers:
            ports = c.get('Ports', [])
            if any(
                p.get('PublicPort') == port or p.get('PrivatePort') == port
                for p in ports
                if isinstance(p, dict)
            ):
                port_lock.release()
                return self._find_available_port_with_lock(port_range, max_attempts - 1)

        return port, port_lock

    def _release_port_locks(self) -> None:
        """Release all acquired port locks."""
        if self._host_port_lock:
            self._host_port_lock.release()
            self._host_port_lock = None

    # --- Lifecycle ---

    def close(self, rm_all_containers: bool | None = None) -> None:
        """Stop and remove the container."""
        if self._runtime_closed:
            return
        self._runtime_closed = True

        self.session.close()

        if self.config.sandbox.keep_runtime_alive or self.attach_to_existing:
            return

        if rm_all_containers:
            # Bulk cleanup: stop all containers with the thin prefix
            stop_all_containers(CONTAINER_NAME_PREFIX)
        else:
            # Normal path: only stop/remove this specific container.
            # Avoids listing all containers which overloads the Docker daemon
            # when running with many concurrent workers.
            try:
                if self.container is not None:
                    self.container.stop(timeout=10)
                    self.container.remove(force=True)
            except Exception:
                # Container may already be gone — that's fine
                pass
        self._release_port_locks()

    @property
    def vscode_url(self) -> str | None:
        return None

    @property
    def web_hosts(self) -> dict[str, int]:
        return {}

    def browse(self, action):
        return ErrorObservation('Browsing is not supported in ThinDockerRuntime')

    def browse_interactive(self, action):
        return ErrorObservation('Browsing is not supported in ThinDockerRuntime')

    def run_ipython(self, action):
        return ErrorObservation('IPython is not supported in ThinDockerRuntime')

    def get_mcp_config(self, extra_stdio_servers=None):
        from openhands.mcp.mcp_config import MCPConfig
        return MCPConfig()

    async def call_tool_mcp(self, action):
        return ErrorObservation('MCP is not supported in ThinDockerRuntime')
