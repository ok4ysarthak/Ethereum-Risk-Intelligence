from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vercel.oidc import Credentials, get_credentials
from .api_client import APIClient, AsyncAPIClient
from .command import (
    AsyncCommand,
    AsyncCommandFinished,
    Command,
    CommandFinished,
)
from .models import (
    CommandResponse,
    Sandbox as SandboxModel,
    SandboxAndRoutesResponse,
    WriteFile,
)


@dataclass
class AsyncSandbox:
    client: AsyncAPIClient
    sandbox: SandboxModel
    routes: list[dict[str, Any]]

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.id

    @property
    def status(self) -> str:
        return self.sandbox.status

    @staticmethod
    async def create(
        *,
        source: dict[str, Any] | None = None,
        ports: list[int] | None = None,
        timeout: int | None = None,
        resources: dict[str, Any] | None = None,
        runtime: str | None = None,
        token: str | None = None,
        project_id: str | None = None,
        team_id: str | None = None,
    ) -> "AsyncSandbox":
        creds: Credentials = get_credentials(
            token=token, project_id=project_id, team_id=team_id
        )
        client = AsyncAPIClient(team_id=creds.team_id, token=creds.token)
        resp: SandboxAndRoutesResponse = await client.create_sandbox(
            project_id=creds.project_id,
            source=source,
            ports=ports,
            timeout=timeout,
            resources=resources,
            runtime=runtime,
        )
        return AsyncSandbox(
            client=client,
            sandbox=resp.sandbox,
            routes=[r.model_dump() for r in resp.routes],
        )

    @staticmethod
    async def get(
        *,
        sandbox_id: str,
        token: str | None = None,
        project_id: str | None = None,
        team_id: str | None = None,
    ) -> "AsyncSandbox":
        creds: Credentials = get_credentials(  # type: ignore[import-untyped]
            token=token, project_id=project_id, team_id=team_id
        )
        client = AsyncAPIClient(team_id=creds.team_id, token=creds.token)
        resp: SandboxAndRoutesResponse = await client.get_sandbox(sandbox_id=sandbox_id)
        return AsyncSandbox(
            client=client,
            sandbox=resp.sandbox,
            routes=[r.model_dump() for r in resp.routes],
        )

    def domain(self, port: int) -> str:
        for r in self.routes:
            if r.get("port") == port:
                # Prefer URL when provided by the API; fall back to subdomain
                return r.get("url") or f"https://{r['subdomain']}.vercel.run"
        raise ValueError(f"No route for port {port}")

    async def get_command(self, cmd_id: str) -> AsyncCommand:
        resp = await self.client.get_command(sandbox_id=self.sandbox.id, cmd_id=cmd_id)
        assert isinstance(resp, CommandResponse)
        return AsyncCommand(
            client=self.client, sandbox_id=self.sandbox.id, cmd=resp.command
        )

    async def run_command(
        self,
        cmd: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        sudo: bool = False,
    ) -> AsyncCommandFinished:
        command_response = await self.client.run_command(
            sandbox_id=self.sandbox.id,
            command=cmd,
            args=args or [],
            cwd=cwd,
            env=env or {},
            sudo=sudo,
        )
        command = AsyncCommand(
            client=self.client, sandbox_id=self.sandbox.id, cmd=command_response.command
        )
        # Wait for completion
        return await command.wait()

    async def run_command_detached(
        self,
        cmd: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        sudo: bool = False,
    ) -> AsyncCommand:
        command_response = await self.client.run_command(
            sandbox_id=self.sandbox.id,
            command=cmd,
            args=args or [],
            cwd=cwd,
            env=env or {},
            sudo=sudo,
        )
        return AsyncCommand(
            client=self.client, sandbox_id=self.sandbox.id, cmd=command_response.command
        )

    async def mk_dir(self, path: str, *, cwd: str | None = None) -> None:
        await self.client.mk_dir(sandbox_id=self.sandbox.id, path=path, cwd=cwd)

    async def read_file(self, path: str, *, cwd: str | None = None) -> bytes | None:
        return await self.client.read_file(
            sandbox_id=self.sandbox.id, path=path, cwd=cwd
        )

    async def write_files(self, files: list[WriteFile]) -> None:
        await self.client.write_files(
            sandbox_id=self.sandbox.id,
            cwd=self.sandbox.cwd,
            extract_dir="/",
            files=files,
        )

    async def stop(self) -> None:
        await self.client.stop_sandbox(sandbox_id=self.sandbox.id)

    # Async context manager to ensure cleanup
    async def __aenter__(self) -> "AsyncSandbox":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self.stop()
        except Exception:
            # Best-effort stop; ignore errors during teardown
            pass
        await self.client.aclose()


@dataclass
class Sandbox:
    client: APIClient
    sandbox: SandboxModel
    routes: list[dict[str, Any]]

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.id

    @property
    def status(self) -> str:
        return self.sandbox.status

    @staticmethod
    def create(
        *,
        source: dict[str, Any] | None = None,
        ports: list[int] | None = None,
        timeout: int | None = None,
        resources: dict[str, Any] | None = None,
        runtime: str | None = None,
        token: str | None = None,
        project_id: str | None = None,
        team_id: str | None = None,
    ) -> "Sandbox":
        creds: Credentials = get_credentials(
            token=token, project_id=project_id, team_id=team_id
        )
        client = APIClient(team_id=creds.team_id, token=creds.token)
        resp: SandboxAndRoutesResponse = client.create_sandbox(
            project_id=creds.project_id,
            source=source,
            ports=ports,
            timeout=timeout,
            resources=resources,
            runtime=runtime,
        )
        return Sandbox(
            client=client,
            sandbox=resp.sandbox,
            routes=[r.model_dump() for r in resp.routes],
        )

    @staticmethod
    def get(
        *,
        sandbox_id: str,
        token: str | None = None,
        project_id: str | None = None,
        team_id: str | None = None,
    ) -> "Sandbox":
        creds: Credentials = get_credentials(
            token=token, project_id=project_id, team_id=team_id
        )
        client = APIClient(team_id=creds.team_id, token=creds.token)
        resp: SandboxAndRoutesResponse = client.get_sandbox(sandbox_id=sandbox_id)
        return Sandbox(
            client=client,
            sandbox=resp.sandbox,
            routes=[r.model_dump() for r in resp.routes],
        )

    def domain(self, port: int) -> str:
        for r in self.routes:
            if r.get("port") == port:
                return r.get("url") or f"https://{r['subdomain']}.vercel.run"
        raise ValueError(f"No route for port {port}")

    def get_command(self, cmd_id: str) -> Command:
        resp = self.client.get_command(sandbox_id=self.sandbox.id, cmd_id=cmd_id)
        assert isinstance(resp, CommandResponse)
        return Command(client=self.client, sandbox_id=self.sandbox.id, cmd=resp.command)

    def run_command(
        self,
        cmd: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        sudo: bool = False,
    ) -> CommandFinished:
        command_response = self.client.run_command(
            sandbox_id=self.sandbox.id,
            command=cmd,
            args=args or [],
            cwd=cwd,
            env=env or {},
            sudo=sudo,
        )
        command = Command(
            client=self.client, sandbox_id=self.sandbox.id, cmd=command_response.command
        )
        return command.wait()

    def run_command_detached(
        self,
        cmd: str,
        args: list[str] | None = None,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        sudo: bool = False,
    ) -> Command:
        command_response = self.client.run_command(
            sandbox_id=self.sandbox.id,
            command=cmd,
            args=args or [],
            cwd=cwd,
            env=env or {},
            sudo=sudo,
        )
        return Command(
            client=self.client, sandbox_id=self.sandbox.id, cmd=command_response.command
        )

    def mk_dir(self, path: str, *, cwd: str | None = None) -> None:
        self.client.mk_dir(sandbox_id=self.sandbox.id, path=path, cwd=cwd)

    def read_file(self, path: str, *, cwd: str | None = None) -> bytes | None:
        return self.client.read_file(sandbox_id=self.sandbox.id, path=path, cwd=cwd)

    def write_files(self, files: list[WriteFile]) -> None:
        self.client.write_files(
            sandbox_id=self.sandbox.id,
            cwd=self.sandbox.cwd,
            extract_dir="/",
            files=files,
        )

    def stop(self) -> None:
        self.client.stop_sandbox(sandbox_id=self.sandbox.id)

    def __enter__(self) -> "Sandbox":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.stop()
        except Exception:
            pass
        self.client.close()
