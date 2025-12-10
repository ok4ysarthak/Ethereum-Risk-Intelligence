from __future__ import annotations

from typing import Literal, TypedDict
from pydantic import BaseModel


class Sandbox(BaseModel):
    id: str
    memory: int
    vcpus: int
    region: str
    runtime: str
    timeout: int
    status: Literal["pending", "running", "stopping", "stopped", "failed"]
    requestedAt: int
    startedAt: int | None = None
    requestedStopAt: int | None = None
    stoppedAt: int | None = None
    duration: int | None = None
    createdAt: int
    cwd: str
    updatedAt: int


class SandboxRoute(BaseModel):
    url: str
    subdomain: str
    port: int


class Pagination(BaseModel):
    count: int
    next: int | None = None
    prev: int | None = None


class Command(BaseModel):
    id: str
    name: str
    args: list[str]
    cwd: str
    sandboxId: str
    exitCode: int | None = None
    startedAt: int


class CommandFinished(Command):
    exitCode: int


class SandboxResponse(BaseModel):
    sandbox: Sandbox


class SandboxAndRoutesResponse(SandboxResponse):
    routes: list[SandboxRoute]


class CommandResponse(BaseModel):
    command: Command


class CommandFinishedResponse(BaseModel):
    command: CommandFinished


class EmptyResponse(BaseModel):
    pass


class LogLine(BaseModel):
    stream: Literal["stdout", "stderr"]
    data: str


class SandboxesResponse(BaseModel):
    sandboxes: list[Sandbox]
    pagination: Pagination


class WriteFile(TypedDict):
    path: str
    content: bytes
