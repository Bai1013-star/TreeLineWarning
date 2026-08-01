"""Start the TreeLineWarning web service with explicit local-runtime controls."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

DATABASE_PATH_ENV = "TREELINE_WARNING_DATABASE_PATH"
TEMPORAL_MODEL_PATH_ENV = "TREELINE_WARNING_TEMPORAL_MODEL_PATH"
SPECIES_GROWTH_MODEL_PATH_ENV = "TREELINE_WARNING_SPECIES_GROWTH_MODEL_PATH"
TRAINING_OUTPUT_DIR_ENV = "TREELINE_WARNING_TRAINING_OUTPUT_DIR"
WEB_INDEX_PATH_ENV = "TREELINE_WARNING_WEB_INDEX_PATH"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_PID = 2_147_483_647


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_python(project_root: Path) -> Path:
    if os.name == "nt":
        virtual_environment_python = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        virtual_environment_python = project_root / ".venv" / "bin" / "python"
    if virtual_environment_python.is_file():
        return virtual_environment_python
    current_python = Path(sys.executable)
    if current_python.is_file():
        return current_python
    raise RuntimeError("未找到项目 .venv 或当前可执行 Python。")


def _configured_value(
    value: str | None, environment: dict[str, str], variable: str
) -> tuple[str | None, str]:
    if value:
        return value, "argument"
    inherited = environment.get(variable)
    if inherited:
        return inherited, "environment"
    return None, "unconfigured"


def _configuration_summary(environment: dict[str, str], args: argparse.Namespace) -> None:
    configurations = (
        ("database", args.database_path, DATABASE_PATH_ENV),
        ("temporal model", args.temporal_model_path, TEMPORAL_MODEL_PATH_ENV),
        (
            "species growth model",
            args.species_growth_model_path,
            SPECIES_GROWTH_MODEL_PATH_ENV,
        ),
        ("training output", args.training_output_dir, TRAINING_OUTPUT_DIR_ENV),
        ("web index", args.web_index_path, WEB_INDEX_PATH_ENV),
    )
    for label, value, variable in configurations:
        configured, source = _configured_value(value, environment, variable)
        detail = Path(configured).name if configured else "unconfigured"
        print(f"{label}: {source} ({detail})")


def _uvicorn_command(python: Path, args: argparse.Namespace) -> list[str]:
    target = "api.prediction_api:create_app" if args.factory else "api.prediction_api:app"
    command = [str(python), "-m", "uvicorn", target]
    if args.factory:
        command.append("--factory")
    return [*command, "--host", args.host, "--port", str(args.port)]


def _masked_command_summary(command: list[str], args: argparse.Namespace) -> str:
    target = "factory" if args.factory else "app"
    return f"python -m uvicorn ({target}) --host {args.host} --port {args.port}"


def _is_own_uvicorn_process(process: psutil.Process, port: int) -> bool:
    try:
        command_line = process.cmdline()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return False
    joined = " ".join(command_line).lower()
    if "uvicorn" not in joined or "api.prediction_api" not in joined:
        return False
    return any(
        part == "--port" and index + 1 < len(command_line)
        and command_line[index + 1] == str(port)
        for index, part in enumerate(command_line)
    )


def _listening_process(port: int) -> int | None:
    try:
        connections = psutil.net_connections(kind="inet")
    except psutil.Error:
        return None
    for connection in connections:
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port:
            return connection.pid
    return None


def _paths_for_background(log_dir: Path, port: int) -> tuple[Path, Path, Path]:
    prefix = f"treeline-warning-{port}"
    return (
        log_dir / f"{prefix}.stdout.log",
        log_dir / f"{prefix}.stderr.log",
        log_dir / f"{prefix}.pid",
    )


def _inspect_pid_file(pid_path: Path, port: int) -> tuple[str, psutil.Process | None]:
    if not pid_path.is_file():
        return "absent", None
    try:
        content = pid_path.read_text(encoding="utf-8").strip()
        pid = int(content)
    except (OSError, ValueError):
        return "invalid", None
    if pid <= 0 or pid > MAX_PID:
        return "invalid", None
    try:
        process = psutil.Process(pid)
        is_running = process.is_running()
    except psutil.NoSuchProcess:
        try:
            pid_path.unlink()
        except OSError:
            return "invalid", None
        return "stale", None
    except psutil.Error:
        return "foreign", None
    if not is_running:
        try:
            pid_path.unlink()
        except OSError:
            return "invalid", None
        return "stale", None
    if _is_own_uvicorn_process(process, port):
        return "own", process
    return "foreign", None


def _write_pid(pid_path: Path, pid: int) -> None:
    with pid_path.open("x", encoding="utf-8") as pid_file:
        pid_file.write(str(pid))


def _terminate_started_process(process: subprocess.Popen[bytes]) -> None:
    try:
        process.terminate()
        process.wait(timeout=5)
    except (OSError, psutil.Error, subprocess.TimeoutExpired):
        return


def _stop_background_instance(log_dir: Path, port: int) -> int:
    _, _, pid_path = _paths_for_background(log_dir, port)
    if not pid_path.is_file():
        print("未找到受管理的后台实例。")
        return 0
    state, process = _inspect_pid_file(pid_path, port)
    if state == "invalid":
        print("PID file is invalid; inspect it manually before stopping a service.")
        return 1
    if state == "foreign":
        print("PID file is owned by another running process; refusing to stop it.")
        return 1
    if process is None:
        if pid_path.exists():
            print("PID 文件不属于当前项目实例，已拒绝停止。")
            return 1
        print("已移除陈旧 PID 文件。")
        return 0
    try:
        process.terminate()
        process.wait(timeout=5)
    except psutil.TimeoutExpired:
        print("后台实例未在限定时间内停止；未执行强制终止。")
        return 1
    except psutil.Error:
        print("停止后台实例失败。")
        return 1
    pid_path.unlink(missing_ok=True)
    print("后台实例已停止。")
    return 0


def _validate_arguments(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("端口必须在 1–65535 之间。")
    if args.host not in LOCAL_HOSTS and not args.allow_external:
        raise ValueError("非本地绑定必须同时显式提供 --allow-external。")
    if args.background and not args.log_dir:
        raise ValueError("--background 必须同时提供 --log-dir。")
    if args.log_dir and not (args.background or args.stop):
        raise ValueError("--log-dir 仅适用于 --background 或 --stop。")
    if args.stop and not args.log_dir:
        raise ValueError("--stop 仅用于受管理后台实例，必须提供 --log-dir。")
    if args.stop and args.foreground:
        raise ValueError("--stop 不能与 --foreground 一起使用。")


def _environment_for(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    overrides = {
        DATABASE_PATH_ENV: args.database_path,
        TEMPORAL_MODEL_PATH_ENV: args.temporal_model_path,
        SPECIES_GROWTH_MODEL_PATH_ENV: args.species_growth_model_path,
        TRAINING_OUTPUT_DIR_ENV: args.training_output_dir,
        WEB_INDEX_PATH_ENV: args.web_index_path,
    }
    for variable, value in overrides.items():
        if value:
            environment[variable] = value
    return environment


def _start_foreground(command: list[str], project_root: Path, environment: dict[str, str]) -> int:
    try:
        return subprocess.run(command, cwd=project_root, env=environment).returncode
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        raise RuntimeError("无法启动 Uvicorn。") from exc


def _start_background(
    command: list[str], project_root: Path, environment: dict[str, str], log_dir: Path, port: int
) -> int:
    stdout_path, stderr_path, pid_path = _paths_for_background(log_dir, port)
    state, managed_process = _inspect_pid_file(pid_path, port)
    if state == "own" and managed_process is not None:
        print("受管理的后台实例已在运行。")
        return 0
    if state == "foreign":
        print("PID file is owned by another running process; refusing to overwrite it.")
        return 1
    if state == "invalid":
        print("PID file is invalid; inspect it manually before starting a service.")
        return 1
    log_dir.mkdir(parents=True, exist_ok=True)
    owner_pid = _listening_process(port)
    if owner_pid is not None:
        print("端口已被其他进程占用；不会停止未知进程。")
        return 1
    creation_flags = 0
    start_new_session = False
    if os.name == "nt":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        start_new_session = True
    process: subprocess.Popen[bytes] | None = None
    try:
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=True,
                creationflags=creation_flags,
                start_new_session=start_new_session,
            )
        time.sleep(0.2)
        if process.poll() is not None:
            print(f"后台启动失败；请检查 {stderr_path.name}。")
            return 1
        try:
            _write_pid(pid_path, process.pid)
        except FileExistsError:
            _terminate_started_process(process)
            print("PID file appeared during startup; stopped this new process safely.")
            return 1
    except OSError as exc:
        if process is not None:
            _terminate_started_process(process)
        raise RuntimeError(f"后台启动失败；请检查 {stderr_path.name}。") from exc
    print(f"后台实例已启动；PID 已记录到 {pid_path.name}。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start TreeLineWarning safely.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--factory", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--foreground", action="store_true")
    mode.add_argument("--background", action="store_true")
    parser.add_argument("--allow-external", action="store_true")
    parser.add_argument("--database-path")
    parser.add_argument("--temporal-model-path")
    parser.add_argument("--species-growth-model-path")
    parser.add_argument("--training-output-dir")
    parser.add_argument("--web-index-path")
    parser.add_argument("--log-dir")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_arguments(args)
        project_root = _project_root()
        python = _resolve_python(project_root)
        environment = _environment_for(args)
        command = _uvicorn_command(python, args)
        if args.dry_run:
            python_source = (
                "project .venv" if ".venv" in python.parts else "current interpreter"
            )
            print("dry-run: no service, log, PID, database, or model files will be created.")
            print(f"python: {python.name} ({python_source})")
            print(f"mode: {'factory' if args.factory else 'app'}")
            print(f"binding: {args.host}:{args.port}")
            print(f"execution: {'background' if args.background else 'foreground'}")
            _configuration_summary(environment, args)
            print(f"command: {_masked_command_summary(command, args)}")
            return 0
        if args.stop:
            return _stop_background_instance(Path(args.log_dir), args.port)
        if args.background:
            return _start_background(
                command, project_root, environment, Path(args.log_dir), args.port
            )
        owner_pid = _listening_process(args.port)
        if owner_pid is not None:
            print("端口已被其他进程占用；不会停止未知进程。")
            return 1
        return _start_foreground(command, project_root, environment)
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
