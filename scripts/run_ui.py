"""Launch the local FastAPI service and Streamlit dashboard together."""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-port', type=int, default=8000)
    parser.add_argument('--ui-port', type=int, default=8501)
    args = parser.parse_args()
    if args.api_port == args.ui_port or any(not 1 <= port <= 65535 for port in [args.api_port, args.ui_port]):
        parser.error('choose different API/UI ports between 1 and 65535')
    for port in [args.api_port, args.ui_port]:
        with socket.socket() as probe:
            # Permit immediate restart after the previous listener enters TIME_WAIT.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(('127.0.0.1', port))
            except OSError:
                parser.exit(1, f'Port {port} is already in use. Stop the previous application or choose --api-port / --ui-port.\n')
    processes = []
    try:
        environment = {**os.environ, 'HF_HUB_DISABLE_TELEMETRY': '1', 'TOKENIZERS_PARALLELISM': 'false',
                       'RAG_LAB_API_URL': f'http://127.0.0.1:{args.api_port}'}
        processes.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.api.main:app', '--host', '127.0.0.1', '--port', str(args.api_port)],
                                          cwd=PROJECT_ROOT, env=environment, start_new_session=True))
        deadline = time.monotonic() + 30
        while True:
            if processes[0].poll() is not None:
                raise RuntimeError('API startup failed; check the terminal output and .env settings')
            try:
                response = httpx.get(environment['RAG_LAB_API_URL'] + '/health', timeout=1, trust_env=False)
                if response.is_success and response.json().get('application') == 'rag-indexing-lab':
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('API did not become ready within 30 seconds')
            time.sleep(0.2)
        processes.append(subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app/ui/dashboard.py', '--server.address', '127.0.0.1',
                                          '--server.port', str(args.ui_port), '--server.headless', 'true', '--browser.gatherUsageStats', 'false'],
                                          cwd=PROJECT_ROOT, env=environment, start_new_session=True))
        print(f'Open the dashboard: http://127.0.0.1:{args.ui_port}\nAPI documentation: http://127.0.0.1:{args.api_port}/docs\nPress Ctrl+C to stop both services.', flush=True)
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError('A service stopped; check the terminal output')
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f'Launcher failed: {error}', file=sys.stderr)
        return 1
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
