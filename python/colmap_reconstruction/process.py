# SPDX-License-Identifier: BSD-3-Clause
"""User-owned POSIX child sessions; not a sandbox or hard memory limiter."""
from pathlib import Path
import os
import signal
import subprocess
import time


class NativeFailure(RuntimeError):
    pass


class NativeProcess:
    def __init__(self, workspace, timeout_seconds, host_budget_bytes=None, threads=1):
        self.workspace = Path(workspace)
        self.deadline = time.monotonic() + timeout_seconds
        self.child = None
        self.sequence = 0
        self.host_budget_bytes = host_budget_bytes
        self.cancelled = False
        self.environment = dict(os.environ, OMP_NUM_THREADS=str(threads),
                                OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))

    def remaining(self):
        if self.cancelled:
            raise NativeFailure('reconstruction cancelled')
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise NativeFailure('reconstruction deadline expired')
        return remaining

    def run(self, argv, *, capture=False):
        """Retain stdout/stderr, propagate failures, drain our own process group.

        Native COLMAP children are trusted not to detach with setsid. SIGKILL of
        this Python controller cannot be handled; use an external service/cgroup
        for controller-death containment or hard host-memory enforcement.
        """
        self.remaining()
        self.sequence += 1
        stem = self.workspace / ('command-%03d' % self.sequence)
        with stem.with_suffix('.stdout').open('xb') as stdout, stem.with_suffix('.stderr').open('xb') as stderr:
            self.child = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, start_new_session=True, env=self.environment)
            try:
                while True:
                    try:
                        status = self.child.wait(timeout=min(self.remaining(), 0.2))
                        break
                    except subprocess.TimeoutExpired:
                        self.remaining()
                        # Direct COLMAP-process RSS only, not a hard tree limit.
                        if self.host_budget_bytes is not None:
                            try:
                                status_text = Path('/proc/%d/status' % self.child.pid).read_text()
                            except FileNotFoundError:
                                continue
                            for line in status_text.splitlines():
                                if line.startswith('VmRSS:') and int(line.split()[1]) * 1024 > self.host_budget_bytes:
                                    raise NativeFailure('observed native host RSS exceeds budget')
                if status:
                    raise NativeFailure('native exit %d: %s (see %s.stderr)' %
                                        (status, argv[1], stem))
            finally:
                try:
                    os.killpg(self.child.pid, signal.SIGTERM)
                    try:
                        self.child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    # Kill any remaining members, even after the leader exits.
                    os.killpg(self.child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.child.wait()
                self.child = None
        if capture:
            path = stem.with_suffix('.stdout')
            if path.stat().st_size > 64 * 1024 * 1024:
                raise NativeFailure('native response exceeds 64 MiB')
            return path.read_bytes()
        return b''
