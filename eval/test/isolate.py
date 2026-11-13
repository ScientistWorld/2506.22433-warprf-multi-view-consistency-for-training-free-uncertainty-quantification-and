"""Run one method command with the hidden test material out of its reach.

Usage (by the evaluator only):
    python -I isolate.py --hide PATH [--hide PATH ...] -- CMD [ARGS ...]

The command runs in fresh Linux user, mount and PID namespaces:

* every ``--hide`` directory is covered by an empty read-only tmpfs and every
  ``--hide`` file by ``/dev/null`` (the evaluator hides ``eval/test/`` and the
  git object store, which also holds the test files);
* the command gets a fresh ``/proc`` showing only its own processes, so it cannot
  reach the evaluator's view of the filesystem through ``/proc/<pid>/root``;
* it runs as the invoking user with no capabilities, so it cannot unmount the
  covers; when it exits, every process it started is killed with it.

Prints ``ISOLATION_UNAVAILABLE: <reason>`` on stderr and exits with code 97 if
the kernel refuses to create the namespaces (the evaluator then falls back to
file-permission locking).
"""

from __future__ import annotations

import ctypes
import os
import signal
import sys

CLONE_NEWNS = 0x00020000
CLONE_NEWUSER = 0x10000000
CLONE_NEWPID = 0x20000000
MS_RDONLY = 0x1
MS_NOSUID = 0x2
MS_NODEV = 0x4
MS_NOEXEC = 0x8
MS_BIND = 0x1000
MS_REC = 0x4000
MS_PRIVATE = 1 << 18
PR_SET_PDEATHSIG = 1
UNAVAILABLE = 97

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]


def _enc(s):
    return None if s is None else s.encode()


def _mount(src, target, fstype, flags, data=None) -> None:
    if _libc.mount(_enc(src), _enc(target), _enc(fstype), flags, _enc(data)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mount {target}: {os.strerror(err)}")


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="ascii") as f:
        f.write(text)


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        sys.exit("usage: isolate.py --hide PATH ... -- CMD ...")
    sep = argv.index("--")
    opts, cmd = argv[:sep], argv[sep + 1 :]
    hide = [opts[i + 1] for i in range(len(opts) - 1) if opts[i] == "--hide"]
    if not cmd:
        sys.exit("isolate.py: empty command")
    uid, gid = os.getuid(), os.getgid()
    try:
        if _libc.unshare(CLONE_NEWUSER | CLONE_NEWNS | CLONE_NEWPID) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"unshare: {os.strerror(err)}")
        _write("/proc/self/setgroups", "deny")
        _write("/proc/self/uid_map", f"{uid} {uid} 1")
        _write("/proc/self/gid_map", f"{gid} {gid} 1")
        _mount(None, "/", None, MS_REC | MS_PRIVATE)
        for path in hide:
            if os.path.isdir(path) and not os.path.islink(path):
                _mount("none", path, "tmpfs", MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC, "size=4k,mode=0555")
            elif os.path.exists(path):
                _mount("/dev/null", path, None, MS_BIND)
    except OSError as exc:
        print(f"ISOLATION_UNAVAILABLE: {exc}", file=sys.stderr, flush=True)
        sys.exit(UNAVAILABLE)

    pid = os.fork()
    if pid == 0:  # first process of the new PID namespace
        try:
            _libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
            try:
                _mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
            except OSError as exc:
                print(f"ISOLATION_UNAVAILABLE: {exc}", file=sys.stderr, flush=True)
                os._exit(UNAVAILABLE)
            os.execvp(cmd[0], cmd)
        except BaseException as exc:  # noqa: BLE001 - report and exit, never return into the launcher
            print(f"isolate.py: cannot run {cmd[0]!r}: {exc}", file=sys.stderr, flush=True)
            os._exit(127)

    def _forward(signum, _frame):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _forward)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        sys.exit(128 + os.WTERMSIG(status))
    sys.exit(os.WEXITSTATUS(status))


if __name__ == "__main__":
    main()
