# SPDX-License-Identifier: BSD-3-Clause
"""Local resource observations, not hard allocator or cgroup enforcement."""
from pathlib import Path
import ctypes
import os
import uuid


def cuda_device_uuid(ordinal):
    """Resolve a CUDA-visible ordinal using the same driver/environment as COLMAP.

    Management indices are not CUDA ordinals (even without visibility overrides).
    cuInit/device queries create no CUDA context and require no toolkit/runtime.
    """
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError('CUDA ordinal must be nonnegative')
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        initialize = driver.cuInit
        initialize.argtypes = [ctypes.c_uint]
        initialize.restype = ctypes.c_int
        get_device = driver.cuDeviceGet
        get_device.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        get_device.restype = ctypes.c_int
        get_uuid = driver.cuDeviceGetUuid_v2
        get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
        get_uuid.restype = ctypes.c_int
        device = ctypes.c_int()
        raw = (ctypes.c_ubyte * 16)()
        status = initialize(0)
        if status:
            raise ValueError('cuInit failed: %d' % status)
        status = get_device(ctypes.byref(device), ordinal)
        if status:
            raise ValueError('cuDeviceGet failed for CUDA ordinal %d: %d' % (ordinal, status))
        status = get_uuid(raw, device.value)
        if status or not any(raw):
            raise ValueError('CUDA device UUID unavailable: %d' % status)
        return 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))
    except (OSError, AttributeError) as error:
        raise ValueError('CUDA Driver API identity unavailable; no management-index fallback') from error


def admit_resources(*, host_available, device_available, host_budget, device_budget,
                    host_headroom, device_headroom):
    for side, available, budget, headroom in (
        ('host', host_available, host_budget, host_headroom),
        ('device', device_available, device_budget, device_headroom),
    ):
        if any(type(value) is not int or value < 0 for value in (available, budget, headroom)) or budget == 0:
            raise ValueError(side + ' resources must be positive byte budgets and nonnegative observations/headroom')
        # Dense forecasts already carry the one canonical 3% margin.
        if budget + headroom > available:
            raise ValueError(side + ' budget exceeds current available memory minus headroom')


def host_available_bytes():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0]) * 1024
    # Respect visible cgroup-v2 limits, including visible ancestors. This is an
    # observation only; hidden ancestors and concurrent allocations remain risks.
    membership = Path('/proc/self/cgroup').read_text().splitlines()
    for line in membership:
        if line.startswith('0::'):
            relative = line[3:]
            if '..' in Path(relative).parts:
                raise ValueError('invalid cgroup membership')
            root = Path('/sys/fs/cgroup')
            path = root / relative.lstrip('/')
            while path == root or root in path.parents:
                limit = path / 'memory.max'
                current = path / 'memory.current'
                if limit.is_file() and current.is_file():
                    raw = limit.read_text().strip()
                    if raw != 'max':
                        available = min(available, max(0, int(raw) - int(current.read_text())))
                if path == root:
                    break
                path = path.parent
    return available


def observe_resources(process, args):
    # Preserve caller visibility/order; the child uses this exact CUDA ordinal
    # environment. Refuse drift rather than querying a different physical GPU.
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER'):
        if process.environment.get(key) != os.environ.get(key):
            raise ValueError('CUDA visibility/order changed during reconstruction')
    device_uuid = cuda_device_uuid(args.gpu_index)
    # nvidia-smi is supplied by the local NVIDIA driver, not a provider API.
    output = process.run(['nvidia-smi', '-i', device_uuid,
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture=True)
    rows = output.decode('ascii').strip().splitlines()
    if len(rows) != 1:
        raise ValueError('expected one selected NVIDIA GPU memory observation')
    device = int(rows[0].strip()) * 1024**2
    host = host_available_bytes()
    admit_resources(host_available=host, device_available=device,
        host_budget=args.host_budget_bytes, device_budget=args.device_budget_bytes,
        host_headroom=args.host_headroom_bytes, device_headroom=args.device_headroom_bytes)
    return dict(host_available_bytes=host, device_available_bytes=device,
                cuda_ordinal=args.gpu_index, cuda_device_uuid=device_uuid,
                cuda_visible_devices=process.environment.get('CUDA_VISIBLE_DEVICES'),
                cuda_device_order=process.environment.get('CUDA_DEVICE_ORDER'))
