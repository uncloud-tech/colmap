# SPDX-License-Identifier: BSD-3-Clause
"""Selectively reused canonical current-owned-shape logical inventory.

Extracted from reviewed colmap_reconstruction/model.py (PR16, unmerged),
originally photogram memory_model.py 0b84800e. No qualification is transferred:
this helper matches native LOWER BOUNDS, not a complete physical forecast.
Neither the optimized PHOTO profile nor resident current-phase-v1 eligibility
is expanded by this shape-only arithmetic. See NOTICE for reconciliation.
"""


def exact_int(value, minimum=0, maximum=(1 << 64) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('expected exact integer in supported native range')
    return value


def _add(a, b):
    a, b = exact_int(a), exact_int(b)
    if b > (1 << 64) - 1 - a:
        raise OverflowError('Resident bank byte sum overflow')
    return a + b


def _mul(a, b):
    a, b = exact_int(a), exact_int(b)
    if a and b > ((1 << 64) - 1) // a:
        raise OverflowError('Resident bank byte product overflow')
    return a * b


def margin_bytes(raw):
    """Canonical integer ceil(1.03*raw). Producer never applies this to items."""
    raw = exact_int(raw)
    return _add(raw, _add(_mul(raw // 100, 3), (_mul(raw % 100, 3) + 99) // 100))


def current_owned_shape_minimum(width, height, sources):
    """Raw ResidentPrivateLogicalMinimum(..., include_owner_normal=false).

    Shape body reused from canonical estimate_owned_private_job's explicit
    current-phase-v1 branch, without resident identity resolution or margins.
    Conservative retained mask matches the native default for BOTH phases.
    """
    w, h, s = (exact_int(v, 1, (1 << 31) - 1)
               for v in (width, height, sources))
    a = _mul(w, h)
    if max(w, h) > ((1 << 31) - 1) // 4:
        raise OverflowError('current selection scratch pitch exceeds native int ABI')
    boundary = (w if h == 1 else h if w == 1 else
                _add(_mul(2, w), _mul(2, h)) - 4)
    if boundary > ((1 << 31) - 1) // 48:
        raise OverflowError('current PRNG boundary pitch exceeds native int ABI')
    common = _add(_mul(48, boundary), _mul(_mul(8, max(w, h)), s))
    initialization = _add(_mul(_add(73, _mul(12, s)), a), common)
    steady = _add(_mul(_add(50, _mul(21, s)), a), common)
    rotation = max(initialization, steady)
    device_raw = _add(_add(_add(rotation, _mul(a, 2)), _mul(692, s)), 32)
    return _mul(a, 24), device_raw
