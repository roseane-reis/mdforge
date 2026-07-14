"""Tests for the native CHARMM/NAMD DCD reader/writer."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from mdforge.formats.dcd import read_dcd, write_dcd


def test_dcd_roundtrip_with_box(tmp_path):
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 20, size=(4, 7, 3))
    box = np.tile([24.6, 24.6, 24.6, 90.0, 90.0, 90.0], (4, 1))
    p = write_dcd(coords, box=box, path=tmp_path / "t.dcd")
    tr = read_dcd(p)
    assert tr.n_frames == 4
    assert tr.n_atoms == 7
    assert np.allclose(tr.coordinates, coords, atol=1e-4)  # float32 storage
    assert np.allclose(tr.box[:, :3], box[:, :3], atol=1e-4)
    assert np.allclose(tr.box[:, 3:], 90.0, atol=1e-3)


def test_dcd_roundtrip_no_box(tmp_path):
    coords = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    p = write_dcd(coords, path=tmp_path / "nobox.dcd")
    tr = read_dcd(p)
    assert tr.box is None
    assert np.allclose(tr.coordinates, coords, atol=1e-4)


def test_dcd_triclinic_angles(tmp_path):
    coords = np.zeros((1, 4, 3))
    box = np.array([[20.0, 22.0, 24.0, 80.0, 100.0, 95.0]])
    p = write_dcd(coords, box=box, path=tmp_path / "tri.dcd")
    tr = read_dcd(p)
    assert np.allclose(tr.box[0, :3], [20.0, 22.0, 24.0], atol=1e-4)
    assert np.allclose(tr.box[0, 3:], [80.0, 100.0, 95.0], atol=1e-2)


def test_dcd_max_frames(tmp_path):
    coords = np.zeros((10, 3, 3))
    p = write_dcd(coords, path=tmp_path / "m.dcd")
    assert read_dcd(p, max_frames=3).n_frames == 3


def _labelled(n, natom=3):
    """n frames whose every coordinate equals the frame index (easy to identify)."""
    return np.stack([np.full((natom, 3), float(i)) for i in range(n)])


def test_dcd_stride(tmp_path):
    p = write_dcd(_labelled(10), path=tmp_path / "s.dcd")
    tr = read_dcd(p, stride=2)                       # frames 0,2,4,6,8
    assert tr.n_frames == 5
    assert np.allclose([c[0, 0] for c in tr.coordinates], [0, 2, 4, 6, 8])


def test_dcd_stride_with_max_frames(tmp_path):
    p = write_dcd(_labelled(10), path=tmp_path / "sm.dcd")
    tr = read_dcd(p, stride=2, max_frames=3)          # frames 0,2,4
    assert tr.n_frames == 3
    assert np.allclose([c[0, 0] for c in tr.coordinates], [0, 2, 4])


def test_dcd_stride_keeps_box_aligned(tmp_path):
    coords = _labelled(6, natom=2)
    box = np.stack([[10.0 + i, 10.0 + i, 10.0 + i, 90.0, 90.0, 90.0] for i in range(6)])
    p = write_dcd(coords, box=box, path=tmp_path / "sb.dcd")
    tr = read_dcd(p, stride=3)                        # frames 0,3
    assert tr.n_frames == 2
    assert np.allclose([c[0, 0] for c in tr.coordinates], [0, 3])
    assert np.allclose(tr.box[:, 0], [10.0, 13.0])    # box strided with coords


def test_dcd_nset_zero_reads_to_eof(tmp_path):
    # Streaming writers (e.g. Tinker9) leave NSET=0; the reader must still read all.
    p = write_dcd(_labelled(5), path=tmp_path / "z.dcd")
    raw = bytearray(p.read_bytes())
    struct.pack_into("<i", raw, 8, 0)                 # NSET word (after marker + 'CORD')
    p.write_bytes(raw)
    assert read_dcd(p).n_frames == 5


def test_dcd_stride_must_be_positive(tmp_path):
    p = write_dcd(_labelled(2), path=tmp_path / "b.dcd")
    with pytest.raises(ValueError):
        read_dcd(p, stride=0)


def test_dcd_bigendian_marker_rejected(tmp_path):
    # A file whose leading marker is neither 84 (LE) nor 84 (BE) is not a DCD.
    bad = tmp_path / "bad.dcd"
    bad.write_bytes(struct.pack("<i", 12) + b"x" * 12 + struct.pack("<i", 12))
    with pytest.raises(ValueError):
        read_dcd(bad)
