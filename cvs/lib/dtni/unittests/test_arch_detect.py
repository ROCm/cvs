"""Arch detection regex."""

import pytest

from cvs.lib.dtni.arch_detect import detect_arch


def test_detect_mi300x():
    out = "GPU[0]\t\t: Card Series: \t\tAMD Instinct MI300X"
    assert detect_arch(out) == "mi300x"


def test_detect_mi355x():
    assert detect_arch("AMD Instinct MI355X") == "mi355x"


def test_detect_350_distinguished_from_355():
    assert detect_arch("AMD Instinct MI350X") == "mi350x"
    assert detect_arch("AMD Instinct MI355X") == "mi355x"


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="could not detect"):
        detect_arch("NVIDIA H100")
