import numpy as np
import main

def test_field_shapes():
    assert main.Ez.shape == (main.Nx, main.Ny)
    assert main.Hx.shape == (main.Nx, main.Ny)
    assert main.Hy.shape == (main.Nx, main.Ny)

def test_field_initialization():
    assert np.all(main.Ez == 0)
    assert np.all(main.Hx == 0)
    assert np.all(main.Hy == 0)

def test_coefficients_positive():
    assert main.ceze > 0
    assert main.cezh > 0
    assert main.chxh > 0
    assert main.chxe > 0