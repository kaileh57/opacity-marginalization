"""Unit tests for the B4 cloud module + v sin i broadening (CPU, tiny grids)."""
import numpy as np
import pytest

from sculptor.forward_model.clouds import compute_cloud_tau, select_condensate
from sculptor.forward_model.spectrum import apply_vsini_broadening, apply_radial_velocity_shift


def _profile(n_layer=40, t_surface=900.0):
    p = np.logspace(-5, 2, n_layer)  # bar, top->bottom
    # monotonic T-P increasing with depth, crosses Na2S/MgSiO3 curves somewhere
    t = t_surface * (p / p[-1]) ** 0.12
    mmw = np.full(n_layer, 2.33)
    return p, t, mmw


def test_select_condensate_regimes():
    assert select_condensate(1600.0) == "MgSiO3"
    assert select_condensate(900.0) == "Na2S"
    assert select_condensate(400.0) == "H2O"


def test_cloud_tau_shape_finite_nonneg():
    p, t, mmw = _profile()
    nu = np.linspace(2000.0, 3500.0, 200)
    tau = compute_cloud_tau(p, t, mmw, gravity_cgs=10 ** 4.5, metallicity=0.0,
                            log_kzz=8.0, f_sed=2.0, sigma_c=2.0, nu_grid_cm1=nu,
                            condensates=("MgSiO3", "Na2S"))
    assert tau.shape == (len(p), len(nu))
    assert np.all(np.isfinite(tau))
    assert np.all(tau >= 0.0)


def test_cloud_increases_as_fsed_decreases():
    p, t, mmw = _profile()
    nu = np.linspace(2000.0, 3500.0, 100)
    kw = dict(pressures_bar=p, temperatures_K=t, mean_molecular_weight=mmw, gravity_cgs=10 ** 4.5,
              metallicity=0.0, log_kzz=8.0, sigma_c=2.0, nu_grid_cm1=nu, condensates=("MgSiO3", "Na2S"))
    thin = compute_cloud_tau(f_sed=8.0, **kw).sum()
    thick = compute_cloud_tau(f_sed=1.0, **kw).sum()
    assert thick > thin, "lower f_sed must give a thicker cloud"


def test_no_cloud_when_too_hot():
    # an all-hot profile never reaches the condensation curve -> zero cloud
    p = np.logspace(-5, 2, 40)
    t = np.full(40, 3500.0)
    mmw = np.full(40, 2.33)
    nu = np.linspace(2000.0, 3500.0, 50)
    tau = compute_cloud_tau(p, t, mmw, 10 ** 4.5, 0.0, 8.0, 2.0, 2.0, nu, condensates=("MgSiO3",))
    assert np.allclose(tau, 0.0)


def test_vsini_widens_and_identity():
    # fine ESLOG grid so per-pixel dv (~9 km/s) is below the test vsini (rotation is resolved)
    nu = np.exp(np.linspace(np.log(2000.0), np.log(3500.0), 20000))
    flux = np.ones_like(nu)
    flux[10000] = 0.0  # an absorption spike
    out = apply_vsini_broadening(nu, flux, 40.0)
    assert np.isfinite(out).all()
    # the spike should be shallower (spread) after broadening, and a neighbour pulled down
    assert out[10000] > flux[10000]
    assert out[10001] < 1.0
    # flux conserved in the interior (normalized kernel; edges have np.convolve zero-pad artifact)
    assert abs(out[100:-100].sum() - flux[100:-100].sum()) < 0.05
    # zero vsini = identity
    assert np.allclose(apply_vsini_broadening(nu, flux, 0.0), flux)


def test_rv_roundtrip():
    nu = np.exp(np.linspace(np.log(2000.0), np.log(3500.0), 2000))
    flux = 1.0 + 0.1 * np.sin(np.linspace(0, 40, 2000))
    shifted = apply_radial_velocity_shift(nu, flux, 30.0)
    back = apply_radial_velocity_shift(nu, shifted, -30.0)
    assert np.median(np.abs(back - flux)) < 0.02


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
