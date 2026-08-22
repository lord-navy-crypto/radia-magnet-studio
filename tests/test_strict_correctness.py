import sys, math, tempfile, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tests"))

from fake_radia import FakeRadia
from devices.factory import build_device
from solver.pipeline import solve_model, RelaxationConvergenceError, sample_on_axis
from analysis.geometry_bounds import geometry_field_range, union_field_range
from analysis.metrics import analyze, electron_phase_error
from calibration.target_b0 import calibrate_br
from export.exporters import hdf5_bytes

BASE = {
    "period_mm":50.0,"periods":4,"gap_mm":12.0,"blocks_per_period":4,
    "block_width_mm":10.0,"block_height_mm":10.0,"longitudinal_fill":0.9,
    "br_t":1.2,"material_mode":"Fixed remanence","mu_parallel":1.05,
    "mu_perpendicular":1.05,"segmentation":(1,1,1),"ellipticity":0.5,
    "apple_phase_deg":90.0,"apple_shift_mode":"Antiparallel",
    "errors_enabled":False,"field_error_pct":1.0,"longitudinal_error_mm":0.05,
    "transverse_error_mm":0.05,"angle_error_deg":0.5,"gap_asymmetry_mm":0.1,
    "bank_imbalance_pct":1.0,"error_seed":12345,
}

def run():
    # 1. Convergence pass and fail gates.
    p=dict(BASE); p["material_mode"]="Linear NdFeB + relaxation"
    rad=FakeRadia()
    m=build_device(rad,"Planar",p)
    info=solve_model(rad,m,relax=True,precision=1e-4,max_iter=1000,method=4)
    assert info[0] < 1e-4 and info[3] < 1000

    rad_fail=FakeRadia(fail_relax=True)
    m_fail=build_device(rad_fail,"Planar",p)
    try:
        solve_model(rad_fail,m_fail,relax=True,precision=1e-4,max_iter=1000,method=4)
        raise AssertionError("non-converged relaxation was not rejected")
    except RelaxationConvergenceError:
        pass

    # 2. Geometry field bounds include actual APPLE-II longitudinal shift + margin.
    p=dict(BASE); p["apple_phase_deg"]=180.0
    rad=FakeRadia()
    apple=build_device(rad,"APPLE-II",p)
    raw_lo=min(b["center"][2]-b["size"][2]/2 for b in apple["blocks"])
    raw_hi=max(b["center"][2]+b["size"][2]/2 for b in apple["blocks"])
    lo,hi=geometry_field_range(apple["blocks"],p["period_mm"],1.0)
    assert abs(lo-(raw_lo-50.0))<1e-9 and abs(hi-(raw_hi+50.0))<1e-9
    assert hi-lo > p["period_mm"]*p["periods"]

    # 3. K definitions: circular synthetic field should not report sqrt(2) as K_peak.
    lam=50.0
    z=np.linspace(-500,500,4001)
    B0=0.2
    phi=2*np.pi*z/lam
    B=np.column_stack([B0*np.cos(phi),B0*np.sin(phi),np.zeros_like(z)])
    metrics=analyze(z,B,lam,3.0)
    expected=0.934*B0*(lam/10.0)
    assert abs(metrics["K_peak"]-expected)/expected < 1e-3
    assert abs(metrics["K_vector_norm"]-math.sqrt(2)*expected)/(math.sqrt(2)*expected) < 2e-3
    assert abs(metrics["resonance_K2_over_2"]-expected**2)/expected**2 < 3e-3

    # 4. Electron phase is a distinct finite metric.
    ep=electron_phase_error(z,B,lam,3.0,exclude_end_periods=2.5)
    assert np.isfinite(ep["rms_deg"])
    assert np.isfinite(ep["radiation_wavelength_m"])
    assert len(ep["phase_error_deg"]) > 5
    assert "zero_crossing_field_phase_rms_deg" in metrics
    assert "electron_phase_error_rms_deg" in metrics

    # 5. HDF5 exporter returns real HDF5 bytes and does not silently drop unsupported metadata.
    params=dict(BASE); params["unsupported_for_h5"]={"x":1}
    metrics2=dict(metrics)
    blob=hdf5_bytes(z,B,params,metrics2)
    assert blob[:8] == b"\x89HDF\r\n\x1a\n"
    import h5py
    fd,path=tempfile.mkstemp(suffix=".h5"); os.close(fd)
    try:
        Path(path).write_bytes(blob)
        with h5py.File(path,"r") as f:
            assert "export_skipped_items" in f
            vals=[v.decode() if isinstance(v,bytes) else str(v) for v in f["export_skipped_items"][:]]
            assert any("unsupported_for_h5" in v for v in vals)
    finally:
        os.unlink(path)

    # 6. B0 central-period calibration verifies final field.
    rad=FakeRadia()
    p=dict(BASE)
    br,hist=calibrate_br(
        rad,"Planar",p,0.24,mode="Central-period peak B⊥",
        relax=False,samples=201
    )
    assert hist and br>0 and abs(hist[-1]["B0_T"]-0.24)<0.02
    assert hist[-1]["mode"]=="Central-period peak B⊥"

    print("ALL STRICT CORRECTNESS TESTS PASSED")

if __name__=="__main__":
    run()
