import importlib.util
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))

from fake_radia import FakeRadia
from devices.factory import build_device
from solver.pipeline import solve_model, sample_on_axis, sample_3d
from analysis.metrics import analyze, compare_metrics, classify_k
from calibration.target_b0 import calibrate_br
from export.exporters import fieldmap3d_csv_bytes

BASE={
    "period_mm":50.0,"periods":2,"gap_mm":12.0,"blocks_per_period":4,
    "block_width_mm":10.0,"block_height_mm":10.0,"longitudinal_fill":0.9,
    "br_t":1.2,"material_mode":"Fixed remanence","mu_parallel":1.05,
    "mu_perpendicular":1.05,"segmentation":(1,1,1),"ellipticity":0.5,
    "apple_phase_deg":90.0,"apple_shift_mode":"Antiparallel",
    "errors_enabled":False,"field_error_pct":1.0,"longitudinal_error_mm":0.05,
    "transverse_error_mm":0.05,"angle_error_deg":0.5,"gap_asymmetry_mm":0.1,
    "bank_imbalance_pct":1.0,"error_seed":12345,
}

def run():
    for kind in ["Planar","Helical","Elliptical","APPLE-II","Wiggler"]:
        rad=FakeRadia(); p=dict(BASE)
        if kind in ("Helical","Elliptical"): p["blocks_per_period"]=8
        model=build_device(rad,kind,p)
        assert model["blocks"], kind
        assert all("center" in b and "axis" in b for b in model["blocks"])
        z=np.linspace(-50,50,101); B=sample_on_axis(rad,model["obj"],z)
        m=analyze(z,B,50.0,3.0)
        assert "K_peak" in m and classify_k(m["K_peak"])
        B3=sample_3d(rad,model["obj"],[-1,1],[-1,1],[-10,0,10])
        assert B3.shape==(3,2,2,3)
        csv=fieldmap3d_csv_bytes([-1,1],[-1,1],[-10,0,10],B3)
        assert b"x_m,y_m,z_m,Bx_T,By_T,Bz_T" in csv
        if importlib.util.find_spec("plotly") is not None:
            from visualization.plots import geometry_view
            fig=geometry_view(model["blocks"],max_blocks=100)
            assert len(fig.data)>0

    # Error model is deterministic and changes geometry/strength.
    p=dict(BASE); p["errors_enabled"]=True
    r1=FakeRadia(); a=build_device(r1,"Planar",p)
    r2=FakeRadia(); b=build_device(r2,"Planar",p)
    assert a["blocks"][0]["center"]==b["blocks"][0]["center"]
    assert a["blocks"][0]["center"]!=a["blocks"][0]["ideal_center"] or a["blocks"][0]["br_scale"]!=1.0

    # Ideal comparison.
    z=np.linspace(-50,50,101)
    Bi=sample_on_axis(r1,a["obj"],z); mi=analyze(z,Bi,50.0,3.0)
    cmp=compare_metrics(mi,mi)
    assert abs(cmp["K_peak"]["delta"])<1e-12

    # Material / segmentation / relaxation API.
    rad=FakeRadia(); p=dict(BASE)
    p["material_mode"]="Linear NdFeB + relaxation"; p["segmentation"]=(2,2,2)
    model=build_device(rad,"Planar",p)
    r=solve_model(rad,model,relax=True)
    assert rad.applied and rad.divided and rad.relaxed and len(r)==4

    # Target B0 calibration should move Br toward the requested synthetic field.
    rad=FakeRadia(); p=dict(BASE)
    br,hist=calibrate_br(rad,"Planar",p,0.24,relax=False,samples=101)
    assert hist and br>0 and abs(hist[-1]["B0_T"]-0.24)<0.02

    print("ALL CORE DEVICE MOCK TESTS PASSED")
    if importlib.util.find_spec("plotly") is None:
        print("SKIP visualization assertion: plotly is not installed")

if __name__=="__main__":
    run()
