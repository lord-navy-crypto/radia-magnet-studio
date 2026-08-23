from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run():
    studio = (ROOT / "app" / "studio.py").read_text(encoding="utf-8")
    studies = (ROOT / "app" / "advanced_analysis.py").read_text(encoding="utf-8")
    launcher = (ROOT / "START_HERE_RADIA_MAGNET_STUDIO.command").read_text(encoding="utf-8")

    # Original magnetic-field generator and inspector must remain the primary path.
    for marker in (
        "Build + Solve + Analyze", "sample_on_axis", "sample_slice_xz",
        "sample_slice_yz", "sample_3d", "geometry_view", "trajectory_plot",
        "fieldmap3d_csv_bytes", "research_package_bytes", "calibrate_br",
        "ideal_error_field_plot", "electron_phase_plot",
    ):
        assert marker in studio, f"Original studio feature missing: {marker}"

    # Advanced studies must be reachable from the same app and reuse a completed run.
    assert "render_advanced_analysis" in studio
    assert 'st.session_state["magnet_study_source"]' in studio
    assert "Import preset (.json)" in studio
    assert "Export current preset (.json)" in studio
    assert "Download realized preset (.json)" in studio
    assert "Connected to the magnetic field generated above" in studies
    assert "Continue analysis" in studies
    assert "@st.fragment" in studies
    assert "app/studio.py" in launcher
    assert "st.set_page_config" not in studies
    assert not (ROOT / "app" / "studies.py").exists()
    assert not (ROOT / "START_RADIA_STUDIES.command").exists()
    print("UNIFIED UI AND ORIGINAL-FEATURE RETENTION TEST PASSED")


if __name__ == "__main__":
    run()
