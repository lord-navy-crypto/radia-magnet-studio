from __future__ import annotations
import math
import platform
import numpy as np
import pandas as pd
import streamlit as st

from radia_support import load_radia
from devices.factory import build_device
from solver.pipeline import solve_model
from solver.pipeline import sample_on_axis, sample_slice_xz, sample_slice_yz, sample_3d
from analysis.metrics import analyze, compare_metrics, classify_k
from analysis.geometry_bounds import union_field_range
from calibration.target_b0 import calibrate_br
from visualization.plots import (
    field_lines, slice_heatmap, field_cones, trajectory_plot,
    geometry_view, ideal_error_field_plot, electron_phase_plot
)
from export.exporters import (
    csv_bytes, json_bytes, hdf5_bytes, pdf_bytes, fieldmap3d_csv_bytes,
    research_package_bytes, validate_research_package_bytes,
)
from presets import (
    BUILTIN_PRESETS, build_preset, parse_preset, preset_json_bytes,
    runtime_to_widget_state,
)

st.set_page_config(page_title="RADIA Magnet Studio", layout="wide")
st.title("RADIA Magnet Studio 3.1 — Magnetic Field Generator & Inspector")
st.caption(
    "Build and inspect RADIA magnetic devices; solve and sample on-axis, 2D, and 3D fields; "
    "analyze trajectories, field integrals, harmonics, phase error, and polarization-related metrics; "
    "then export the same results for downstream trajectory and radiation tools."
)

with st.sidebar:
    st.header("Presets")
    builtin_name = st.selectbox("Built-in preset", list(BUILTIN_PRESETS), key="preset_builtin_name")
    if st.button("Load built-in preset", width="stretch"):
        runtime = parse_preset(BUILTIN_PRESETS[builtin_name])
        st.session_state.update(runtime_to_widget_state(runtime))
        st.session_state["preset_extensions"] = runtime["extensions"]
        st.session_state["preset_study_defaults"] = runtime["study_defaults"]
        st.rerun()
    uploaded_preset = st.file_uploader("Import preset (.json)", type=["json"], key="preset_upload")
    if uploaded_preset is not None:
        try:
            imported_runtime = parse_preset(uploaded_preset.getvalue())
            st.caption(
                f"Validated: {imported_runtime['metadata'].get('name', 'Unnamed preset')} "
                f"({imported_runtime['parameters']['device']})"
            )
            for warning in imported_runtime["warnings"]:
                st.warning(warning)
            if st.button("Apply imported preset", type="primary", width="stretch"):
                st.session_state.update(runtime_to_widget_state(imported_runtime))
                st.session_state["preset_extensions"] = imported_runtime["extensions"]
                st.session_state["preset_study_defaults"] = imported_runtime["study_defaults"]
                st.rerun()
        except Exception as exc:
            st.error(f"Preset rejected: {exc}")
    st.divider()
    st.header("Device")
    device = st.selectbox("Type", ["Planar", "Helical", "Elliptical", "APPLE-II", "Wiggler"], key="cfg_device")
    period_mm = st.number_input("Period λu (mm)", min_value=1.0, value=50.0, step=1.0, key="cfg_period_mm")
    periods = st.number_input("Number of periods", min_value=1, value=20, step=1, key="cfg_periods")
    gap_mm = st.number_input("Magnetic gap (mm)", min_value=0.5, value=12.0, step=0.5, key="cfg_gap_mm")
    blocks_per_period = st.selectbox(
        "Blocks per period", [4, 8, 12, 16],
        index=1 if device in ("Helical", "Elliptical") else 0, key="cfg_blocks_per_period"
    )

    st.header("Magnet blocks")
    block_width_mm = st.number_input("Block width x (mm)", min_value=0.1, value=10.0, key="cfg_block_width_mm")
    block_height_mm = st.number_input("Block height / radial thickness (mm)", min_value=0.1, value=10.0, key="cfg_block_height_mm")
    longitudinal_fill = st.slider("Longitudinal fill factor", 0.50, 0.99, 0.90, 0.01, key="cfg_longitudinal_fill")
    br_t = st.number_input("Remanent induction Br (T)", min_value=0.01, value=1.20, step=0.05, key="cfg_br_t")

    st.header("Target B0 calibration")
    target_b0_enabled = st.checkbox("Calibrate Br to target B0", value=False, key="cfg_target_b0_enabled")
    target_b0_t = st.number_input(
        "Target B0 (T)", min_value=0.001, value=0.15, step=0.01,
        disabled=not target_b0_enabled, key="cfg_target_b0_t"
    )
    b0_mode = st.selectbox(
        "B0 definition",
        ["Central-period peak B⊥", "Central 3-period peak B⊥", "Global peak B⊥"],
        disabled=not target_b0_enabled, key="cfg_b0_definition"
    )

    st.header("Material / solve")
    material_mode = st.selectbox("Magnet model", ["Fixed remanence", "Linear NdFeB + relaxation"], key="cfg_material_mode")
    mu_parallel = st.number_input("μr parallel", min_value=1.0, value=1.05, step=0.01, key="cfg_mu_parallel")
    mu_perpendicular = st.number_input("μr perpendicular", min_value=1.0, value=1.05, step=0.01, key="cfg_mu_perpendicular")
    seg_n = st.selectbox("Magnet subdivision", [1, 2, 3], index=0, key="cfg_seg_n")
    relax = (material_mode == "Linear NdFeB + relaxation") and st.checkbox(
        "Run RADIA relaxation", value=True, key="cfg_relax"
    )
    precision = st.number_input("Relaxation precision (T)", min_value=1e-7, value=1e-4, format="%.1e", key="cfg_precision")
    max_iter = st.number_input("Relaxation max iterations", min_value=1, value=1000, step=100, key="cfg_max_iter")

    st.header("Device-specific")
    ellipticity = st.slider("Ellipticity", 0.0, 1.0, 0.5, 0.01, disabled=device != "Elliptical", key="cfg_ellipticity")
    apple_phase_deg = st.slider(
        "APPLE-II magnetic row phase (deg)", -180.0, 180.0, 90.0, 1.0,
        disabled=device != "APPLE-II", key="cfg_apple_phase_deg"
    )
    apple_shift_mode = st.selectbox(
        "APPLE-II shift mode", ["Antiparallel", "Parallel"],
        disabled=device != "APPLE-II", key="cfg_apple_shift_mode"
    )
    if device == "APPLE-II":
        st.caption(
            "Prototype four-array geometry; phase is implemented as real longitudinal "
            "array displacement Δz = φ λu / 360°."
        )

    st.header("Manufacturing error model")
    errors_enabled = st.checkbox("Enable manufacturing errors", value=False, key="cfg_errors_enabled")
    field_error_pct = st.number_input("Field amplitude error σ (%)", min_value=0.0, value=1.0, step=0.1, disabled=not errors_enabled, key="cfg_field_error_pct")
    longitudinal_error_mm = st.number_input("Longitudinal position error σ (mm)", min_value=0.0, value=0.05, step=0.01, disabled=not errors_enabled, key="cfg_longitudinal_error_mm")
    transverse_error_mm = st.number_input("Transverse position error σ (mm)", min_value=0.0, value=0.05, step=0.01, disabled=not errors_enabled, key="cfg_transverse_error_mm")
    angle_error_deg = st.number_input("Magnetization angle error σ (deg)", min_value=0.0, value=0.5, step=0.1, disabled=not errors_enabled, key="cfg_angle_error_deg")
    gap_asymmetry_mm = st.number_input("Gap asymmetry (mm)", value=0.0, step=0.01, disabled=not errors_enabled, key="cfg_gap_asymmetry_mm")
    bank_imbalance_pct = st.number_input("Bank strength imbalance (%)", value=0.0, step=0.1, disabled=not errors_enabled, key="cfg_bank_imbalance_pct")
    error_seed = st.number_input("Random seed", min_value=0, value=12345, step=1, disabled=not errors_enabled, key="cfg_error_seed")
    compare_ideal = st.checkbox("Compute ideal-vs-error comparison", value=True, disabled=not errors_enabled, key="cfg_compare_ideal")

    st.header("Field sampling")
    axis_samples = st.slider("On-axis samples", 100, 4000, 1000, 100, key="cfg_axis_samples")
    field_margin_periods = st.number_input(
        "Longitudinal field margin (periods)", min_value=0.0, value=1.0, step=0.5,
        help="Added beyond the actual outer magnet-block edges for fringe-field integrals.", key="cfg_field_margin_periods"
    )
    electron_energy_GeV = st.number_input("Electron energy (GeV)", min_value=0.01, value=3.0, step=0.1, key="cfg_electron_energy_GeV")
    make_2d = st.checkbox("Calculate 2D field slice", value=True, key="cfg_make_2d")
    make_3d = st.checkbox("Calculate sparse 3D field map", value=True, key="cfg_make_3d")
    transverse_half_mm = st.number_input("Transverse map half-width (mm)", min_value=0.1, value=5.0, step=0.5, key="cfg_transverse_half_mm")
    geometry_limit = st.slider("Maximum blocks in 3D geometry viewer", 100, 1200, 600, 100, key="cfg_geometry_limit")

current_params = {
    "device": device, "period_mm": float(period_mm), "periods": int(periods),
    "gap_mm": float(gap_mm), "blocks_per_period": int(blocks_per_period),
    "block_width_mm": float(block_width_mm), "block_height_mm": float(block_height_mm),
    "longitudinal_fill": float(longitudinal_fill), "br_t": float(br_t),
    "material_mode": material_mode, "mu_parallel": float(mu_parallel),
    "mu_perpendicular": float(mu_perpendicular),
    "segmentation": (int(seg_n), int(seg_n), int(seg_n)),
    "ellipticity": float(ellipticity), "apple_phase_deg": float(apple_phase_deg),
    "apple_shift_mode": apple_shift_mode, "errors_enabled": bool(errors_enabled),
    "field_error_pct": float(field_error_pct),
    "longitudinal_error_mm": float(longitudinal_error_mm),
    "transverse_error_mm": float(transverse_error_mm),
    "angle_error_deg": float(angle_error_deg), "gap_asymmetry_mm": float(gap_asymmetry_mm),
    "bank_imbalance_pct": float(bank_imbalance_pct), "error_seed": int(error_seed),
    "target_b0_enabled": bool(target_b0_enabled), "target_b0_t": float(target_b0_t),
    "b0_definition": b0_mode,
}
current_settings = {
    "axis_samples": int(axis_samples), "field_margin_periods": float(field_margin_periods),
    "electron_energy_GeV": float(electron_energy_GeV), "relax": bool(relax),
    "precision": float(precision), "max_iter": int(max_iter), "method": 4,
    "calculate_2d": bool(make_2d), "calculate_3d": bool(make_3d),
    "transverse_half_width_mm": float(transverse_half_mm),
}
current_ui = {"compare_ideal": bool(compare_ideal), "geometry_limit": int(geometry_limit)}

st.download_button(
    "Export current preset (.json)",
    preset_json_bytes(
        current_params, current_settings, name=f"{device} requested configuration",
        ui_settings=current_ui,
        study_defaults=st.session_state.get("preset_study_defaults"),
        extensions=st.session_state.get("preset_extensions"),
    ),
    "radia_magnet_preset_v1.json", "application/json",
    on_click="ignore", width="stretch",
)

run = st.button("Build + Solve + Analyze", type="primary", width="stretch")

if run:
    try:
        rad = load_radia()
        if hasattr(rad, "UtiDelAll"):
            rad.UtiDelAll()

        params = dict(current_params)
        params.update({
            "relax_enabled": bool(relax), "relax_precision_t": float(precision),
            "relax_max_iterations": int(max_iter), "axis_samples": int(axis_samples),
            "field_margin_periods": float(field_margin_periods),
            "electron_energy_GeV": float(electron_energy_GeV),
            "calculate_2d_slice": bool(make_2d),
            "calculate_3d_field_map": bool(make_3d),
            "transverse_map_half_width_mm": float(transverse_half_mm),
        })
        progress = st.progress(0, text="Preparing model…")

        calibration_history = []
        if target_b0_enabled:
            calibrated_br, calibration_history = calibrate_br(
                rad, device, params, float(target_b0_t),
                mode=b0_mode,
                relax=bool(relax), precision=float(precision), max_iter=int(max_iter)
            )
            params["br_t"] = float(calibrated_br)
            progress.progress(12, text=f"B0 calibration complete: Br={calibrated_br:.6g} T")
            if hasattr(rad, "UtiDelAll"):
                rad.UtiDelAll()

        # Build models first. Sampling range is derived from actual generated geometry,
        # including APPLE-II row shifts and random longitudinal block errors.
        ideal_model = None
        if errors_enabled and compare_ideal:
            p_ideal = dict(params)
            p_ideal["errors_enabled"] = False
            ideal_model = build_device(rad, device, p_ideal)

        model = build_device(rad, device, params)
        progress.progress(25, text=f"Generated {len(model['blocks'])} magnetic blocks.")

        ideal_rlx = None
        if ideal_model is not None:
            ideal_rlx = solve_model(
                rad, ideal_model, relax=bool(relax),
                precision=float(precision), max_iter=int(max_iter), method=4
            )

        rlx = solve_model(
            rad, model, relax=bool(relax),
            precision=float(precision), max_iter=int(max_iter), method=4
        )
        progress.progress(40, text="RADIA solve stage converged / completed.")

        range_models = [model] + ([ideal_model] if ideal_model is not None else [])
        z_lo, z_hi = union_field_range(
            range_models, float(period_mm), float(field_margin_periods)
        )
        z = np.linspace(z_lo, z_hi, int(axis_samples))
        params["field_range_mm"] = [float(z_lo), float(z_hi)]
        B = sample_on_axis(rad, model["obj"], z)
        metrics = analyze(z, B, float(period_mm), float(electron_energy_GeV))

        Bideal = None
        ideal_metrics = None
        if ideal_model is not None:
            Bideal = sample_on_axis(rad, ideal_model["obj"], z)
            ideal_metrics = analyze(z, Bideal, float(period_mm), float(electron_energy_GeV))
        progress.progress(58, text="Geometry-derived on-axis field range sampled.")

        slice_data = None
        slice_axis = None
        slice_plane = "XZ"
        if make_2d:
            t = np.linspace(-float(transverse_half_mm), float(transverse_half_mm), 31)
            z2 = np.linspace(z_lo, z_hi, min(181, max(61, int(axis_samples)//5)))
            if abs(metrics["By_peak_T"]) >= abs(metrics["Bx_peak_T"]):
                slice_data = sample_slice_xz(rad, model["obj"], t, z2, 0.0)
                slice_plane = "XZ"
            else:
                slice_data = sample_slice_yz(rad, model["obj"], t, z2, 0.0)
                slice_plane = "YZ"
            slice_axis = (t, z2)
        progress.progress(72, text="2D map complete." if make_2d else "2D map skipped.")

        field3 = None
        grid3 = None
        if make_3d:
            x3 = np.linspace(-float(transverse_half_mm), float(transverse_half_mm), 5)
            y3 = np.linspace(-float(transverse_half_mm), float(transverse_half_mm), 5)
            z3 = np.linspace(z_lo, z_hi, 17)
            field3 = sample_3d(rad, model["obj"], x3, y3, z3)
            grid3 = (x3, y3, z3)
        progress.progress(90, text="3D field map complete." if make_3d else "3D map skipped.")
        progress.progress(100, text="Complete.")
        # Publish the baseline only after the magnetic model, field sampling, and
        # primary analysis have all completed successfully. Advanced analysis below
        # consumes this exact realized configuration (including calibrated Br).
        st.session_state["magnet_study_source"] = {
            "base_parameters": dict(params),
            "settings": dict(current_settings),
            "study_defaults": dict(st.session_state.get("preset_study_defaults", {})),
            "extensions": dict(st.session_state.get("preset_extensions", {})),
        }

        classification = classify_k(metrics["K_peak"])
        eph = metrics["electron_phase_error_rms_deg"]

        st.subheader("Key results")

        # Two rows of three cards are much more readable than six compressed
        # cards on laptops / narrow browser windows.
        r1 = st.columns(3)
        r1[0].metric(
            "Peak transverse field |B⊥|",
            f"{metrics['Bperp_peak_T']:.6g} T",
            help="Maximum sqrt(Bx² + By²) over the sampled on-axis field range."
        )
        r1[1].metric(
            "Undulator K (peak)",
            f"{metrics['K_peak']:.6g}",
            help="Kpeak = 0.934 × max(|B⊥|)[T] × λu[cm]."
        )
        r1[2].metric(
            "Remanent induction Br",
            f"{params['br_t']:.6g} T",
            help="Actual Br used after optional target-B0 calibration."
        )

        r2 = st.columns(3)
        r2[0].metric(
            "Kx / Ky",
            f"{metrics['Kx_peak']:.6g} / {metrics['Ky_peak']:.6g}",
            help="Component K amplitudes from By and Bx respectively."
        )
        r2[1].metric(
            "3rd harmonic H3/H1",
            f"{metrics['H3_over_H1']:.6e}",
            help="FFT amplitude ratio of the dominant transverse field component."
        )
        r2[2].metric(
            "Electron phase error RMS",
            "n/a" if not math.isfinite(eph) else f"{eph:.6g}°",
            help="Trajectory/slippage-derived electron phase-error RMS."
        )

        # Exact-value table: preserves the same computed results while making
        # units and definitions visible in one place.
        key_rows = [
            {
                "Quantity": "Peak transverse field |B⊥|",
                "Value": float(metrics["Bperp_peak_T"]),
                "Unit": "T",
            },
            {
                "Quantity": "K peak",
                "Value": float(metrics["K_peak"]),
                "Unit": "dimensionless",
            },
            {
                "Quantity": "Kx peak",
                "Value": float(metrics["Kx_peak"]),
                "Unit": "dimensionless",
            },
            {
                "Quantity": "Ky peak",
                "Value": float(metrics["Ky_peak"]),
                "Unit": "dimensionless",
            },
            {
                "Quantity": "K vector norm",
                "Value": float(metrics["K_vector_norm"]),
                "Unit": "dimensionless",
            },
            {
                "Quantity": "H3/H1",
                "Value": float(metrics["H3_over_H1"]),
                "Unit": "ratio",
            },
            {
                "Quantity": "H5/H1",
                "Value": float(metrics["H5_over_H1"]),
                "Unit": "ratio",
            },
            {
                "Quantity": "Electron phase error RMS",
                "Value": float(eph) if math.isfinite(eph) else None,
                "Unit": "deg",
            },
            {
                "Quantity": "Zero-crossing field phase RMS",
                "Value": (
                    float(metrics["zero_crossing_field_phase_rms_deg"])
                    if math.isfinite(metrics["zero_crossing_field_phase_rms_deg"])
                    else None
                ),
                "Unit": "deg",
            },
            {
                "Quantity": "I1x / I1y",
                "Value": f"{metrics['I1x_Tm']:.9g} / {metrics['I1y_Tm']:.9g}",
                "Unit": "T·m",
            },
            {
                "Quantity": "I2x / I2y",
                "Value": f"{metrics['I2x_Tm2']:.9g} / {metrics['I2y_Tm2']:.9g}",
                "Unit": "T·m²",
            },
        ]

        with st.expander("Detailed numerical results", expanded=False):
            st.dataframe(
                pd.DataFrame(key_rows),
                width="stretch",
                hide_index=True,
            )

        st.info(
            f"**Field integration range:** {z_lo:.3f} → {z_hi:.3f} mm  "
            f"\n\n**Fringe-field margin:** {float(field_margin_periods):.2f} period(s)  "
            f"\n\n**Magnetic regime:** {classification}"
        )

        if device == "Wiggler":
            if metrics["K_peak"] >= 3:
                st.success(f"Wiggler mode — Kpeak={metrics['K_peak']:.3g}; {classification}.")
            else:
                st.warning(
                    f"Wiggler mode selected, but Kpeak={metrics['K_peak']:.3g}; "
                    f"{classification}. Increase field/period if a high-K wiggler is intended."
                )
        else:
            st.info(f"Computed magnetic regime: {classification}.")

        tabs = st.tabs([
            "On-axis field", "2D map", "3D field map", "3D magnet geometry",
            "Trajectory", "Electron phase", "Ideal comparison", "Metrics & export"
        ])

        with tabs[0]:
            st.plotly_chart(field_lines(z, B), width="stretch")

        with tabs[1]:
            if slice_data is None:
                st.info("2D map was disabled.")
            else:
                comp = 1 if abs(metrics["By_peak_T"]) >= abs(metrics["Bx_peak_T"]) else 0
                axis, z2 = slice_axis
                st.plotly_chart(
                    slice_heatmap(axis, z2, slice_data, comp, "x" if slice_plane == "XZ" else "y"),
                    width="stretch"
                )

        with tabs[2]:
            if field3 is None:
                st.info("3D field map was disabled.")
            else:
                st.plotly_chart(field_cones(*grid3, field3), width="stretch")
                st.download_button(
                    "Export V11-compatible 3D field map CSV",
                    fieldmap3d_csv_bytes(*grid3, field3),
                    "radia_3d_field_map.csv", "text/csv", on_click="ignore"
                )

        with tabs[3]:
            st.plotly_chart(
                geometry_view(model["blocks"], int(geometry_limit), True),
                width="stretch"
            )
            st.caption(
                "Cuboids use the actual centres/dimensions passed to RADIA; cones show "
                "easy-axis/magnetization directions."
            )
            st.dataframe(pd.DataFrame(model["blocks"][:min(1000, len(model["blocks"]))]), width="stretch")

        with tabs[4]:
            st.plotly_chart(trajectory_plot(z, metrics["trajectory"]), width="stretch")
            st.caption("Ultra-relativistic small-angle trajectory from the sampled transverse field.")

        with tabs[5]:
            st.plotly_chart(electron_phase_plot(metrics["electron_phase"]), width="stretch")
            st.metric(
                "Trajectory-derived electron phase error RMS",
                "n/a" if not math.isfinite(eph) else f"{eph:.5g}°"
            )
            zc = metrics["zero_crossing_field_phase_rms_deg"]
            st.metric(
                "Zero-crossing field phase RMS (diagnostic only)",
                "n/a" if not math.isfinite(zc) else f"{zc:.5g}°"
            )
            st.caption(
                "Electron phase uses longitudinal slippage from x′ and y′ and removes the "
                "best-fit linear slippage. The zero-crossing metric is retained only as a "
                "field-shape diagnostic."
            )

        with tabs[6]:
            if ideal_metrics is None:
                st.info("Enable Manufacturing errors + Ideal-vs-error comparison to populate this tab.")
            else:
                st.plotly_chart(ideal_error_field_plot(z, Bideal, B), width="stretch")
                cmp = compare_metrics(ideal_metrics, metrics)
                rows = [
                    {"metric": key, "ideal": val["ideal"], "error_model": val["error"], "delta": val["delta"]}
                    for key, val in cmp.items()
                ]
                st.dataframe(pd.DataFrame(rows), width="stretch")

        with tabs[7]:
            shown = {k: v for k, v in metrics.items() if k not in ("trajectory", "electron_phase")}
            shown["classification"] = classification
            shown["field_range_mm"] = [float(z_lo), float(z_hi)]
            if rlx is not None:
                shown["RADIA_relaxation_result"] = rlx
                shown["RADIA_relaxation_converged"] = True
            st.json(shown)

            if calibration_history:
                st.subheader("Target B0 calibration")
                st.dataframe(pd.DataFrame(calibration_history), width="stretch")
                st.write(
                    f"Target: **{float(target_b0_t):.6g} T** using **{b0_mode}** "
                    f"→ calibrated Br: **{params['br_t']:.6g} T**"
                )

            st.subheader("Export")
            try:
                st.download_button(
                    "Download realized preset (.json)",
                    preset_json_bytes(
                        params, current_settings,
                        name=f"{device} realized configuration",
                        description="Exact successfully solved configuration, including calibrated Br.",
                        realized=True, calibration_history=calibration_history,
                        ui_settings=current_ui,
                        study_defaults=st.session_state.get("preset_study_defaults"),
                        extensions=st.session_state.get("preset_extensions"),
                    ),
                    "radia_magnet_realized_preset_v1.json", "application/json",
                    on_click="ignore", width="stretch",
                    help="Portable settings-only file for reproducing this solved model or loading it in another application.",
                )
            except Exception as exc:
                st.warning(f"Realized-preset export unavailable: {exc}")

            e1, e2 = st.columns(2)
            try:
                e1.download_button(
                    "Download on-axis field CSV",
                    csv_bytes(z, B),
                    "radia_on_axis_field.csv",
                    "text/csv",
                    on_click="ignore",
                    width="stretch",
                )
            except Exception as exc:
                e1.warning(f"CSV export unavailable: {exc}")

            try:
                e2.download_button(
                    "Download complete JSON",
                    json_bytes(params, metrics),
                    "radia_summary.json",
                    "application/json",
                    on_click="ignore",
                    width="stretch",
                )
            except Exception as exc:
                e2.warning(f"JSON export unavailable: {exc}")

            e3, e4 = st.columns(2)
            try:
                e3.download_button(
                    "Download HDF5",
                    hdf5_bytes(z, B, params, metrics),
                    "radia_field.h5",
                    "application/x-hdf5",
                    on_click="ignore",
                    width="stretch",
                )
            except Exception as exc:
                e3.warning(f"HDF5 export unavailable: {exc}")

            try:
                e4.download_button(
                    "Download PDF report",
                    pdf_bytes(z, B, params, metrics),
                    "radia_report.pdf",
                    "application/pdf",
                    on_click="ignore",
                    width="stretch",
                )
            except Exception as exc:
                e4.warning(f"PDF export unavailable: {exc}")

            try:
                comparison = compare_metrics(ideal_metrics, metrics) if ideal_metrics is not None else None
                run_metadata = {
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "radia_module": getattr(rad, "__file__", None),
                    "radia_relaxation_result": rlx,
                    "radia_relaxation_converged": rlx is not None,
                    "calibration_history": calibration_history,
                }
                package = research_package_bytes(
                    params, metrics, z, B,
                    grid3=grid3, field3=field3,
                    blocks=model.get("blocks"),
                    comparison=comparison,
                    run_metadata=run_metadata,
                )
                package_status = validate_research_package_bytes(package)
                st.caption(
                    f"Transfer package verified: schema {package_status['schema_version']}; "
                    f"{package_status['payload_files_checked']} payload files checked."
                )
                st.download_button(
                    "Download downstream research package (.zip)",
                    package,
                    "radia_magnet_studio_transfer_v1.zip",
                    "application/zip",
                    on_click="ignore",
                    width="stretch",
                    help=(
                        "Versioned package containing device settings, units, coordinate definitions, "
                        "geometry, checksums, on-axis field data and the optional 3D field map."
                    ),
                )
            except Exception as exc:
                st.warning(f"Research-package export unavailable: {exc}")

            if device == "APPLE-II":
                st.warning(
                    "APPLE-II remains a physics-informed four-array research prototype, "
                    "not a facility/manufacturer-certified magnetic model."
                )

    except Exception as exc:
        st.exception(exc)

from app.advanced_analysis import render_advanced_analysis

render_advanced_analysis()
