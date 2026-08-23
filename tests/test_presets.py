from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presets import BUILTIN_PRESETS, build_preset, parse_preset, preset_json_bytes, runtime_to_widget_state


PARAMETERS = {
    "device": "APPLE-II", "period_mm": 48.0, "periods": 16, "gap_mm": 10.5,
    "blocks_per_period": 4, "block_width_mm": 11.0, "block_height_mm": 9.0,
    "longitudinal_fill": 0.88, "br_t": 1.17, "material_mode": "Fixed remanence",
    "mu_parallel": 1.04, "mu_perpendicular": 1.06, "segmentation": [2, 2, 2],
    "ellipticity": 0.4, "apple_phase_deg": 45.0, "apple_shift_mode": "Parallel",
    "errors_enabled": True, "field_error_pct": 0.8, "longitudinal_error_mm": 0.04,
    "transverse_error_mm": 0.03, "angle_error_deg": 0.2,
    "gap_asymmetry_mm": 0.01, "bank_imbalance_pct": 0.4, "error_seed": 42,
    "target_b0_enabled": True, "target_b0_t": 0.2,
    "b0_definition": "Central 3-period peak B⊥",
}
SETTINGS = {
    "axis_samples": 1200, "field_margin_periods": 1.5,
    "electron_energy_GeV": 2.75, "relax": False, "precision": 2e-5,
    "max_iter": 800, "method": 4, "calculate_2d": True,
    "calculate_3d": False, "transverse_half_width_mm": 4.0,
}


def run():
    raw = preset_json_bytes(
        PARAMETERS, SETTINGS, name="Round trip", extensions={"radiation_simulator": {"mode": "future"}},
        study_defaults={"monte_carlo_samples": 31}, ui_settings={"geometry_limit": 700},
    )
    payload = json.loads(raw)
    runtime = parse_preset(raw)
    for key, value in PARAMETERS.items():
        assert runtime["parameters"][key] == value, key
    for key, value in SETTINGS.items():
        assert runtime["settings"][key] == value, key
    assert runtime["extensions"]["radiation_simulator"]["mode"] == "future"
    assert runtime["study_defaults"]["monte_carlo_samples"] == 31
    widget_state = runtime_to_widget_state(runtime)
    assert widget_state["cfg_device"] == "APPLE-II"
    assert widget_state["cfg_seg_n"] == 2
    assert widget_state["cfg_axis_samples"] == 1200

    tampered = copy.deepcopy(payload)
    tampered["device"]["gap_mm"] = 99.0
    try:
        parse_preset(tampered)
        raise AssertionError("Tampered fingerprint was accepted")
    except ValueError as exc:
        assert "fingerprint" in str(exc)

    wrong_units = copy.deepcopy(payload)
    wrong_units.pop("fingerprint_sha256")
    wrong_units["conventions"]["length_unit"] = "m"
    try:
        parse_preset(wrong_units)
        raise AssertionError("Wrong units were accepted")
    except ValueError as exc:
        assert "length_unit" in str(exc)

    missing_fingerprint = copy.deepcopy(payload)
    missing_fingerprint.pop("fingerprint_sha256")
    try:
        parse_preset(missing_fingerprint)
        raise AssertionError("Missing fingerprint was accepted")
    except ValueError as exc:
        assert "fingerprint" in str(exc)

    string_boolean = copy.deepcopy(payload)
    string_boolean.pop("fingerprint_sha256")
    string_boolean["manufacturing_errors"]["enabled"] = "false"
    try:
        parse_preset(string_boolean)
        raise AssertionError("String boolean was accepted")
    except ValueError as exc:
        assert "JSON boolean" in str(exc)

    assert len(BUILTIN_PRESETS) >= 6
    for name, builtin in BUILTIN_PRESETS.items():
        assert parse_preset(builtin)["metadata"]["name"], name
    print("PRESET IMPORT/EXPORT ROUND-TRIP TESTS PASSED")


if __name__ == "__main__":
    run()
