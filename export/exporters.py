from __future__ import annotations
import io, json, tempfile, os
import numpy as np

def csv_bytes(z_mm,B):
    out=io.StringIO()
    out.write("z_mm,Bx_T,By_T,Bz_T\n")
    for z,b in zip(z_mm,B):
        out.write(f"{float(z)},{float(b[0])},{float(b[1])},{float(b[2])}\n")
    return out.getvalue().encode("utf-8")

def _json_safe(value):
    """
    Recursively convert NumPy/Python scientific objects to standard JSON types.
    This keeps the numerical values unchanged while making arrays serializable.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # Last-resort representation for non-numerical metadata only.
    return str(value)

def json_bytes(params, metrics):
    payload = {
        "parameters": _json_safe(params),
        "metrics": _json_safe(metrics),
    }
    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        allow_nan=True,
    ).encode("utf-8")

def hdf5_bytes(z_mm, B, params, metrics):
    import h5py
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    skipped = []
    try:
        with h5py.File(path, "w") as f:
            f.create_dataset("z_mm", data=np.asarray(z_mm, float))
            f.create_dataset("B_T", data=np.asarray(B, float))

            p = f.create_group("parameters")
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    p.attrs[k] = v
                elif isinstance(v, (tuple, list)) and all(isinstance(x, (int, float)) for x in v):
                    p.create_dataset(k, data=np.asarray(v, float))
                else:
                    skipped.append(f"parameter:{k}:{type(v).__name__}")

            m = f.create_group("metrics")
            for k, v in metrics.items():
                if k in ("trajectory", "electron_phase"):
                    continue
                if isinstance(v, (int, float, str, bool, np.number)):
                    m.attrs[k] = v
                elif v is None:
                    m.attrs[k] = "None"
                else:
                    skipped.append(f"metric:{k}:{type(v).__name__}")

            if skipped:
                dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset("export_skipped_items", data=np.asarray(skipped, dtype=object), dtype=dt)

        return open(path, "rb").read()
    finally:
        try:
            os.unlink(path)
        except OSError:
            # Failure to remove a temporary file does not corrupt the exported
            # HDF5 bytes; deliberately ignore only this cleanup condition.
            pass

def pdf_bytes(z_mm,B,params,metrics):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    bio=io.BytesIO()
    with PdfPages(bio) as pdf:
        fig=plt.figure(figsize=(8.5,11))
        fig.text(0.08,0.95,"RADIA Magnet Studio Research Report",fontsize=16)
        y=0.90
        for k,v in params.items():
            fig.text(0.08,y,f"{k}: {v}",fontsize=9); y-=0.025
        y-=0.02
        for k,v in metrics.items():
            if k in ("trajectory","electron_phase"): continue
            fig.text(0.08,y,f"{k}: {v}",fontsize=9); y-=0.025
            if y<0.08: break
        plt.axis("off"); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(10,6))
        plt.plot(z_mm,B[:,0],label="Bx")
        plt.plot(z_mm,B[:,1],label="By")
        plt.plot(z_mm,B[:,2],label="Bz")
        plt.xlabel("z (mm)"); plt.ylabel("B (T)"); plt.legend(); plt.title("On-axis magnetic field")
        pdf.savefig(fig); plt.close(fig)

        tr=metrics.get("trajectory")
        if tr is not None:
            fig=plt.figure(figsize=(10,6))
            plt.plot(z_mm,tr["x_mm"],label="x(z)")
            plt.plot(z_mm,tr["y_mm"],label="y(z)")
            plt.xlabel("z (mm)"); plt.ylabel("displacement (mm)"); plt.legend(); plt.title("Electron trajectory")
            pdf.savefig(fig); plt.close(fig)
    return bio.getvalue()


def fieldmap3d_csv_bytes(x_mm,y_mm,z_mm,B3):
    out=io.StringIO()
    out.write("x_m,y_m,z_m,Bx_T,By_T,Bz_T\n")
    for iz,z in enumerate(z_mm):
        for iy,y in enumerate(y_mm):
            for ix,x in enumerate(x_mm):
                b=B3[iz,iy,ix]
                out.write(
                    f"{float(x)*1e-3},{float(y)*1e-3},{float(z)*1e-3},"
                    f"{float(b[0])},{float(b[1])},{float(b[2])}\n"
                )
    return out.getvalue().encode("utf-8")
