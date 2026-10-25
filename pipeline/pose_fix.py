"""
Fixes for Mobility-Scooter-Project/pipeline pose output.

Two independent defects are addressed:

(1) LETTERBOX NORMALISATION.  Yolov7Pose.convert() divides both x and y by
    self.input_size (192).  letterbox(img, 192, stride=64, auto=True) does not
    produce a 192x192 canvas: for a 16:9 frame it produces 192x128, with the
    108-row image content padded by 10 grey rows top and bottom.  x is therefore
    correct (no horizontal padding) but y is compressed and offset.  Recovery for
    any 16:9 source:  y_true = (y_stored * 192 - 10) / 108.
    Aspect ratio is preserved by letterbox, so angles computed from the stored
    values are only wrong through this affine y term.

(2) NO FRAME INDEX.  process_file() writes a row only when the detector returns
    landmarks, so row i does not correspond to frame i.  The indices are printed
    by the pipeline ("Estimation failed in frames: [...]") - recover them from the
    run log if it was kept, otherwise re-run with the patch below.
"""
import numpy as np, pandas as pd

INPUT_SIZE = 192
STRIDE = 64

def letterbox_geometry(h, w, input_size=INPUT_SIZE, stride=STRIDE):
    """Returns (scale, unpadded_w, unpadded_h, pad_x, pad_y) used by the pipeline."""
    r = min(input_size/h, input_size/w)
    uw, uh = int(round(w*r)), int(round(h*r))
    dw, dh = np.mod(input_size-uw, stride)/2, np.mod(input_size-uh, stride)/2
    return r, uw, uh, dw, dh

def correct_pose_csv(df, frame_h=480, frame_w=854, input_size=INPUT_SIZE):
    """Map stored coordinates to fractions of the original frame.

    x_out, y_out are in [0,1] relative to frame width / height.
    """
    r, uw, uh, dw, dh = letterbox_geometry(frame_h, frame_w, input_size)
    out = df.copy()
    for i in range(9):
        out[f'x{i}'] = (df[f'x{i}']*input_size - dw) / uw
        out[f'y{i}'] = (df[f'y{i}']*input_size - dh) / uh
    return out

def normalise_body(df, origin='hip_mid', scale='shoulder_width'):
    """Body-centred, scale-free coordinates for modelling.

    Keypoint order (COCO indices 0,5,6,7,8,9,10,11,12):
      0 nose, 1 L shoulder, 2 R shoulder, 3 L elbow, 4 R elbow,
      5 L wrist, 6 R wrist, 7 L hip, 8 R hip
    Requires frame-corrected input and an isotropic pixel scale, i.e. multiply
    y by (frame_h/frame_w) first, otherwise angles are distorted by the aspect.
    """
    X = df[[f'x{i}' for i in range(9)]].values.copy()
    Y = df[[f'y{i}' for i in range(9)]].values.copy()
    ox = X[:, [7, 8]].mean(1) if origin == 'hip_mid' else X[:, [1, 2]].mean(1)
    oy = Y[:, [7, 8]].mean(1) if origin == 'hip_mid' else Y[:, [1, 2]].mean(1)
    s = np.hypot(X[:, 1]-X[:, 2], Y[:, 1]-Y[:, 2]) if scale == 'shoulder_width' else 1.0
    s = np.where(s < 1e-3, np.nan, s)
    cols = {}
    for i in range(9):
        cols[f'nx{i}'] = (X[:, i]-ox)/s
        cols[f'ny{i}'] = (Y[:, i]-oy)/s
    return pd.DataFrame(cols, index=df.index)

def quality_flags(df, min_shoulder=0.10, centre=(0.35, 0.75)):
    """Rows that should be dropped or down-weighted when confidences are absent."""
    X = df[[f'x{i}' for i in range(9)]].values
    Y = df[[f'y{i}' for i in range(9)]].values
    sw = np.hypot(X[:, 1]-X[:, 2], Y[:, 1]-Y[:, 2])
    cx = X.mean(1)
    cs = np.r_[0, np.hypot(np.diff(cx), np.diff(Y.mean(1)))]
    return pd.DataFrame({'collapsed': sw < min_shoulder,
                         'off_centre': (cx < centre[0]) | (cx > centre[1]),
                         'jump': cs > 0.1,
                         'out_of_frame': (X < 0).any(1) | (X > 1).any(1)}, index=df.index)
