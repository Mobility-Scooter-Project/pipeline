"""
Per-session preprocessing for mobility-scooter pose + IMU.

Inputs per session
  imu_csv      acc_x..gyro_z, datetime   (~5 Hz, wall clock)
  video_paths  chunked GoPro clips IN ORDER (splits of one continuous recording)
  pose_paths   one pose CSV per clip, same order

Pipeline
  1. IMU        bias/scale/gravity calibration on stationary windows
  2. pose       letterbox y-correction, quality flags
  3. frame id   monotone DP recovery of row -> frame when the CSV has no index
  4. sync       one global offset T0 from scooter-motion state agreement
  5. merge      pose averaged onto the IMU sample grid + kinematics
  6. QC         gates; sessions that fail are flagged, not silently emitted

Each clip is decoded ONCE at 160x90; the motion signal and the frame-recovery
evidence are both derived from that pass.

"""
import argparse, gc, glob, json, os, re, shutil, subprocess, sys, tempfile, time
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt

FFMPEG = os.environ.get('FFMPEG_BINARY') or 'ffmpeg'
FFPROBE = os.environ.get('FFPROBE_BINARY') or 'ffprobe'

INSTALL_HINT = """ffmpeg not found. The PROGRAM is required (every clip is decoded
through it) - note that `pip install ffmpeg` / `ffmpeg-python` only install Python
wrappers and provide NO binary. Install one of:
    macOS        brew install ffmpeg
    conda        conda install -c conda-forge ffmpeg
    Debian       sudo apt install ffmpeg
    pip, no admin  pip install imageio-ffmpeg   (ships a static ffmpeg; this script
                   finds it automatically, and ffprobe is then unavailable, so the
                   webcam frame-count check is skipped)
or point at a build with --ffmpeg /path/to/ffmpeg, or export FFMPEG_BINARY."""


def _resolve(val):
    if shutil.which(val): return shutil.which(val)
    if os.path.isfile(val) and os.access(val, os.X_OK): return val
    return None


def check_tools(ffmpeg=None, ffprobe=None, verbose=True):
    """Resolve ffmpeg (required) and ffprobe (optional).

    Called before any session is touched: a missing binary surfaced three clips
    into a 78-session batch wastes an hour. Falls back to the static binary that
    imageio-ffmpeg bundles, which is installable with pip and needs no admin
    rights. ffprobe is NOT required - its only use is cross-checking the webcam
    frame count against the IMU row count, so it degrades to a skipped check.
    """
    global FFMPEG, FFPROBE
    if ffmpeg: FFMPEG = ffmpeg
    if ffprobe: FFPROBE = ffprobe
    got = _resolve(FFMPEG)
    if got is None and not ffmpeg:
        try:                                     # pip install imageio-ffmpeg
            import imageio_ffmpeg
            cand = imageio_ffmpeg.get_ffmpeg_exe()
            if cand and os.path.isfile(cand):
                got = cand
                if verbose: print(f'using the ffmpeg bundled with imageio-ffmpeg: {cand}')
        except Exception:
            pass
    if got is None:
        raise FileNotFoundError(f'ffmpeg (looked for {FFMPEG!r})\n\n{INSTALL_HINT}')
    FFMPEG = got
    FFPROBE = _resolve(FFPROBE)
    if FFPROBE is None and verbose:
        print('note: ffprobe not found - the webcam/IMU frame-count check will be skipped')
    return FFMPEG, FFPROBE


G = 9.80665
ACC = ['acc_x','acc_y','acc_z']; GYR = ['gyro_x','gyro_y','gyro_z']
INPUT_SIZE = 192; STRIDE = 64
KP = ['nose','sho_L','sho_R','elb_L','elb_R','wri_L','wri_R','hip_L','hip_R']
THR_GRID = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12)
GATES = dict(acc_scale=(0.90, 1.12), stationary_frac_min=0.02, pose_coverage_min=0.50,
             off_centre_max=0.30, kappa_margin_min=0.05, stopped_contrast_max=-0.05)

# ============================================================== IMU

def load_imu(path):
    d = pd.read_csv(path).dropna(axis=1, how='all')
    d['datetime'] = pd.to_datetime(d['datetime'])
    d = d.sort_values('datetime').reset_index(drop=True)
    t = d['datetime'].astype('int64')/1e9
    d['t_imu'] = t - t.iloc[0]
    d['dt'] = d['t_imu'].diff()
    d['gap_flag'] = d['dt'] > 1.5*np.nanmedian(d['dt'])
    return d

def stationary_mask(d, win=10, acc_thr=0.01, gyr_thr=0.01):
    a = np.linalg.norm(d[ACC].values, axis=1); g = np.linalg.norm(d[GYR].values, axis=1)
    sa = pd.Series(a).rolling(win, center=True, min_periods=3).std()
    sg = pd.Series(g).rolling(win, center=True, min_periods=3).std()
    return ((sa < acc_thr) & (sg < gyr_thr)).values

def calibrate_imu(d, fs=5.0, fc=0.2):
    """Scale the accelerometer so stationary |a| = 1 g, remove a time-varying gyro
    bias, rotate into a gravity-aligned frame and subtract gravity. Heading is not
    identified: acc_lin_h1/h2 are horizontal but not forward/lateral."""
    still = stationary_mask(d)
    if still.sum() < 50:
        raise ValueError('too few stationary samples to calibrate (%d)' % still.sum())
    t = d['t_imu'].values; g = d[GYR].values
    blk = (t//100).astype(int); cent, bias = [], []
    for b in np.unique(blk[still]):
        m = still & (blk == b)
        if m.sum() >= 20: cent.append(t[m].mean()); bias.append(g[m].mean(0))
    cent = np.asarray(cent); bias = np.asarray(bias)
    bias_t = (np.column_stack([np.interp(t, cent, bias[:, k]) for k in range(3)])
              if len(cent) >= 2 else np.repeat(g[still].mean(0)[None], len(d), 0))
    k_scale = 1.0/np.linalg.norm(d.loc[still, ACC].values, axis=1).mean()
    acc = d[ACC].values*k_scale*G; gyr = g - bias_t
    b_, a_ = butter(2, fc/(fs/2), 'low'); grav = filtfilt(b_, a_, acc, axis=0)
    u = grav/np.linalg.norm(grav, axis=1, keepdims=True); z = np.array([0., 0., 1.])
    v = np.cross(u, z); s = np.linalg.norm(v, axis=1); c = u@z
    R = np.tile(np.eye(3), (len(u), 1, 1)); nz = s > 1e-8
    vx = np.zeros((len(u), 3, 3))
    vx[:,0,1], vx[:,0,2] = -v[:,2], v[:,1]
    vx[:,1,0], vx[:,1,2] =  v[:,2], -v[:,0]
    vx[:,2,0], vx[:,2,1] = -v[:,1], v[:,0]
    kk = ((1-c)/np.where(s**2 == 0, 1, s**2))[:, None, None]
    R[nz] = (np.eye(3) + vx + kk*(vx@vx))[nz]
    acc_l = np.einsum('nij,nj->ni', R, acc); gyr_l = np.einsum('nij,nj->ni', R, gyr)
    lin = acc_l - np.array([0, 0, G])
    d = d.copy(); d['is_stationary'] = still
    for i, c_ in enumerate(['acc_x_cal','acc_y_cal','acc_z_cal']): d[c_] = acc[:, i]
    for i, c_ in enumerate(['gyro_x_cal','gyro_y_cal','gyro_z_cal']): d[c_] = gyr[:, i]
    d['acc_lin_h1'], d['acc_lin_h2'], d['acc_vert'] = lin[:,0], lin[:,1], lin[:,2]
    d['acc_lin_horiz'] = np.linalg.norm(lin[:, :2], axis=1)
    d['yaw_rate'] = gyr_l[:, 2]
    d['pitch_deg'] = np.degrees(np.arctan2(-grav[:,1], grav[:,2]))
    d['roll_deg']  = np.degrees(np.arctan2( grav[:,0], grav[:,2]))
    gm = d.loc[still, ACC].values.mean(0)
    cal = dict(acc_scale=float(k_scale), stationary_frac=float(still.mean()),
               gyro_bias_range_degs=[float(np.ptp(np.degrees(bias[:, i]))) for i in range(3)]
                                     if len(cent) >= 2 else [0., 0., 0.],
               mount_tilt_deg=float(np.degrees(np.arccos(gm[2]/np.linalg.norm(gm)))))
    return d, cal

# ============================================================== pose

def letterbox_geometry(h, w, input_size=INPUT_SIZE, stride=STRIDE):
    r = min(input_size/h, input_size/w)
    uw, uh = int(round(w*r)), int(round(h*r))
    return r, uw, uh, np.mod(input_size-uw, stride)/2, np.mod(input_size-uh, stride)/2

def correct_pose(df, frame_h=480, frame_w=854, input_size=INPUT_SIZE):
    """Undo the letterbox normalisation bug in Yolov7Pose.convert(), which divides
    both axes by input_size on a canvas that is not square. For a 16:9 frame the
    canvas is 192x128 with 108 rows of content and 10 rows of padding, so x is
    correct and y is compressed and offset. Output is fractions of the frame.
    Skip for pose produced by the patched extractor, which already emits fractions."""
    r, uw, uh, dw, dh = letterbox_geometry(frame_h, frame_w, input_size)
    out = df.copy()
    for i in range(9):
        out[f'x{i}'] = (df[f'x{i}']*input_size - dw)/uw
        out[f'y{i}'] = (df[f'y{i}']*input_size - dh)/uh
    return out

def pose_needs_correction(df, tol=0.08):
    """Letterbox bug leaves y capped near (10+108)/192 = 0.615 instead of ~1."""
    y = df[[f'y{i}' for i in range(9)]].values.astype(float)
    if not np.isfinite(y).any():
        return False
    return bool(np.nanpercentile(y, 99) < 0.615 + tol)

def quality_flags(df, min_shoulder=0.10, centre=(0.35, 0.75)):
    X = df[[f'x{i}' for i in range(9)]].values; Y = df[[f'y{i}' for i in range(9)]].values
    sw = np.hypot(X[:,1]-X[:,2], Y[:,1]-Y[:,2]); cx = X.mean(1)
    cs = np.r_[0, np.hypot(np.diff(cx), np.diff(Y.mean(1)))]
    return pd.DataFrame({'collapsed': sw < min_shoulder,
                         'off_centre': (cx < centre[0]) | (cx > centre[1]),
                         'jump': cs > 0.1}, index=df.index)

# ============================================================== video

def decode(path, w=160, h=90):
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f: raw = f.name
    try:
        subprocess.run([FFMPEG,'-v','error','-i',path,'-vf',f'scale={w}:{h}',
                        '-pix_fmt','rgb24','-f','rawvideo',raw,'-y'], check=True)
        return np.fromfile(raw, dtype=np.uint8).reshape(-1, h, w, 3)
    finally:
        os.unlink(raw)

def video_features(path=None, rgb=None, w=160, h=90, diff_thr=8.0, side_frac=0.25,
                   skin=True):
    """One decode per clip -> (coherence, skin mask, dark mask, n_frames).

    coherence = fraction of BACKGROUND pixels displaced between consecutive frames.
    This separates SCOOTER motion from rider or bystander motion: a pedestrian
    moves a patch of the image, a moving scooter displaces the whole scene. Mean
    frame-difference ENERGY cannot tell them apart and mis-reads pedestrians as
    departures. Per-frame median difference removed so exposure drift is not motion.
    """
    A = decode(path, w, h) if rgb is None else rgb
    n = len(A)
    R, Gc, B = A[...,0].astype(np.int16), A[...,1].astype(np.int16), A[...,2].astype(np.int16)
    gray = (R*0.299 + Gc*0.587 + B*0.114).astype(np.float32)
    s = int(w*side_frac)
    bg = np.concatenate([gray[:, :, :s], gray[:, :, -s:]], axis=2)
    D = np.diff(bg, axis=0); D = D - np.median(D, axis=(1,2), keepdims=True)
    coh = np.r_[0.0, (np.abs(D) > diff_thr).mean(axis=(1,2))].astype(np.float32)
    del D, bg
    sk = ((R > 95) & (Gc > 40) & (B > 20) & (R > Gc) & (R > B) & ((R-Gc) > 15)) if skin else None
    dk = gray < 90
    del R, Gc, B, gray
    return coh, sk, dk, n

# ============================================================== frame index

def _score_at(skin, dark, X, Y, off):
    N = len(X); f = np.arange(N) + off
    ok = (f >= 0) & (f < len(skin))
    s = np.full(N, -np.inf, np.float32)
    if ok.sum() == 0: return s
    fo = f[ok]; H, W = skin.shape[1], skin.shape[2]
    hit = np.zeros(ok.sum(), bool)
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            hit |= skin[fo, np.clip(Y[ok,0]+dy, 0, H-1), np.clip(X[ok,0]+dx, 0, W-1)]
    dk = np.zeros(ok.sum(), np.float32)
    for j in (1, 2, 7, 8): dk += dark[fo, Y[ok,j], X[ok,j]]
    s[ok] = hit.astype(np.float32) + 0.5*(dk/4.0)
    return s

def _monotone_align(S):
    """Maximise the summed score over non-decreasing offsets (frames are only ever
    dropped, never reordered)."""
    N, W = S.shape; idx = np.arange(W)
    arg = np.zeros((N, W), np.int32); best = S[0].copy()
    for i in range(1, N):
        pm = np.maximum.accumulate(best)
        arg[i] = np.maximum.accumulate(np.where(best >= pm, idx, -1))
        best = S[i] + pm
    o = int(np.argmax(best)); path = np.zeros(N, np.int32); path[N-1] = o
    for i in range(N-1, 0, -1):
        o = arg[i][o]; path[i-1] = o
    return path

def recover_frame_index(pose_corr, skin, dark, n_frames, nwin=12,
                        prominence_min=0.20, agree_max=60, bounded_s=1.0, fps=30.0):
    """Recover row -> frame for a pose CSV written without a frame index.

    The extractor emits a row only when the detector fired, so the mapping is a
    non-decreasing step function with exactly (n_frames - n_rows) skips. Candidate
    pairings are scored by whether the stored skeleton agrees with the image (nose
    on skin, torso keypoints on dark clothing) and resolved by DP.

    Each window of rows is then re-checked by an INDEPENDENT local offset search;
    windows where that search has a prominent peak agreeing with the DP path are
    marked 'verified', the rest 'interpolated'. Treat interpolated rows as
    segment-level only.
    """
    N = len(pose_corr); D = int(n_frames - N)
    H, W = skin.shape[1], skin.shape[2]
    X = np.clip((pose_corr[[f'x{i}' for i in range(9)]].values*(W-1)).round().astype(np.int32), 0, W-1)
    Y = np.clip((pose_corr[[f'y{i}' for i in range(9)]].values*(H-1)).round().astype(np.int32), 0, H-1)
    if D <= 0:
        return np.arange(N), np.array(['exact']*N, dtype=object), dict(deficit=D, method='identity')
    if D/fps <= bounded_s:
        S = np.stack([_score_at(skin, dark, X, Y, o) for o in range(D+1)], axis=1)
        path = _monotone_align(S)
        return (np.arange(N)+path, np.array(['bounded']*N, dtype=object),
                dict(deficit=D, method='dp', max_error_s=float(D/fps)))
    S = np.stack([_score_at(skin, dark, X, Y, o) for o in range(D+1)], axis=1)
    grid = np.arange(0, D+1, max(1, D//60)); edges = np.linspace(0, N, nwin+1).astype(int)
    # Three candidate mappings. The DP path maximises the total score but can wander
    # through stretches where the rider is static and the score surface is flat; a
    # CONSTANT offset is the right answer when the extractor simply started late
    # (leading trim) or stopped early (trailing drops). Pick by the number of
    # windows an independent local search confirms, not by total score.
    cands = {'dp': _monotone_align(S),
             'const_0': np.zeros(N, np.int32),
             'const_deficit': np.full(N, D, np.int32)}
    wcurves = {}
    for w in range(nwin):
        rows = np.arange(edges[w], edges[w+1])
        wcurves[w] = np.array([float(np.nanmean(np.where(np.isfinite(_score_at(skin, dark, X, Y, o)[rows]),
                                                         _score_at(skin, dark, X, Y, o)[rows], np.nan)))
                               for o in grid])
    scored = {}
    for name, p in cands.items():
        nv, win = 0, []
        for w in range(nwin):
            rows = np.arange(edges[w], edges[w+1]); curve = wcurves[w]
            prom = float(np.nanmax(curve) - np.nanmedian(curve))
            loc = int(grid[np.nanargmax(curve)]); mid = float(np.mean(p[rows]))
            good = (prom >= prominence_min) and (abs(loc-mid) <= agree_max)
            nv += int(good)
            win.append(dict(row_lo=int(edges[w]), row_hi=int(edges[w+1]), dp_mid=mid,
                            local_best=loc, prominence=round(prom, 3), verified=bool(good)))
        ms = float(np.nanmean(np.where(np.isfinite(S[np.arange(N), p]), S[np.arange(N), p], np.nan)))
        scored[name] = dict(path=p, n_verified=nv, mean_score=ms, windows=win)
    pick = max(scored, key=lambda k: (scored[k]['n_verified'], scored[k]['mean_score']))
    sel = scored[pick]; path = sel['path']
    conf = np.array(['interpolated']*N, dtype=object)
    for wd in sel['windows']:
        if wd['verified']: conf[wd['row_lo']:wd['row_hi']] = 'verified'
    return (np.arange(N)+path, conf,
            dict(deficit=D, method='candidates+verify', chosen=pick,
                 windows=sel['windows'], verified_frac=float((conf == 'verified').mean()),
                 candidate_scores={k: dict(n_verified=v['n_verified'],
                                           mean_score=round(v['mean_score'], 3))
                                   for k, v in scored.items()}))

# ============================================================== sync

def _kappa(a, b):
    po = (a == b).mean(); pe = a.mean()*b.mean() + (1-a.mean())*(1-b.mean())
    return (po-pe)/(1-pe) if pe < 1 else 0.0

def _sync_at(imu, coh30, src_fps, move_thr, min_overlap_frac, smooth):
    step = int(round(src_fps/5)); n5 = len(coh30)//step
    coh = pd.Series(coh30[:n5*step].reshape(-1, step).mean(1)
                    ).rolling(smooth, center=True, min_periods=2).mean().values
    vid_mv = coh > move_thr
    imu_dur = float(imu['t_imu'].iloc[-1]); vid_dur = len(coh30)/src_fps
    grid = np.arange(0, imu_dur, 0.2)
    imu_mv = np.interp(grid, imu['t_imu'].values,
                       (~imu['is_stationary'].values).astype(float)) > 0.5
    L = len(vid_mv); need = int(min_overlap_frac*min(L, len(imu_mv)))
    offs, ks = [], []
    for sh in range(-(L-need), len(imu_mv)-need+1):
        i0, j0 = max(0, -sh), max(0, sh); Lo = min(L-i0, len(imu_mv)-j0)
        if Lo < need: continue
        offs.append(sh*0.2); ks.append(_kappa(vid_mv[i0:i0+Lo], imu_mv[j0:j0+Lo]))
    if not offs: return dict(T0=np.nan, status='no feasible overlap')
    offs = np.array(offs); ks = np.array(ks)
    j = int(np.nanargmax(ks)); T0 = float(offs[j])
    far = np.abs(offs-T0) > 10
    margin = float(ks[j] - (np.nanmax(ks[far]) if far.any() else 0))
    near = offs[ks >= ks[j]-0.01]
    still = imu['is_stationary'].values.astype(bool); t = imu['t_imu'].values
    runs, i = [], 0
    while i < len(still):
        if still[i]:
            k2 = i
            while k2 < len(still) and still[k2]: k2 += 1
            if k2-i >= 15: runs.append((t[i], t[k2-1]))
            i = k2
        else: i += 1
    vt = np.arange(len(coh))*0.2 + T0; inm = np.zeros(len(coh), bool)
    for a_, b_ in runs: inm |= (vt >= a_) & (vt <= b_)
    val = (vt >= 0) & (vt <= imu_dur); n_stop = int((inm & val).sum())
    contrast = (float(np.nanmean(coh[inm & val]) - np.nanmean(coh[(~inm) & val]))
                if n_stop >= 50 else np.nan)
    ok = (margin >= GATES['kappa_margin_min'] and n_stop >= 50
          and np.isfinite(contrast) and contrast <= GATES['stopped_contrast_max'])
    return dict(T0=T0, move_thr=float(move_thr), kappa=float(ks[j]), kappa_margin=margin,
                precision_s=float((near.max()-near.min())/2), stopped_contrast=contrast,
                n_stopped_video_samples=n_stop, n_stopped_episodes=len(runs),
                video_dur_s=float(vid_dur), imu_dur_s=float(imu_dur),
                video_moving_frac=float(vid_mv.mean()), imu_moving_frac=float(imu_mv.mean()),
                overlap_s=float(min(imu_dur, T0+vid_dur)-max(0.0, T0)),
                status='ok' if ok else 'weak')

def estimate_sync(imu, coherence_30fps, src_fps=30.0, move_thr=None,
                  min_overlap_frac=0.6, smooth=5, thr_grid=THR_GRID):
    """One global offset T0 = seconds into the IMU log at which the continuous video
    stream begins; NEGATIVE when the camera was started before the logger.

    Aligns the binary 'scooter is moving' state from the video against the IMU's
    own stationary/moving state, maximising Cohen's kappa, so every sample votes.
    Onset matching, which uses only transitions, is degenerate over a wide search
    range because many offsets tie on match count.

    The moving/stopped threshold is scanned and the one with the largest MARGIN
    (peak kappa minus the best competing peak >10 s away) is kept -- not the
    highest kappa, and not the threshold that matches the video and IMU moving
    fractions, both of which fail on sessions where the rider rarely stops.
    """
    if move_thr is not None:
        return _sync_at(imu, coherence_30fps, src_fps, move_thr, min_overlap_frac, smooth)
    cand = []
    for thr in thr_grid:
        r = _sync_at(imu, coherence_30fps, src_fps, thr, min_overlap_frac, smooth)
        if np.isfinite(r.get('kappa_margin', np.nan)): cand.append(r)
    if not cand: return dict(T0=np.nan, status='no feasible overlap')
    cand.sort(key=lambda r: -r['kappa_margin'])
    best = dict(cand[0])
    top = cand[:3]
    best['t0_spread_s'] = float(max(r['T0'] for r in top) - min(r['T0'] for r in top))
    best['n_thresholds_tried'] = len(cand)
    return best

# ============================================================== merge

def build_session(imu_csv, video_paths, pose_paths, frame_wh=(854, 480), src_fps=30.0,
                  frame_index='auto', half_window=3, verbose=True):
    """Run the whole pipeline for one session -> (merged DataFrame, qc dict).

    frame_index: 'auto' decides per clip (use the pose CSV's frame column when it
    has one, recover it otherwise); 'file' requires the column and raises without
    it; 'recover' always recovers, useful for checking recovery against known
    indices. The letterbox y-correction is decided separately per clip.
    """
    W, H = frame_wh
    imu, cal = calibrate_imu(load_imu(imu_csv))
    coh_all, nfr, clips = [], [], []
    for ci, (vp, pp) in enumerate(zip(video_paths, pose_paths), start=1):
        rgb = decode(vp, 160, 90)
        need_skin = pp is not None
        coh, skin, dark, n = video_features(rgb=rgb, skin=need_skin); del rgb
        if pp is None:
            # No pose for this clip, but it is still part of the SAME continuous
            # recording, so its frames must stay in the concatenated motion signal
            # or the offsets of every later clip shift.
            clips.append(dict(idx=ci, n_frames=n, pose=None, frame=None, conf=None,
                              flags=None, diag=dict(deficit=None, method='no pose CSV'),
                              y_corrected=None, frame_index_source='no pose CSV'))
            coh_all.append(coh); nfr.append(n)
            del skin, dark
            if verbose:
                print(f'  clip {ci}: {n} frames, NO POSE CSV - used for sync only')
            continue
        d = pd.read_csv(pp)
        n_raw = len(d)
        # Frame index and y-correction are INDEPENDENT per-file properties, detected
        # per clip. A root holds a mix: pose from the patched extractor carries a
        # frame column, legacy pose does not, and either may still need the
        # letterbox fix depending on when it was exported.
        if frame_index not in ('auto', 'file', 'recover'):
            raise ValueError(f"frame_index must be auto|file|recover, got {frame_index!r}")
        if frame_index == 'auto':
            has_idx = 'frame' in d.columns
        else:
            has_idx = (frame_index == 'file')
            if has_idx and 'frame' not in d.columns:
                raise ValueError(f'{os.path.basename(pp)} has no "frame" column but '
                                 "frame_index='file' was forced")
        needs = pose_needs_correction(d)
        dc = correct_pose(d, H, W) if needs else d.copy()
        if has_idx:
            # the patched extractor writes a row for EVERY frame, blank where the
            # detector did not fire; drop those before merging
            if 'detected' in dc.columns:
                dc = dc[dc['detected'].astype(float) > 0]
            dc = dc.dropna(subset=[f'x{i}' for i in range(9)]).reset_index(drop=True)
            fr = dc['frame'].values.astype(int)
            if 'frame_confidence' in dc.columns:
                # a recovered file carries its own per-row confidence; do not
                # upgrade 'interpolated' rows to 'exact' just because a frame
                # column exists
                conf = dc['frame_confidence'].astype(object).values
            else:
                conf = np.array(['exact']*len(dc), dtype=object)
            vf = float(np.mean([c in ('verified', 'exact', 'bounded') for c in conf]))
            diag = dict(deficit=int(n-len(dc)), method='from file',
                        rows_in_file=int(n_raw), rows_with_pose=int(len(dc)),
                        verified_frac=vf)
        else:
            fr, conf, diag = recover_frame_index(dc, skin, dark, n)
        q = quality_flags(dc)
        if all(f'conf{i}' in dc.columns for i in range(9)):
            cm = dc[[f'conf{i}' for i in range(9)]].median(axis=1)
            q['low_conf'] = (cm < 0.3).values
        clips.append(dict(idx=ci, n_frames=n, pose=dc, frame=fr, conf=conf,
                          flags=q, diag=diag, y_corrected=bool(needs),
                          frame_index_source=diag['method']))
        coh_all.append(coh); nfr.append(n)
        del skin, dark
        if verbose:
            print(f'  clip {ci}: {n} frames, {len(dc)} rows, deficit {n-len(dc)}, '
                  f'frame_index={diag["method"]}, y_corrected={needs}, '
                  f'verified={diag.get("verified_frac", 1.0):.2f}')
    sync = estimate_sync(imu, np.concatenate(coh_all), src_fps=src_fps)
    cum = np.cumsum([0]+nfr)
    T0 = sync.get('T0', np.nan)

    t = imu['t_imu'].values
    out_x = np.full((len(imu), 9), np.nan); out_y = np.full((len(imu), 9), np.nan)
    ch = np.zeros(len(imu), int); nused = np.zeros(len(imu), int)
    fcen = np.full(len(imu), np.nan); fconf = np.array(['none']*len(imu), dtype=object)
    if np.isfinite(T0):
        for c in clips:
            if c['pose'] is None: continue
            ax = np.full((c['n_frames'], 9), np.nan); ay = np.full((c['n_frames'], 9), np.nan)
            cv = np.array(['none']*c['n_frames'], dtype=object)
            good = ~(c['flags']['collapsed'] | c['flags']['off_centre']).values
            fr = np.clip(c['frame'], 0, c['n_frames']-1)
            ax[fr] = np.where(good[:, None], c['pose'][[f'x{i}' for i in range(9)]].values, np.nan)
            ay[fr] = np.where(good[:, None], c['pose'][[f'y{i}' for i in range(9)]].values, np.nan)
            cv[fr] = c['conf']
            t0c = T0 + cum[c['idx']-1]/src_fps
            fc = (t - t0c)*src_fps
            sel = np.flatnonzero((fc >= 0) & (fc < c['n_frames']))
            for i in sel:
                lo = int(max(0, round(fc[i])-half_window))
                hi = int(min(c['n_frames'], round(fc[i])+half_window))
                if hi <= lo: continue
                ch[i] = c['idx']; fcen[i] = fc[i]
                bx = ax[lo:hi]; by = ay[lo:hi]
                k = int(np.sum(~np.isnan(bx[:, 0])))
                if k == 0: continue
                out_x[i] = np.nanmean(bx, axis=0); out_y[i] = np.nanmean(by, axis=0)
                nused[i] = k
                vals = [x for x in cv[lo:hi] if x != 'none']
                fconf[i] = ('verified' if vals and sum(v in ('verified','exact','bounded') for v in vals) > len(vals)/2
                            else ('interpolated' if vals else 'none'))
    m = imu.copy()
    m['pose_chunk'] = np.where(ch > 0, ch, pd.NA)
    m['pose_frame'] = np.where(np.isnan(fcen), pd.NA, np.round(fcen))
    m['pose_frames_used'] = nused
    m['pose_frame_confidence'] = fconf
    m['sync_uncertainty_s'] = np.where(ch > 0, sync.get('precision_s', np.nan), np.nan)
    for i in range(9):
        m[f'pose_x{i}'] = out_x[:, i]; m[f'pose_y{i}'] = out_y[:, i]
    px = out_x*W; py = out_y*H
    sx, sy = (px[:,1]+px[:,2])/2, (py[:,1]+py[:,2])/2
    hx, hy = (px[:,7]+px[:,8])/2, (py[:,7]+py[:,8])/2
    m['trunk_lean_deg'] = np.degrees(np.arctan2(sx-hx, hy-sy))
    m['shoulder_tilt_deg'] = np.degrees(np.arctan2(py[:,1]-py[:,2], px[:,1]-px[:,2]))
    m['shoulder_width_px'] = np.hypot(px[:,1]-px[:,2], py[:,1]-py[:,2])

    _fl = [c['flags']['off_centre'].values for c in clips if c['flags'] is not None]
    off_centre = float(np.mean(np.concatenate(_fl))) if _fl else float('nan')
    cov = float((nused > 0).mean())
    qc = dict(n_imu=len(imu), imu_duration_s=float(imu['t_imu'].iloc[-1]),
              video_frames=nfr, video_duration_s=float(cum[-1]/src_fps),
              pose_rows=[(len(c['pose']) if c['pose'] is not None else None) for c in clips],
              deficit=[(int(c['diag']['deficit']) if c['diag']['deficit'] is not None else None)
                       for c in clips],
              clips_without_pose=[c['idx'] for c in clips if c['pose'] is None],
              y_corrected=[c['y_corrected'] for c in clips],
              frame_index_source=[c['frame_index_source'] for c in clips],
              frame_index=[{k: v for k, v in c['diag'].items() if k != 'windows'} for c in clips],
              pose_coverage=cov, off_centre_frac=off_centre, **cal, sync=sync)
    qc['gates'] = dict(
        acc_scale_ok=bool(GATES['acc_scale'][0] <= 1/cal['acc_scale'] <= GATES['acc_scale'][1]),
        stationary_ok=bool(cal['stationary_frac'] >= GATES['stationary_frac_min']),
        sync_ok=bool(sync.get('status') == 'ok'),
        pose_coverage_ok=bool(cov >= GATES['pose_coverage_min']),
        off_centre_ok=bool(off_centre <= GATES['off_centre_max']),
        frame_index_ok=bool(all(c['diag'].get('verified_frac', 1.0) >= 0.5
                                for c in clips if c['pose'] is not None)))
    qc['usable'] = bool(all(qc['gates'].values()))
    return m, qc


# ============================================================== discovery + CLI

def _front_index(name):
    # Match on the STEM: 'p30_front_!.mp4' must not resolve to clip 4 via the '4'
    # in the extension. Digits must follow 'front' with only separators between.
    stem = os.path.splitext(name)[0]
    m = re.search(r'front\D*?(\d+)', stem, re.I)   # front_1, FrontView_1, front 1
    return int(m.group(1)) if m else None

def _norm(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())

IMU_SIGNATURE = {'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'}


def _is_imu_csv(path):
    """True when the header carries the accelerometer/gyroscope columns."""
    try:
        with open(path, 'r', errors='ignore') as fh:
            head = fh.readline()
    except Exception:
        return False
    cols = {c.strip().strip('"').lower() for c in head.split(',')}
    return IMU_SIGNATURE <= cols


VIDEO_DIR_ALIASES = ('processedvideos', 'processvideos')
MOTION_DIR_ALIASES = ('videoandmotiondata', 'videomotiondata')

def _find_dir(root, aliases, max_depth=2):
    """Locate a folder by normalised name at depth 0..max_depth below root.

    Real sessions vary: the participant level sits above the folder in some
    ('<session>/P011/processed videos') and below it in others
    ('<session>/processed videos/P66'), and the motion folder appears both as
    'video and motion data' and 'video_and_motion_data'.
    """
    hits = []
    for d in range(max_depth+1):
        pat = os.path.join(root, *(['*']*d)) if d else root
        for p in sorted(glob.glob(os.path.join(pat, '*')) if d else [root]):
            pass
        cands = sorted(glob.glob(os.path.join(root, *(['*']*d), '*'))) if d == 0 else \
                sorted(glob.glob(os.path.join(root, *(['*']*d), '*')))
        for p in cands:
            if os.path.isdir(p) and _norm(os.path.basename(p)) in aliases:
                hits.append(p)
        if hits: break
    return hits


def _participant(vid_dir, video_paths):
    """Participant id (P\\d+). The folder sits ABOVE the videos folder in some
    sessions ('<session>/P011/processed videos') and BELOW it in others
    ('<session>/processed videos/P66'), so check both, then fall back to the
    clip filenames ('p66_front_1.mp4').
    """
    up = os.path.basename(os.path.dirname(vid_dir))
    if re.fullmatch(r'[Pp]\d+', up): return up.upper()
    for p in video_paths:
        d = os.path.basename(os.path.dirname(p))
        if re.fullmatch(r'[Pp]\d+', d): return d.upper()
    for p in video_paths:
        m = re.match(r'[Pp](\d+)[_\- ]', os.path.basename(p))
        if m: return 'P' + m.group(1).zfill(3)
    return ''


def discover(pose_path, video_motion_path):
    """Locate the four inputs from the two directory arguments.

    pose_path          *front*.csv pose files, one per GoPro clip
    video_motion_path  a 'processed videos' folder (at session or participant level,
                       searched to depth 2) holding *front*.mp4, and a
                       'video and motion data' folder holding the IMU CSV and the
                       webcam MP4.

    Clips are ordered by the integer in 'front_<n>' and pose files matched to videos
    by that integer, not by sort position; 'side' files are ignored. The IMU CSV is
    identified as the one SHARING ITS STEM with the webcam MP4 -- some motion folders
    also hold keypoint and task-labelling CSVs, so "the only .csv" is not a safe rule.
    """
    vds = _find_dir(video_motion_path, VIDEO_DIR_ALIASES)
    if not vds:
        raise FileNotFoundError(f'no "processed videos" folder under {video_motion_path}')
    mds = _find_dir(video_motion_path, MOTION_DIR_ALIASES)
    if not mds:
        # Older sessions keep the IMU CSV at participant level with no motion folder.
        mds = sorted({os.path.dirname(p) for p in
                      glob.glob(os.path.join(video_motion_path, '*', '*.csv'))
                      if _norm(os.path.basename(os.path.dirname(p))) not in VIDEO_DIR_ALIASES})
    if not mds:
        raise FileNotFoundError(f'no "video and motion data" folder under {video_motion_path}')
    if len(vds) > 1 or len(mds) > 1:
        raise FileNotFoundError(f'ambiguous folders: videos={vds}, motion={mds}')
    vid_dir, vm_dir = vds[0], mds[0]

    vids = {}; unindexed = []
    for f in glob.glob(os.path.join(vid_dir, '**', '*.mp4'), recursive=True):
        b = os.path.basename(f)
        if 'front' not in b.lower() or 'side' in b.lower(): continue
        i = _front_index(b)
        if i is None:
            unindexed.append(b); continue
        if i in vids: raise FileNotFoundError(f'two clips numbered {i}: {vids[i]}, {f}')
        vids[i] = f
    if not vids:
        raise FileNotFoundError(f'no *front*.mp4 under {vid_dir}')

    poses = {}
    for f in glob.glob(os.path.join(pose_path, '**', '*.csv'), recursive=True):
        b = os.path.basename(f)
        if 'front' not in b.lower() or 'side' in b.lower(): continue
        i = _front_index(b)
        if i is None: continue
        if i in poses: raise FileNotFoundError(f'two pose files numbered {i}: {poses[i]}, {f}')
        poses[i] = f
    if not poses:
        raise FileNotFoundError(f'no *front*.csv under {pose_path}')
    missing = sorted(set(vids) - set(poses))
    if not (set(poses) & set(vids)):
        raise FileNotFoundError(f'no pose CSV matches any clip in {pose_path}')
    extra = sorted(set(poses) - set(vids))

    mp4s = sorted(glob.glob(os.path.join(vm_dir, '*.mp4')))
    csvs = sorted(glob.glob(os.path.join(vm_dir, '*.csv')))
    if not csvs:
        raise FileNotFoundError(f'no CSV at all in {vm_dir}')
    # Identify the IMU by its COLUMN SIGNATURE, not its filename. Motion folders
    # also hold task-labelling and keypoint CSVs - up to 13 of them - so "the only
    # .csv" and stem-matching against the webcam both fail there, and the latter
    # cannot distinguish "ambiguous" from "absent".
    imu_like = [c for c in csvs if _is_imu_csv(c)]
    if len(imu_like) == 1:
        imu_csv = imu_like[0]
    elif not imu_like:
        raise FileNotFoundError(
            f'no IMU CSV in {vm_dir}: none of the {len(csvs)} CSVs has the expected '
            f'columns {sorted(IMU_SIGNATURE)} (they look like label/keypoint files)')
    else:
        stem_hit = [c for c in imu_like for m in mp4s
                    if os.path.splitext(os.path.basename(m))[0]
                    == os.path.splitext(os.path.basename(c))[0]]
        if len(stem_hit) == 1:
            imu_csv = stem_hit[0]
        else:
            raise FileNotFoundError(f'{len(imu_like)} CSVs in {vm_dir} look like IMU '
                                    f'data: {[os.path.basename(c) for c in imu_like]}')
    stem = os.path.splitext(os.path.basename(imu_csv))[0]
    same = [m for m in mp4s if os.path.splitext(os.path.basename(m))[0] == stem]
    webcam = same[0] if same else (mp4s[0] if len(mp4s) == 1 else None)

    order = sorted(vids)
    part = _participant(vid_dir, [vids[i] for i in order])
    return dict(clips=order,
                video_paths=[vids[i] for i in order],
                pose_paths=[poses.get(i) for i in order],
                clips_without_pose=missing, unindexed_videos=unindexed,
                imu_csv=imu_csv, webcam=webcam,
                participant=part,
                session=os.path.basename(os.path.normpath(video_motion_path)),
                unmatched_pose_clips=extra)


def probe_frames(path):
    if not FFPROBE:
        return None
    try:
        out = subprocess.run([FFPROBE,'-v','error','-select_streams','v:0',
                              '-count_packets','-show_entries','stream=nb_read_packets',
                              '-of','csv=p=0', path], capture_output=True, text=True, check=True)
        return int(out.stdout.strip().split(',')[0])
    except Exception:
        return None

def find_sessions(pose_root, vm_root):
    """Session folders present under BOTH roots, matched by folder name.

    <pose_root>/<session>/            *front*.csv
    <vm_root>/<session>/<participant>/{processed videos, video and motion data}
    Names present under only one root are reported, not silently dropped.
    """
    def subdirs(root):
        return {os.path.basename(p.rstrip(os.sep)): p
                for p in sorted(glob.glob(os.path.join(root, '*'))) if os.path.isdir(p)}
    P, V = subdirs(pose_root), subdirs(vm_root)
    common = sorted(set(P) & set(V))
    return ([(n, P[n], V[n]) for n in common],
            sorted(set(P) - set(V)), sorted(set(V) - set(P)))


SUMMARY_COLS = ['session', 'participant', 'n_clips', 'imu_duration_s', 'video_duration_s',
                'T0_s', 'precision_s', 'kappa', 'kappa_margin', 'stopped_contrast',
                'sync_status', 'pose_coverage', 'off_centre_frac', 'acc_scale',
                'stationary_frac', 'mount_tilt_deg', 'min_verified_frac', 'deficits',
                'webcam_matches_imu', 'clips_without_pose', 'usable', 'failed_gates', 'error']


def _summary_row(tag, qc, err=None):
    if err is not None:
        return dict(session=tag, error=str(err)[:300])
    s = qc.get('sync', {})
    vf = [c.get('verified_frac', 1.0) for c in qc.get('frame_index', [])
          if c.get('method') != 'no pose CSV']
    return dict(session=tag, participant=qc.get('inputs', {}).get('participant'),
                n_clips=len(qc.get('video_frames', [])),
                imu_duration_s=round(qc.get('imu_duration_s', float('nan')), 1),
                video_duration_s=round(qc.get('video_duration_s', float('nan')), 1),
                T0_s=s.get('T0'), precision_s=s.get('precision_s'), kappa=s.get('kappa'),
                kappa_margin=s.get('kappa_margin'), stopped_contrast=s.get('stopped_contrast'),
                sync_status=s.get('status'), pose_coverage=round(qc.get('pose_coverage', float('nan')), 3),
                off_centre_frac=round(qc.get('off_centre_frac', float('nan')), 3),
                acc_scale=round(qc.get('acc_scale', float('nan')), 4),
                stationary_frac=round(qc.get('stationary_frac', float('nan')), 3),
                mount_tilt_deg=round(qc.get('mount_tilt_deg', float('nan')), 1),
                min_verified_frac=round(min(vf), 2) if vf else None,
                deficits=';'.join('-' if d is None else str(d) for d in qc.get('deficit', [])),
                clips_without_pose=';'.join(str(i) for i in qc.get('clips_without_pose', [])) or None,
                webcam_matches_imu=qc.get('webcam', {}).get('matches_imu_rows'),
                usable=qc.get('usable'),
                failed_gates=';'.join(k for k, v in qc.get('gates', {}).items() if not v),
                error=None)


def process_one(pose_dir, vm_dir, a, tag=None):
    """One session -> (tag, qc). Writes <out>/<tag>_merged.csv and <tag>_qc.json."""
    W, H = (int(v) for v in a.frame_size.lower().split('x'))
    f = discover(pose_dir, vm_dir)
    if tag is None:
        tag = (f['session'] if f['participant'] in ('', f['session'])
               else f'{f["session"]}_{f["participant"]}')
    print(f'session {tag}: {len(f["clips"])} clips {f["clips"]}')
    for i, (v, p) in enumerate(zip(f['video_paths'], f['pose_paths']), start=1):
        print(f'  clip {i}: {os.path.basename(v)}  <-  '
              f'{os.path.basename(p) if p else "(no pose CSV)"}')
    print(f'  IMU    : {os.path.basename(f["imu_csv"])}')
    print(f'  webcam : {os.path.basename(f["webcam"]) if f["webcam"] else "(none - QC check skipped)"}')
    if f['unmatched_pose_clips']:
        print(f'  note   : pose files with no matching video, ignored: {f["unmatched_pose_clips"]}')
    if f.get('unindexed_videos'):
        print(f'  WARNING: clip file(s) with no readable number, EXCLUDED: '
              f'{f["unindexed_videos"]} - check for a filename typo')
    if f.get('clips_without_pose'):
        print(f'  note   : clip(s) {f["clips_without_pose"]} have video but NO pose CSV - '
              'kept in the sync signal, excluded from the merge')
    merged, qc = build_session(f['imu_csv'], f['video_paths'], f['pose_paths'],
                               frame_wh=(W, H), src_fps=a.fps,
                               frame_index=a.frame_index)
    qc['session'] = tag
    qc['inputs'] = {k: (v if isinstance(v, (str, list)) else str(v))
                    for k, v in f.items() if k != 'clips'}
    if not a.skip_webcam_check and f['webcam']:
        n = probe_frames(f['webcam'])
        qc['webcam'] = dict(frames=n, imu_rows=int(qc['n_imu']),
                            matches_imu_rows=(n == qc['n_imu']) if n else None)
        if n is not None and n != qc['n_imu']:
            print(f'  WARNING: webcam has {n} frames but the IMU has {qc["n_imu"]} rows; '
                  'these are normally equal (one sample per captured frame) - check the pairing')
    os.makedirs(a.out, exist_ok=True)
    merged.to_csv(os.path.join(a.out, f'{tag}_merged.csv'), index=False)
    with open(os.path.join(a.out, f'{tag}_qc.json'), 'w') as fh:
        json.dump(qc, fh, indent=2, default=str)
    def _f(x, n=2):
        return 'n/a' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{n}f}'
    s = qc['sync']
    print(f'  T0 = {_f(s.get("T0"), 1)} s (+-{_f(s.get("precision_s"), 1)} s), '
          f'kappa {_f(s.get("kappa"), 3)}, margin {_f(s.get("kappa_margin"), 3)}, '
          f'status {s.get("status")}')
    print(f'  pose coverage {qc["pose_coverage"]:.2f} | ' +
          ', '.join(f'{k}={"PASS" if v else "FAIL"}' for k, v in qc['gates'].items()))
    print(f'  VERDICT: {"USABLE" if qc["usable"] else "FLAGGED"}')
    return tag, qc


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Preprocess mobility-scooter sessions: calibrate the IMU, correct '
                    'and frame-index the pose, synchronise, merge. Give either one '
                    'session or the two ROOTS, in which case every session folder '
                    'present under both roots is processed.')
    ap.add_argument('pose_path', help='pose root, or one session directory of *front*.csv')
    ap.add_argument('video_motion_path',
                    help='video/motion root, or one session directory holding '
                         '<participant>/{processed videos, video and motion data}')
    ap.add_argument('-o', '--out', default='.', help='output directory (default: .)')
    ap.add_argument('--fps', type=float, default=30.0, help='GoPro frame rate (default 30)')
    ap.add_argument('--frame-size', default='854x480', help='GoPro WxH (default 854x480)')
    ap.add_argument('--frame-index', choices=('auto', 'file', 'recover'), default='auto',
                    help='auto (default): per CLIP, use the frame column when the pose '
                         'CSV has one and recover it otherwise - a root may hold both; '
                         'file: require a frame column; recover: always recover')
    ap.add_argument('--skip-webcam-check', action='store_true')
    ap.add_argument('--ffmpeg', default=None, help='path to the ffmpeg binary (default: PATH)')
    ap.add_argument('--ffprobe', default=None, help='path to the ffprobe binary (default: PATH)')
    ap.add_argument('--strict', action='store_true', help='exit 1 if any session fails its gates')
    ap.add_argument('--batch', action='store_true', help='force batch mode over the two roots')
    ap.add_argument('--single', action='store_true', help='force single-session mode')
    ap.add_argument('--sessions', default=None,
                    help='comma-separated session folder names to restrict a batch to')
    ap.add_argument('--limit', type=int, default=None,
                    help='process only the first N sessions (use for a dry run)')
    ap.add_argument('--skip-existing', action='store_true',
                    help='skip sessions whose _qc.json already exists (resume)')
    ap.add_argument('--summary', default=None,
                    help='summary CSV path (default <out>/sessions_summary.csv)')
    a = ap.parse_args(argv)
    try:
        check_tools(a.ffmpeg, a.ffprobe)
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr); return 2

    batch = a.batch
    if not batch and not a.single:
        try:                                    # single session if the layout resolves
            discover(a.pose_path, a.video_motion_path); batch = False
        except Exception:
            batch = True
    if not batch:
        try:
            _, qc = process_one(a.pose_path, a.video_motion_path, a)
        except Exception as e:
            print(f'FAILED: {e}'); return 1
        print(f'wrote {a.out}/{qc["session"]}_merged.csv and _qc.json')
        return 0 if (qc['usable'] or not a.strict) else 1

    sessions, only_pose, only_vm = find_sessions(a.pose_path, a.video_motion_path)
    if a.sessions:
        keep = {x.strip() for x in a.sessions.split(',')}
        sessions = [s for s in sessions if s[0] in keep]
    print(f'{len(sessions)} session folders under both roots'
          + (f' | {len(only_pose)} pose-only' if only_pose else '')
          + (f' | {len(only_vm)} video-only' if only_vm else ''))
    if only_pose: print(f'  pose-only (no video/motion): {only_pose[:10]}{" ..." if len(only_pose) > 10 else ""}')
    if only_vm:   print(f'  video-only (no pose): {only_vm[:10]}{" ..." if len(only_vm) > 10 else ""}')
    if a.limit: sessions = sessions[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    summary_path = a.summary or os.path.join(a.out, 'sessions_summary.csv')

    rows, t_start = [], time.time()
    for i, (name, pdir, vdir) in enumerate(sessions, start=1):
        done = sorted(glob.glob(os.path.join(a.out, f'{name}*_qc.json')))
        if a.skip_existing and done:
            print(f'[{i}/{len(sessions)}] {name}: skipped (exists)')
            try:
                rows.append(_summary_row(name, json.load(open(done[0]))))
            except Exception as e:
                rows.append(_summary_row(name, {}, err=f'reread failed: {e}'))
            continue
        el = time.time()-t_start
        eta = (el/(i-1)*(len(sessions)-i+1)) if i > 1 else float('nan')
        print(f'\n[{i}/{len(sessions)}] {name}'
              + (f'  (elapsed {el/60:.0f} min, eta {eta/60:.0f} min)' if i > 1 else ''))
        try:
            tag, qc = process_one(pdir, vdir, a, tag=None)
            rows.append(_summary_row(tag, qc))
        except Exception as e:
            print(f'  FAILED: {type(e).__name__}: {e}')
            rows.append(_summary_row(name, {}, err=f'{type(e).__name__}: {e}'))
        finally:
            gc.collect()
        pd.DataFrame(rows).reindex(columns=SUMMARY_COLS).to_csv(summary_path, index=False)

    df = pd.DataFrame(rows).reindex(columns=SUMMARY_COLS)
    df.to_csv(summary_path, index=False)
    # counted from the table, not accumulated during the loop: sessions reused via
    # --skip-existing contribute rows without passing through the processing branch
    n_err = int(df['error'].notna().sum())
    n_ok = int((df['usable'] == True).sum())
    print(f'\n{len(df)} sessions: {n_ok} usable, {len(df)-n_ok-n_err} flagged, {n_err} errored '
          f'in {(time.time()-t_start)/60:.0f} min')
    print(f'summary -> {summary_path}')
    return 0 if (n_err == 0 and (n_ok == len(df) or not a.strict)) else 1


if __name__ == '__main__':
    sys.exit(main())
