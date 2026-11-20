"""
healthy_signature.py

This is series of functions to create the healthy movement signature for the PADS smartwatch dataset.
The steps include preparation -> features -> unit table -> one normative model per (task, wrist role)
-> out-of-fold calibration -> frozen bundle; plus scoring of new participants (e.g. PD).
The SAME functions are used for healthy and PD participants.
"""
import numpy as np
import pandas as pd
import joblib
import scipy, sklearn
from joblib import Parallel, delayed
from scipy.signal import welch
from scipy.stats import median_abs_deviation
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

# ------CONFIGURATION -------- 
#Details on data organization and thresholds set for later functions 
FS = 100                                       #sampling rate
END = 974                                      # keep first 974 samples, the last 50 are padding added upstream
TRIM_S = {"CrossArms": 3.0, "LiftHold": 3.0}   # seconds dropped from start; two tasks had a median mvmt start at 3s, other tasks 0
LO, HI = 0.5, 20.0                             #band thresholds
NPERSEG = 256                                  #samples per segment, as per Welch
BANDS = {"0.5-3": (0.5, 3), "3-7": (3, 7), "4-6": (4, 6), "7-12": (7, 12), "12-20": (12, 20)} #band power bands
REL_BANDS = ["4-6", "3-7"]                    #relative band power bands
EPS = 1e-12                                   #constnat added since log values are being used

_PER_SENSOR = ["rms", "range95", "jerk", "bp_0.5-3", "bp_3-7", "bp_4-6", "bp_7-12", "bp_12-20",
               "rel_4-6", "rel_3-7", "spec_entropy"]
FEATURES = [f"{p}_{n}" for p in ("acc", "gyr") for n in _PER_SENSOR]   
COVS = ["age", "sex_male"]                  #used for ridge regression step
MIN_N = 30
RIDGE_ALPHA = 10.0                          #ridge regression alpha, pulls age/sex coefficients toward 0 to be cautious
CLIP = 8.0                                  #clipping threshold used after scaling
Z_CLIP = (-3.0, 10.0)                       #z-score threshold range
N_SPLITS, N_REPEATS, SEED = 5, 3, 0
REQUIRED_META = ["subject_id", "task", "role", "seg", "label", "age", "gender"]


def current_config():
    return dict(FS=FS, END=END, TRIM_S=dict(TRIM_S), LO=LO, HI=HI, NPERSEG=NPERSEG, BANDS=BANDS,
                REL_BANDS=REL_BANDS, FEATURES=list(FEATURES), COVS=list(COVS), MIN_N=MIN_N,
                RIDGE_ALPHA=RIDGE_ALPHA, CLIP=CLIP, Z_CLIP=Z_CLIP)


# ----- LOADING DATA ------
##This reads the segments array and metadata, ensures all data present
def load_data(segments_path, index_path):
    X = np.load(segments_path)["X"] if str(segments_path).endswith(".npz") else np.load(segments_path)
    meta = pd.read_csv(index_path)
    meta["subject_id"] = meta["subject_id"].astype(str).str.zfill(3)
    missing = [c for c in REQUIRED_META if c not in meta.columns]
    assert not missing, f"segment index is missing columns: {missing}"
    assert len(meta) == X.shape[0], "array and metadata are misaligned"
    assert X.shape[2] == 6, "expected 6 channels: acc x/y/z, gyr x/y/z"
    return X, meta


# ----- PREPARATION AND FEATURE EXTRACTION -----
##This account for segment padding and movement onset differences
def prep_segment(seg, task):
    start = int(TRIM_S.get(task, 0.0) * FS)
    return np.asarray(seg[start:END, :], dtype=np.float64)

#This step is computing the necessary features per sensor
def sensor_features(seg, p):
    f, P = welch(seg, fs=FS, nperseg=NPERSEG, axis=0)
    Pt = P.sum(axis=1)                          # sum over accel x/y/z, independent of watch rotation
    df = f[1] - f[0]
    keep = (f >= LO) & (f <= HI)
    total = Pt[keep].sum() * df
    out = {}
    out[f"{p}_rms"] = np.log10(np.sqrt(np.mean(np.sum(seg ** 2, axis=1))) + EPS)  ##RMS
    lo95, hi95 = np.percentile(seg, [5, 95], axis=0)
    out[f"{p}_range95"] = np.log10(np.sqrt(np.sum((hi95 - lo95) ** 2)) + EPS) ##range of amp
    out[f"{p}_jerk"] = np.log10(np.sqrt(np.mean(np.sum((np.diff(seg, axis=0) * FS) ** 2, axis=1))) + EPS) #jerk; how fast signal changes
    
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        bp = Pt[m].sum() * df
        out[f"{p}_bp_{name}"] = np.log10(bp + EPS) #band power
        if name in REL_BANDS: #band power for 3-7 and 4-6 ranges
            out[f"{p}_rel_{name}"] = bp / (total + EPS)
    q = Pt[keep] / (Pt[keep].sum() + EPS)
    out[f"{p}_spec_entropy"] = float(-(q * np.log(q + EPS)).sum() / np.log(keep.sum())) #spectral entropy
    return out

#This matches one movement segment to one set of features 
def segment_features(seg):
    return {**sensor_features(seg[:, :3], "acc"), **sensor_features(seg[:, 3:], "gyr")}

#This prepares the segment
def _one(seg, task):
    return segment_features(prep_segment(seg, task))

#Run for each segment to extract features 
def extract_features(X, meta, n_jobs=-1):
    tasks = meta["task"].to_numpy()
    rows = Parallel(n_jobs=n_jobs)(delayed(_one)(X[i], tasks[i]) for i in range(len(X)))
    return pd.DataFrame(rows)

#Segment features averaged within each person,task and wrist dominance; age and sex are attached
def build_unit_table(meta, F):
    feat = pd.concat([meta.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    unit = feat.groupby(["subject_id", "task", "role"])[F.columns.tolist()].mean().reset_index()
    person = meta.drop_duplicates("subject_id").set_index("subject_id")[["age", "gender"]].copy()
    person["sex_male"] = (person["gender"].astype(str).str.strip().str.lower() == "male").astype(float)
    unit = unit.merge(person[COVS], left_on="subject_id", right_index=True, how="left")
    return unit, person


# ----- NORMATIVE MODEL CREATION ------
#A model is created in 5 stages per unit; stages outlined in report
class NormativeModel:
    """Healthy-only model for one (task, wrist role) unit: ridge removes the age/sex effect,
    robust scaling + clipping, Gaussian with shrinkage covariance, score = log squared Mahalanobis."""

    def __init__(self, use_covariates=True, alpha=RIDGE_ALPHA, clip=CLIP):
        self.use_cov, self.alpha, self.clip = use_covariates, alpha, clip

    def _cz(self, C):
        C = np.where(np.isnan(C), self.cfill_, C)
        C[:, 0] = np.clip(C[:, 0], self.amin_, self.amax_)       # never extrapolate age
        return (C - self.cmu_) / self.csd_

    def fit(self, X, C):
        X, C = np.asarray(X, float), np.asarray(C, float).copy()
        self.imp_ = SimpleImputer(strategy="median").fit(X)     #Median imputation step
        X = self.imp_.transform(X)
        self.cfill_ = np.nan_to_num(np.nanmedian(C, axis=0))
        C = np.where(np.isnan(C), self.cfill_, C)
        self.amin_, self.amax_ = C[:, 0].min(), C[:, 0].max()
        self.cmu_ = C.mean(0)
        sd = C.std(0)
        self.csd_ = np.where(sd < 1e-8, 1.0, sd)
        if self.use_cov:
            self.reg_ = Ridge(alpha=self.alpha).fit((C - self.cmu_) / self.csd_, X)  #Ridge regression age/sex adjustment
            X = X - self.reg_.predict((C - self.cmu_) / self.csd_)
        self.sc_ = RobustScaler().fit(X)      #Scaling and clipping 
        Z = np.clip(self.sc_.transform(X), -self.clip, self.clip)
        self.cov_ = LedoitWolf().fit(Z)       #Gaussian model and covariance shrinkage
        return self

    def score(self, X, C):
        X = self.imp_.transform(np.asarray(X, float))
        if self.use_cov:
            X = X - self.reg_.predict(self._cz(np.asarray(C, float).copy()))
        Z = np.clip(self.sc_.transform(X), -self.clip, self.clip)
        return np.log(self.cov_.mahalanobis(Z) + 1e-9) #Deviation score - not computed at this step


def fit_units(d, features=FEATURES, min_n=MIN_N, use_cov=True):  #Splits the unit table by task and wrist role and fits model per pair
    return {u: NormativeModel(use_cov).fit(g[features], g[COVS])
            for u, g in d.groupby(["task", "role"]) if len(g) >= min_n}


def score_units(models, d, features=FEATURES):   #This reverses the above - for each group, it finds its associated model and scores those within it
    out = []
    for u, g in d.groupby(["task", "role"]):
        if u in models:
            out.append(pd.DataFrame({"subject_id": g["subject_id"].values, "task": u[0], "role": u[1],
                                     "raw": models[u].score(g[features], g[COVS])}))
    return pd.concat(out, ignore_index=True)


# ---- OUT OF FOLD SCORES AND CALIBRATION -------
## Split healthy participants by person into K=5 folds
##In each fold, models are fit on the other four folds (n = 63) and score the leftover people
##repeated 3 times with different splits and average scores 
def oof_scores(d, features=FEATURES, n_splits=N_SPLITS, n_repeats=N_REPEATS, use_cov=True, seed=SEED):
    ids = d["subject_id"].unique()
    res = []
    for r in range(n_repeats):
        for tr, te in KFold(n_splits, shuffle=True, random_state=seed + r).split(ids):
            m = fit_units(d[d["subject_id"].isin(ids[tr])], features, use_cov=use_cov)
            res.append(score_units(m, d[d["subject_id"].isin(ids[te])], features))
    return pd.concat(res).groupby(["subject_id", "task", "role"], as_index=False)["raw"].mean()

#Change these score to z-scores using median and SD from out-of-fold values
def to_z(oof):
    cal = oof.groupby(["task", "role"])["raw"].agg(
        mu="median", sd=lambda s: max(median_abs_deviation(s, scale="normal"), 1e-6))
    oof = oof.join(cal, on=["task", "role"])
    oof["z"] = ((oof["raw"] - oof["mu"]) / oof["sd"]).clip(*Z_CLIP)
    return oof, cal


# ------ FREEZE MODEL -----
def build_healthy_model(X, meta, out_path="healthy_model_v2.joblib", n_jobs=-1):
    h = (meta["label"] == 0).to_numpy() ##This keeps only the healthy participants
    X0, meta0 = X[h], meta[h].reset_index(drop=True)
    F = extract_features(X0, meta0, n_jobs)
    unit, person = build_unit_table(meta0, F)
    oof, cal = to_z(oof_scores(unit))
    subj = oof.groupby("subject_id")["z"].agg(deviation="mean", n_units="count").join(person)
    models = fit_units(unit)
    bundle = dict(models=models, calibration=cal, features=list(FEATURES), covariates=list(COVS),
                  config=current_config(),
                  versions=dict(numpy=np.__version__, scipy=scipy.__version__,
                                sklearn=sklearn.__version__, pandas=pd.__version__))
    joblib.dump(bundle, out_path)
    return bundle, unit, oof, subj


# ----- SCORING NEW PARTICIPANTS (e.g. PD, new input) --------
def score_participants(bundle, X, meta, n_jobs=-1):
    assert bundle["config"] == current_config(), "preprocessing/config differs from the frozen model"
    F = extract_features(X, meta, n_jobs)
    unit, person = build_unit_table(meta, F)
    raw = score_units(bundle["models"], unit, bundle["features"])
    raw = raw.join(bundle["calibration"], on=["task", "role"])
    raw["z"] = ((raw["raw"] - raw["mu"]) / raw["sd"]).clip(*Z_CLIP)
    subj = raw.groupby("subject_id")["z"].agg(deviation="mean", n_units="count").join(person)
    return raw, subj
