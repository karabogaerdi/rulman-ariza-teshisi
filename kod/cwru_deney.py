"""
Bienefeld ve ark. (2023, Entropy 25:1278) makalesinin tekrar üretimi
ve veri sızıntısı (data leakage) testi.

Deneyler
  D1  Makalenin protokolü: 2 hp, 10 sınıf, rastgele tabakalı 5 katlı CV
  D1b Aynı 10 sınıf, ama her kaydın ilk %80'i eğitim / son %20'si test (zamansal blok)
  D2  Sızıntısız: 4 sınıf (Sağlam/İç/Dış/Bilye), bir hata boyutu hiç görülmeden test
  D3  D2 + önerimiz: göreli bant enerjisi (bantlar toplam enerjiye bölünür)
  D5  Yük genellemesi: 2 hp ile eğit, 0/1/3 hp ile test (10 sınıf)

Kullanım:  python3 kod/cwru_deney.py data/CWRU-dataset
"""
import sys, glob, os, json
import numpy as np
import scipy.io as sio
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/CWRU-dataset"
FS = 48_000
WIN = 1670            # makaledeki paket boyu
N_TOTAL = 467_600     # kayıt başına kullanılan örnek -> 280 pencere
SEEDS = [0, 1, 2]

# ------------------------------------------------------------------ veri
def find(pattern):
    hits = glob.glob(os.path.join(ROOT, pattern), recursive=True)
    assert len(hits) == 1, (pattern, hits)
    return hits[0]

def load_de(path):
    m = sio.loadmat(path)
    num = os.path.basename(path).split("_")[0].split("@")[0]
    key = f"X{int(num):03d}_DE_time"      # 99_Normal_2.mat içinde X098 de var
    if key not in m:                      # 174.mat içindeki kayıt X173 olarak adlandırılmış
        keys = [k for k in m if k.endswith("_DE_time")]
        assert len(keys) == 1, (path, keys); key = keys[0]
    return m[key].ravel().astype(np.float64)

FAULT_DIRS = {
    "IR": "48k_Drive_End_Bearing_Fault_Data/IR/{s}/*_{l}.mat",
    "OR": "48k_Drive_End_Bearing_Fault_Data/OR/{s}/**/*_{l}.mat",
    "B":  "48k_Drive_End_Bearing_Fault_Data/B/{s}/*_{l}.mat",
}
NORMAL = {0: "Normal/97_Normal_0.mat", 1: "Normal/98_Normal_1.mat",
          2: "Normal/99_Normal_2.mat", 3: "Normal/100_Normal_3.mat"}
SIZES = ["007", "014", "021"]

def recording(kind, size, load):
    if kind == "N":
        return load_de(os.path.join(ROOT, NORMAL[load]))
    pat = FAULT_DIRS[kind].format(s=size, l=load)
    if kind == "OR" and size != "014":
        pat = pat.replace("**", "@6")
    return load_de(find(pat))

def windows(x, n_total=N_TOTAL):
    x = x[: min(n_total, len(x) // WIN * WIN)]
    return x.reshape(-1, WIN)

# ------------------------------------------------------------- özellikler
HANN = np.hanning(WIN)
FREQ = np.fft.rfftfreq(WIN, 1 / FS)[1:]          # DC atılır, 835 kutu

def spectrum(W):
    S = np.abs(np.fft.rfft(W * HANN, axis=1))[:, 1:] * 2 / WIN
    return S

def time_feats_lei(W):
    """Lei ve ark. 11 zaman domeni özelliği (Tablo 1, T1-T11)."""
    a = np.abs(W)
    T1 = W.mean(1); T2 = W.std(1, ddof=1)
    T3 = np.sqrt(a).mean(1) ** 2; T4 = np.sqrt((W ** 2).mean(1)); T5 = a.max(1)
    T6 = skew(W, axis=1); T7 = kurtosis(W, axis=1, fisher=False)
    T8 = T5 / T4; T9 = T5 / T3; T10 = T4 / a.mean(1); T11 = T5 / a.mean(1)
    return np.column_stack([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11])

def freq_feats(S, f):
    """14 frekans domeni özelliği (Tablo 2, F1-F14). S: (n, M), f: (M,)"""
    M = S.shape[1]; eps = 1e-20
    F1 = S.mean(1)
    F2 = ((S - F1[:, None]) ** 2).sum(1) / (M - 1)
    F3 = ((S - F1[:, None]) ** 3).sum(1) / (M * np.sqrt(F2) ** 3 + eps)
    F4 = ((S - F1[:, None]) ** 4).sum(1) / (M * F2 ** 2 + eps)
    ss = S.sum(1) + eps
    F5 = (f * S).sum(1) / ss
    d = f[None, :] - F5[:, None]
    F6 = np.sqrt((d ** 2 * S).sum(1) / M)
    F7 = np.sqrt((f ** 2 * S).sum(1) / ss)
    F8 = np.sqrt((f ** 4 * S).sum(1) / ((f ** 2 * S).sum(1) + eps))
    F9 = (f ** 2 * S).sum(1) / np.sqrt(ss * (f ** 4 * S).sum(1) + eps)
    F10 = F6 / (F5 + eps)
    F11 = (d ** 3 * S).sum(1) / (M * F6 ** 3 + eps)
    F12 = (d ** 4 * S).sum(1) / (M * F6 ** 4 + eps)
    F13 = (np.sqrt(np.abs(d)) * S).sum(1) / (M * np.sqrt(F6) + eps)
    F14 = np.sqrt((d ** 2 * S).sum(1) / ss)
    return np.column_stack([F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14])

def band_edges(nb):
    return np.array_split(np.arange(len(FREQ)), nb)

def fs_RAW_Lei(W, S):
    return np.hstack([time_feats_lei(W), freq_feats(S, FREQ)])

def fs_FB20_FD(W, S):
    return np.hstack([freq_feats(S[:, b], FREQ[b]) for b in band_edges(20)])

def fs_FB20_mean(W, S):
    return np.column_stack([S[:, b].mean(1) for b in band_edges(20)])

def fs_FB20_rel(W, S):
    """Önerimiz: her bandın enerjisi / toplam enerji -> kazançtan bağımsız"""
    E = np.column_stack([(S[:, b] ** 2).sum(1) for b in band_edges(20)])
    return E / E.sum(1, keepdims=True)

FEATURE_SETS = {"RAW_Lei": fs_RAW_Lei, "FB_20_FD": fs_FB20_FD,
                "FB_20_FD-mean": fs_FB20_mean, "FB_20_rel (öneri)": fs_FB20_rel}

# ---------------------------------------------------------------- model
def rf(seed):
    return RandomForestClassifier(n_estimators=500, max_depth=20,
                                  n_jobs=-1, random_state=seed)

# ---------------------------------------------------------- veri seti üret
def build(classes, load):
    """classes: [(etiket, kind, size)] -> pencere listesi + meta"""
    out = []
    for lab, kind, size in classes:
        W = windows(recording(kind, size, load))
        out.append(dict(label=lab, kind=kind, size=size, load=load, W=W, S=spectrum(W)))
    return out

def feats(recs, fs_name):
    fn = FEATURE_SETS[fs_name]
    return [fn(r["W"], r["S"]) for r in recs]

CLS10 = [("Normal", "N", None)] + [(f"{k}{s}", k, s) for k in ["IR", "OR", "B"] for s in SIZES]
KIND4 = {"N": 0, "IR": 1, "OR": 2, "B": 3}

results = {}

# ======================================================== D1 / D1b / D5
print("Veri yükleniyor (2 hp)...")
R2 = build(CLS10, 2)
print("  pencere/sınıf:", [r["W"].shape[0] for r in R2])

results["D1"] = {}; results["D1b"] = {}
for fs_name in FEATURE_SETS:
    Xs = feats(R2, fs_name)
    X = np.vstack(Xs); y = np.concatenate([[i] * len(x) for i, x in enumerate(Xs)])
    accs = []
    for seed in SEEDS:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            m = rf(seed).fit(X[tr], y[tr]); accs.append(accuracy_score(y[te], m.predict(X[te])))
    results["D1"][fs_name] = (float(np.mean(accs)), float(np.std(accs)))
    # D1b: her kaydın ilk %80'i eğitim, son %20'si test
    tr_X, tr_y, te_X, te_y = [], [], [], []
    for i, x in enumerate(Xs):
        c = int(len(x) * 0.8)
        tr_X.append(x[:c]); tr_y += [i] * c; te_X.append(x[c:]); te_y += [i] * (len(x) - c)
    a = [accuracy_score(te_y, rf(s).fit(np.vstack(tr_X), tr_y).predict(np.vstack(te_X))) for s in SEEDS]
    results["D1b"][fs_name] = (float(np.mean(a)), float(np.std(a)))
    print(f"D1  {fs_name:20s} rastgele CV: %{100*results['D1'][fs_name][0]:.2f}"
          f"   | D1b zamansal blok: %{100*results['D1b'][fs_name][0]:.2f}")

# ================================================================ D2/D3/D4
# 4 sınıf, 2 hp. Her katta bir hata boyutu (007/014/021) eğitimde HİÇ yok.
# Sağlam rulman tek kayıt -> ilk %70 eğitim, son %30 test (kaçınılmaz sınırlama).
print("\nD2/D3: hata boyutuna göre bölme (sızıntısız)")
results["D2"] = {}; results["D2_cm"] = {}
for fs_name in FEATURE_SETS:
    Xs = feats(R2, fs_name)
    per_fold = []; cm_total = np.zeros((4, 4), int)
    for held in SIZES:
        tr_X, tr_y, te_X, te_y = [], [], [], []
        for r, x in zip(R2, Xs):
            lab = KIND4[r["kind"]]
            if r["kind"] == "N":
                c = int(len(x) * 0.7)
                tr_X.append(x[:c]); tr_y += [lab] * c
                te_X.append(x[c:]); te_y += [lab] * (len(x) - c)
            elif r["size"] == held:
                te_X.append(x); te_y += [lab] * len(x)
            else:
                tr_X.append(x); tr_y += [lab] * len(x)
        for s in SEEDS:
            p = rf(s).fit(np.vstack(tr_X), tr_y).predict(np.vstack(te_X))
            per_fold.append((held, balanced_accuracy_score(te_y, p)))
            cm_total += confusion_matrix(te_y, p, labels=[0, 1, 2, 3])
    by = {h: float(np.mean([a for hh, a in per_fold if hh == h])) for h in SIZES}
    allv = [a for _, a in per_fold]
    results["D2"][fs_name] = dict(mean=float(np.mean(allv)), std=float(np.std(allv)), **by)
    results["D2_cm"][fs_name] = cm_total.tolist()
    print(f"D2  {fs_name:20s} dengeli doğruluk: %{100*np.mean(allv):.1f}  "
          + "  ".join(f"test={h}: %{100*v:.1f}" for h, v in by.items()))

# ======================================================================= D5
# 10 sınıf, 2 hp tüm veriyle eğit, farklı yükte test (aynı rulmanlar -> hâlâ sızıntılı,
# ama devir/yük değişince bant özelliklerinin dayanıklılığını gösterir)
print("\nD5: yük genellemesi (2 hp ile eğit)")
results["D5"] = {}
other = {L: build(CLS10, L) for L in [0, 1, 3]}
for fs_name in FEATURE_SETS:
    Xtr = feats(R2, fs_name)
    X = np.vstack(Xtr); y = np.concatenate([[i] * len(x) for i, x in enumerate(Xtr)])
    models = [rf(s).fit(X, y) for s in SEEDS]
    row = {}
    for L, recs in other.items():
        Xte = feats(recs, fs_name)
        Xt = np.vstack(Xte); yt = np.concatenate([[i] * len(x) for i, x in enumerate(Xte)])
        row[f"{L}hp"] = float(np.mean([accuracy_score(yt, m.predict(Xt)) for m in models]))
    results["D5"][fs_name] = row
    print(f"D5  {fs_name:20s} " + "  ".join(f"{k}: %{100*v:.1f}" for k, v in row.items()))

# rpm bilgisi
def rpm_of(L):
    m = sio.loadmat(find(FAULT_DIRS['IR'].format(s='007', l=L)))
    return float(m[[k for k in m if k.endswith('RPM')][0]].ravel()[0])
results["rpm"] = {L: rpm_of(L) for L in [0, 1, 2, 3]}
os.makedirs("sonuclar", exist_ok=True)
json.dump(results, open("sonuclar/sonuclar.json", "w"), indent=1, ensure_ascii=False)
print("\nrpm:", results["rpm"]); print("sonuclar/sonuclar.json yazıldı")
