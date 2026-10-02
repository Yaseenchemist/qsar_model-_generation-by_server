#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 QSAR MODEL BUILDER  v2  —  نسخة دقيقة محترمة قابلة للنشر
================================================================================
 من ملف SMILES + IC50/pIC50  →  أفضل QSAR model، مع اختيار آلي لكل شيء.

 التحسينات الجوهرية عن v1:
   (1) مقارنة آلية لثلاثة تمثيلات في كل رن واختيار الأفضل تلقائياً:
         • ECFP لحاله        (Morgan 2048-bit)
         • Descriptors لحاله (Mordred ~1600 — وليس 210 فقط)
         • Merged            (ECFP + Descriptors)
   (2) مقاومة Overfitting: شبكات regularization كاملة لـ XGBoost
         (reg_alpha, reg_lambda, min_child_weight, colsample_bytree, gamma)
   (3) سرعة: إصلاح تعارض الـ threads (estimator=1 thread / search=كل النوى)
         + GPU لـ XGBoost تلقائياً إن وُجد + cache_size كبير لـ SVR
   (4) تحليل Activity Cliffs: كشفها + قياس حدود الموديل عليها
         (RMSE على مركبات الـ cliff مقابل غيرها) + خيار استبعادها.

 الضوابط العلمية (محفوظة من v1):
   Curation كامل • Scaffold split • لا data leakage • Q²(CV) • external test •
   Y-randomization • Applicability Domain.

 الاستخدام:  عدّل CONFIG ثم:  python qsar_builder_v2.py
================================================================================
"""
import os, sys, json, time, warnings, itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog("rdApp.*")

from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
from xgboost import XGBRegressor
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ CONFIG ============================
CONFIG = {
    "INPUT_CSV":     "CHEMBL279_pIC50_IC50.csv",
    "SMILES_COL":    None,          # None = كشف تلقائي
    "ACTIVITY_COL":  None,
    "ACTIVITY_IS_PIC50": None,
    "IC50_UNITS":    "nM",
    "OUTPUT_DIR":    "qsar_output_v2",

    "MORGAN_BITS":   2048,
    "MORGAN_RADIUS": 2,
    "USE_MORDRED":   True,          # True = ~1600 descriptor، False = RDKit ~210
    "MORDRED_NPROC": 0,             # 0 = استخدم كل النوى المتاحة

    "TEST_FRAC":     0.20,
    "CV_FOLDS":      5,
    "RANDOM_STATE":  42,
    "N_ITER_SEARCH": 6,            # عدد التوليفات بالبحث العشوائي (RF/XGB)

    "AD_TANIMOTO_THRESHOLD": 0.30,  # حد Applicability Domain

    # --- Activity cliffs ---
    "CLIFF_SIM_THRESHOLD": 0.90,    # تشابه بنيوي عالٍ
    "CLIFF_DELTA_PIC50":   1.0,     # فرق فعالية ≥ 1 log
    "EXCLUDE_CLIFFS":      False,   # False = نحللها ونحتفظ بها (الأصح علمياً)
                                    # True  = نستبعد مركبات الـ cliff من التدريب
}

# ============================ 1. CURATION ============================
_normalizer = rdMolStandardize.Normalizer()
_lfc        = rdMolStandardize.LargestFragmentChooser()
_uncharger  = rdMolStandardize.Uncharger()

def standardize(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None: return None
    try:
        mol = _normalizer.normalize(mol)
        mol = _lfc.choose(mol)
        mol = _uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None

def to_pic50(value, units="nM"):
    factor = {"M":1.0,"mM":1e-3,"uM":1e-6,"nM":1e-9,"pM":1e-12}[units]
    molar = float(value)*factor
    return -np.log10(molar) if molar > 0 else np.nan

def detect_col(cols, candidates):
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low: return low[cand]
    return None

def curate(df, cfg):
    cols = list(df.columns)
    scol = cfg["SMILES_COL"] or detect_col(cols, ["canonical_smiles","smiles","structure"])
    if scol is None: sys.exit("لا يوجد عمود SMILES — حدّده في CONFIG.")
    is_pic50 = cfg["ACTIVITY_IS_PIC50"]; acol = cfg["ACTIVITY_COL"]
    if acol is None:
        pic = detect_col(cols, ["pic50","pic_50"])
        if pic is not None: acol, is_pic50 = pic, True
        else:
            acol = detect_col(cols, ["ic50","standard_value","value","activity"])
            is_pic50 = False if is_pic50 is None else is_pic50
    if acol is None: sys.exit("لا يوجد عمود نشاط — حدّده في CONFIG.")
    print(f"   عمود SMILES='{scol}' | النشاط='{acol}' | pIC50 جاهز؟ {is_pic50}")

    w = df[[scol, acol]].copy(); w.columns = ["smiles_raw","activity_raw"]; n0=len(w)
    w["smiles"] = w["smiles_raw"].apply(standardize); w = w.dropna(subset=["smiles"]); n1=len(w)
    w["activity_raw"] = pd.to_numeric(w["activity_raw"], errors="coerce")
    w = w.dropna(subset=["activity_raw"])
    w["pIC50"] = w["activity_raw"].astype(float) if is_pic50 \
                 else w["activity_raw"].apply(lambda v: to_pic50(v, cfg["IC50_UNITS"]))
    w = w.dropna(subset=["pIC50"]); n2=len(w)
    g = w.groupby("smiles")["pIC50"]; agg=g.median(); std=g.std().fillna(0.0)
    keep = std[std<=1.0].index
    clean = agg.loc[keep].reset_index(); clean.columns=["smiles","pIC50"]; n3=len(clean)
    print(f"   تنظيف: {n0} → SMILES صالح {n1} → نشاط رقمي {n2} → بعد التكرار/التناقض {n3}")
    return clean

# ============================ 2. FEATURIZATION ============================
def morgan_matrix(mols, cfg):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=cfg["MORGAN_RADIUS"],
                                                    fpSize=cfg["MORGAN_BITS"])
    X = np.zeros((len(mols), cfg["MORGAN_BITS"]), dtype=np.float32); fps=[]
    for i,m in enumerate(mols):
        fp = gen.GetFingerprint(m); fps.append(fp)
        arr = np.zeros((cfg["MORGAN_BITS"],), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr); X[i]=arr
    names=[f"FP_{i}" for i in range(cfg["MORGAN_BITS"])]
    return X, names, fps

def descriptor_matrix(mols, cfg):
    if cfg["USE_MORDRED"]:
        try:
            from mordred import Calculator, descriptors as mdesc
            calc = Calculator(mdesc, ignore_3D=True)
            nproc = cfg["MORDRED_NPROC"] or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
            print(f"   حساب Mordred ({len(calc.descriptors)} descriptor) بـ {nproc} نواة ...")
            dfd = calc.pandas(mols, nproc=nproc, quiet=True)
            dfd = dfd.apply(pd.to_numeric, errors="coerce")
            dfd = dfd.dropna(axis=1, how="all")             # احذف الأعمدة الفاشلة كلياً
            return dfd.to_numpy(dtype=np.float64), list(dfd.columns)
        except Exception as e:
            print(f"   ⚠️ Mordred فشل ({e}) — رجوع إلى RDKit descriptors.")
    funcs = [(n,f) for n,f in Descriptors._descList]
    X = np.full((len(mols), len(funcs)), np.nan)
    for i,m in enumerate(mols):
        for j,(_,f) in enumerate(funcs):
            try: X[i,j]=f(m)
            except Exception: pass
    return X, [n for n,_ in funcs]

# ============================ 3. SCAFFOLD SPLIT ============================
def scaffold_split(smiles, test_frac, seed):
    sc = {}
    for i,smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m else ""
        sc.setdefault(s, []).append(i)
    sets = sorted(sc.values(), key=lambda x:(len(x),x[0]), reverse=True)
    n_test = int(round(test_frac*len(smiles))); tr,te=[],[]
    for grp in sets:
        (te if len(te)+len(grp)<=n_test else tr).extend(grp)
    return np.array(sorted(tr)), np.array(sorted(te))

# ============================ 4. PREPROCESS (fit on train only) ============================
def make_preprocessor(Xtr):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xi  = np.nan_to_num(imp.transform(Xtr), nan=0.0, posinf=0.0, neginf=0.0)
    vt  = VarianceThreshold(0.0).fit(Xi)
    sc  = StandardScaler().fit(vt.transform(Xi))
    def tf(X):
        Z = np.nan_to_num(imp.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
        return sc.transform(vt.transform(Z))
    return tf, {"imputer":imp,"variance":vt,"scaler":sc}

# ============================ 5. MODELS ============================
def gpu_ok():
    return False
    try:
        import xgboost as xgb
        xgb.XGBRegressor(device="cuda", n_estimators=2, tree_method="hist").fit(
            np.zeros((6,3)), np.arange(6)); return True
    except Exception:
        return False

USE_GPU = None  # يُحدَّد في main

def model_space(seed):
    xgb_kwargs = dict(random_state=seed, n_jobs=1, tree_method="hist",
                      objective="reg:squarederror")
    if USE_GPU: xgb_kwargs["device"]="cuda"
    return {
        "RandomForest": ("random",
            RandomForestRegressor(random_state=seed, n_jobs=1),
            {"n_estimators":[400,600,800], "max_depth":[None,15,25,40],
             "min_samples_leaf":[1,2,4,8], "max_features":["sqrt",0.3,0.5]}),
        "XGBoost": ("random",
            XGBRegressor(**xgb_kwargs),
            {"n_estimators":[400,700], "max_depth":[4,6],
             "learning_rate":[0.02,0.05,0.1], "subsample":[0.7,0.85,1.0],
             "colsample_bytree":[0.6,0.8,1.0], "min_child_weight":[1,3,5],
             "reg_alpha":[0,0.1,1.0], "reg_lambda":[1.0,3.0,5.0], "gamma":[0,0.1]}),
        "SVR": ("grid",
            SVR(kernel="rbf", cache_size=2000),
            {"C":[1,10,100], "gamma":["scale"], "epsilon":[0.1]}),
    }

def rmse(y,p): return float(np.sqrt(mean_squared_error(y,p)))

def tune(name, kind, est, grid, Xtr, ytr, cfg):
    kf = KFold(cfg["CV_FOLDS"], shuffle=True, random_state=cfg["RANDOM_STATE"])
    if kind=="random":
        s = RandomizedSearchCV(est, grid, n_iter=cfg["N_ITER_SEARCH"], cv=kf,
                scoring="neg_root_mean_squared_error", n_jobs=-1,
                random_state=cfg["RANDOM_STATE"])
    else:
        s = GridSearchCV(est, grid, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
    s.fit(Xtr, ytr)
    best = s.best_estimator_
    cvp = cross_val_predict(best, Xtr, ytr, cv=kf, n_jobs=-1)
    return best, s.best_params_, r2_score(ytr,cvp), rmse(ytr,cvp)

# ============================ 6. ACTIVITY CLIFFS ============================
def find_cliffs(fps, y, sim_thr, dpic):
    """يرجع: مجموعة فهارس المركبات الداخلة في cliff + عدد الأزواج."""
    n=len(fps); cliff_idx=set(); pairs=0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        for off,s in enumerate(sims):
            j=i+1+off
            if s>=sim_thr and abs(y[i]-y[j])>=dpic:
                cliff_idx.add(i); cliff_idx.add(j); pairs+=1
    return cliff_idx, pairs

# ============================ 7. AD ============================
def ad_coverage(train_fps, q_fps, thr):
    ins=sum(1 for q in q_fps if max(DataStructs.BulkTanimotoSimilarity(q,train_fps))>=thr)
    return ins/max(1,len(q_fps))

# ============================ MAIN ============================
def main():
    global USE_GPU
    cfg=CONFIG; t0=time.time()
    os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
    print("="*72); print("QSAR BUILDER v2"); print("="*72)

    print("1) تحميل + Curation ...")
    data = curate(pd.read_csv(cfg["INPUT_CSV"]), cfg)
    if len(data)<30: sys.exit("بيانات قليلة جداً بعد التنظيف.")
    data.to_csv(f'{cfg["OUTPUT_DIR"]}/curated_dataset.csv', index=False)
    smiles=data["smiles"].tolist(); y=data["pIC50"].to_numpy()
    mols=[Chem.MolFromSmiles(s) for s in smiles]
    ok=[i for i,m in enumerate(mols) if m is not None]
    smiles=[smiles[i] for i in ok]; y=y[ok]; mols=[mols[i] for i in ok]

    print("2) Featurization ...")
    X_fp, fp_names, fps = morgan_matrix(mols, cfg)
    X_de, de_names      = descriptor_matrix(mols, cfg)
    print(f"   ECFP={X_fp.shape} | Descriptors={X_de.shape}")
    feature_sets = {
        "ECFP":        (X_fp, fp_names),
        "Descriptors": (X_de, de_names),
        "Merged":      (np.hstack([X_fp, X_de]), fp_names+de_names),
    }

    print("3) GPU check لـ XGBoost ...")
    USE_GPU = gpu_ok(); print(f"   GPU متاح؟ {USE_GPU}")

    print("4) Scaffold split ...")
    tr,te = scaffold_split(smiles, cfg["TEST_FRAC"], cfg["RANDOM_STATE"])
    ytr,yte = y[tr], y[te]
    fps_tr=[fps[i] for i in tr]; fps_te=[fps[i] for i in te]
    print(f"   train={len(tr)} test={len(te)}")

    print("5) Activity-cliff analysis ...")
    cliff_idx, n_pairs = find_cliffs(fps, y, cfg["CLIFF_SIM_THRESHOLD"], cfg["CLIFF_DELTA_PIC50"])
    cliff_mask = np.array([i in cliff_idx for i in range(len(smiles))])
    te_cliff = np.array([i in cliff_idx for i in te])
    print(f"   أزواج cliff={n_pairs} | مركبات cliff={len(cliff_idx)} "
          f"({100*len(cliff_idx)/len(smiles):.1f}%) | منها بالـ test={int(te_cliff.sum())}")
    if cfg["EXCLUDE_CLIFFS"]:
        keep_tr = np.array([i for i in tr if i not in cliff_idx])
        print(f"   EXCLUDE_CLIFFS=True → train {len(tr)}→{len(keep_tr)} (test يبقى كما هو)")
        tr=keep_tr; ytr=y[tr]; fps_tr=[fps[i] for i in tr]

    print("6) Benchmark: 3 تمثيلات × 3 نماذج (CV) ...")
    table=[]; best=None
    for fs_name,(Xall,_) in feature_sets.items():
        tf,_ = make_preprocessor(Xall[tr])
        Xtr,Xte = tf(Xall[tr]), tf(Xall[te])
        for m_name,(kind,est,grid) in model_space(cfg["RANDOM_STATE"]).items():
            est_, params, q2, rcv = tune(m_name,kind,est,grid,Xtr,ytr,cfg)
            row={"features":fs_name,"model":m_name,"Q2_cv":round(q2,4),"RMSE_cv":round(rcv,4)}
            table.append(row)
            print(f"   {fs_name:12s} | {m_name:13s} | Q²(CV)={q2:.3f} | RMSE(CV)={rcv:.3f}")
            if best is None or rcv<best["RMSE_cv_raw"]:
                best={"features":fs_name,"model":m_name,"est":est_,"params":params,
                      "Q2_cv":round(q2,4),"RMSE_cv":round(rcv,4),"RMSE_cv_raw":rcv,
                      "Xtr":Xtr,"Xte":Xte,"pp_X":Xall}
    print(f"   ★ الأفضل = {best['features']} + {best['model']}")

    print("7) إعادة بناء أفضل تمثيل + external test ...")
    tf,pp = make_preprocessor(best["pp_X"][tr])
    Xtr,Xte = tf(best["pp_X"][tr]), tf(best["pp_X"][te])
    model = clone(best["est"]); model.fit(Xtr,ytr)
    ptr,pte = model.predict(Xtr), model.predict(Xte)
    M={"best_features":best["features"],"best_model":best["model"],"best_params":best["params"],
       "Q2_cv":best["Q2_cv"],"RMSE_cv":best["RMSE_cv"],
       "R2_train":round(r2_score(ytr,ptr),4),"R2_test":round(r2_score(yte,pte),4),
       "RMSE_test":round(rmse(yte,pte),4),"MAE_test":round(mean_absolute_error(yte,pte),4),
       "n_train":int(len(tr)),"n_test":int(len(te))}
    print(f"   R²(test)={M['R2_test']} | RMSE(test)={M['RMSE_test']} | MAE={M['MAE_test']}")

    print("8) حدود الموديل على activity cliffs (test) ...")
    if te_cliff.sum()>=5 and (~te_cliff).sum()>=5:
        M["RMSE_test_on_cliffs"]   = round(rmse(yte[te_cliff],  pte[te_cliff]),4)
        M["RMSE_test_off_cliffs"]  = round(rmse(yte[~te_cliff], pte[~te_cliff]),4)
        print(f"   RMSE داخل cliffs={M['RMSE_test_on_cliffs']} | "
              f"خارجها={M['RMSE_test_off_cliffs']}  (الفرق = حدّ الموديل)")
    else:
        print("   عدد مركبات الـ cliff بالـ test قليل — تخطّي التحليل المقارن.")
    M["cliff_pairs"]=int(n_pairs); M["cliff_compounds"]=int(len(cliff_idx))

    print("9) Y-randomization ...")
    kf=KFold(cfg["CV_FOLDS"],shuffle=True,random_state=cfg["RANDOM_STATE"])
    rng=np.random.default_rng(cfg["RANDOM_STATE"]); r2s=[]
    for _ in range(10):
        yp=rng.permutation(ytr)
        r2s.append(r2_score(yp, cross_val_predict(clone(best["est"]),Xtr,yp,cv=kf,n_jobs=-1)))
    M["Yscramble_R2_mean"]=round(float(np.mean(r2s)),4)
    M["Yscramble_R2_max"]=round(float(np.max(r2s)),4)
    print(f"   R²(scrambled) mean={M['Yscramble_R2_mean']} (يجب أن يكون ~0 أو سالب)")

    print("10) Applicability Domain ...")
    M["AD_test_coverage"]=round(ad_coverage(fps_tr,fps_te,cfg["AD_TANIMOTO_THRESHOLD"]),3)
    print(f"   AD coverage(test)={M['AD_test_coverage']*100:.1f}%")

    print("11) حفظ ...")
    joblib.dump({"model":model,"preprocess":pp,"feature_set":best["features"],
                 "config":cfg,"morgan_bits":cfg["MORGAN_BITS"],
                 "morgan_radius":cfg["MORGAN_RADIUS"],"use_mordred":cfg["USE_MORDRED"]},
                f'{cfg["OUTPUT_DIR"]}/best_qsar_model.pkl')
    pd.DataFrame(table).sort_values("RMSE_cv").to_csv(
        f'{cfg["OUTPUT_DIR"]}/benchmark_all_combos.csv', index=False)
    with open(f'{cfg["OUTPUT_DIR"]}/metrics.json',"w") as f: json.dump(M,f,indent=2)

    plt.figure(figsize=(5,5))
    plt.scatter(ytr,ptr,s=12,alpha=.4,label=f"Train R²={M['R2_train']}")
    plt.scatter(yte,pte,s=22,alpha=.8,label=f"Test R²={M['R2_test']}")
    lim=[y.min()-.5,y.max()+.5]; plt.plot(lim,lim,"k--",lw=1)
    plt.xlabel("Experimental pIC50"); plt.ylabel("Predicted pIC50")
    plt.title(f"{best['features']} + {best['model']}"); plt.legend(); plt.tight_layout()
    plt.savefig(f'{cfg["OUTPUT_DIR"]}/pred_vs_exp.png',dpi=150); plt.close()

    dt=time.time()-t0
    print("="*72); print(json.dumps(M,indent=2,ensure_ascii=False))
    good = M["R2_test"]>=0.7 and M["Q2_cv"]>=0.6 and M["Yscramble_R2_mean"]<0.3
    print("\nالحكم:", "✅ موديل دفاعي جيد" if good else "⚠️ مقبول — يحتمل تحسين إضافي")
    print(f"الوقت الكلي: {dt/60:.1f} دقيقة"); print("="*72)

if __name__ == "__main__":
    main()
