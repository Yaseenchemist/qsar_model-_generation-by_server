#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 QSAR VISUALIZER  —  صور وحسابات كاملة للموديل المحفوظ
================================================================================
 يقرأ مخرجات qsar_builder_v2.py ويعيد اشتقاق التنبؤات بدقة (بدون إعادة تدريب)،
 ثم يولّد كل رسومات النشر:

   1. Predicted vs Experimental  (train + test)
   2. Residuals vs Predicted + هيستوغرام البواقي
   3. Applicability Domain plot  (تشابه لأقرب تدريب مقابل |standardized residual|)
   4. Benchmark: مقارنة الـ 9 توليفات  (Q² و RMSE)
   5. Activity cliffs: RMSE داخل/خارج الـ cliffs + scatter ملوّن
   6. Feature importance: أهم 20
   7. Y-scramble: الموديل الحقيقي مقابل العشوائي
   8. لوحة ملخص المقاييس

 يحفظ أيضاً predictions.csv (SMILES + experimental + predicted + split + cliff + AD).

 التشغيل (داخل نفس مجلد qsar_builder_v2.py و qsar_output_v2/):
     python qsar_visualize.py
================================================================================
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from rdkit import Chem, RDLogger
from rdkit.Chem import DataStructs
RDLogger.DisableLog("rdApp.*")

# نستورد نفس دوال البناء لضمان تطابق الـ features والـ split 100%
import qsar_builder_v2 as qb

OUT = "qsar_output_v2"
FIGDIR = os.path.join(OUT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---- ستايل موحّد ----
plt.rcParams.update({"figure.dpi":120, "savefig.dpi":170, "font.size":11,
                     "axes.grid":True, "grid.alpha":0.25, "axes.axisbelow":True})
C_TR, C_TE, C_CL = "#4C72B0", "#DD8452", "#C44E52"

def apply_pp(pp, X):
    """يعيد نفس تحويل make_preprocessor من الكائنات المحفوظة."""
    Z = np.nan_to_num(pp["imputer"].transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    return pp["scaler"].transform(pp["variance"].transform(Z))

def main():
    print("تحميل الموديل والبيانات ...")
    bundle = joblib.load(f"{OUT}/best_qsar_model.pkl")
    model, pp = bundle["model"], bundle["preprocess"]
    fs = bundle["feature_set"]
    cfg = bundle["config"]
    metrics = json.load(open(f"{OUT}/metrics.json"))
    bench = pd.read_csv(f"{OUT}/benchmark_all_combos.csv")
    data = pd.read_csv(f"{OUT}/curated_dataset.csv")

    smiles = data["smiles"].tolist(); y = data["pIC50"].to_numpy()
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ok = [i for i,m in enumerate(mols) if m is not None]
    smiles=[smiles[i] for i in ok]; y=y[ok]; mols=[mols[i] for i in ok]

    print(f"إعادة الـ featurization ({fs}) والـ split ...")
    X_fp, _, fps = qb.morgan_matrix(mols, cfg)
    if fs == "ECFP":
        Xall = X_fp
    else:
        X_de,_ = qb.descriptor_matrix(mols, cfg)
        Xall = X_de if fs=="Descriptors" else np.hstack([X_fp, X_de])

    tr, te = qb.scaffold_split(smiles, cfg["TEST_FRAC"], cfg["RANDOM_STATE"])
    Xtr, Xte = apply_pp(pp, Xall[tr]), apply_pp(pp, Xall[te])
    ytr, yte = y[tr], y[te]
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    fps_tr = [fps[i] for i in tr]; fps_te = [fps[i] for i in te]

    # cliffs + AD
    print("حساب cliffs و AD ...")
    cliff_idx,_ = qb.find_cliffs(fps, y, cfg["CLIFF_SIM_THRESHOLD"], cfg["CLIFF_DELTA_PIC50"])
    te_cliff = np.array([te[i] in cliff_idx for i in range(len(te))])
    max_sim_te = np.array([max(DataStructs.BulkTanimotoSimilarity(q, fps_tr)) for q in fps_te])
    res_te = yte - pte
    std_res = res_te / (res_te.std() + 1e-9)

    # حفظ جدول التنبؤات
    pd.DataFrame({"SMILES":[smiles[i] for i in te], "experimental_pIC50":yte,
                  "predicted_pIC50":pte, "residual":res_te,
                  "is_cliff":te_cliff, "max_tanimoto_to_train":max_sim_te
                 }).to_csv(f"{OUT}/predictions_test.csv", index=False)

    from sklearn.metrics import r2_score, mean_squared_error
    rmse=lambda a,b: float(np.sqrt(mean_squared_error(a,b)))
    r2_te, rmse_te = r2_score(yte,pte), rmse(yte,pte)
    r2_tr = r2_score(ytr,ptr)

    # ============ 1. Predicted vs Experimental ============
    print("رسم 1: predicted vs experimental")
    fig,ax=plt.subplots(figsize=(6,6))
    ax.scatter(ytr,ptr,s=10,alpha=.35,c=C_TR,label=f"Train (R²={r2_tr:.3f}, n={len(tr)})")
    ax.scatter(yte,pte,s=26,alpha=.8,c=C_TE,edgecolor="white",linewidth=.3,
               label=f"Test (R²={r2_te:.3f}, n={len(te)})")
    lim=[y.min()-.5,y.max()+.5]; ax.plot(lim,lim,"k--",lw=1.2,label="ideal")
    ax.plot(lim,[v+1 for v in lim],":",c="gray",lw=.8)
    ax.plot(lim,[v-1 for v in lim],":",c="gray",lw=.8,label="±1 log")
    ax.set(xlabel="Experimental pIC50",ylabel="Predicted pIC50",xlim=lim,ylim=lim,
           title=f"{fs} + {metrics['best_model']}  |  RMSE(test)={rmse_te:.3f}")
    ax.legend(loc="upper left",fontsize=9); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/1_pred_vs_exp.png"); plt.close()

    # ============ 2. Residuals ============
    print("رسم 2: residuals")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.5))
    a1.scatter(pte,res_te,s=22,alpha=.7,c=C_TE); a1.axhline(0,c="k",ls="--",lw=1)
    a1.axhline( res_te.std()*2,c="gray",ls=":",lw=.8)
    a1.axhline(-res_te.std()*2,c="gray",ls=":",lw=.8,label="±2σ")
    a1.set(xlabel="Predicted pIC50",ylabel="Residual",title="Residuals vs Predicted (test)")
    a1.legend(fontsize=9)
    a2.hist(res_te,bins=40,color=C_TE,alpha=.85,edgecolor="white")
    a2.axvline(0,c="k",ls="--",lw=1)
    a2.set(xlabel="Residual",ylabel="Count",
           title=f"Residual distribution (μ={res_te.mean():.2f}, σ={res_te.std():.2f})")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/2_residuals.png"); plt.close()

    # ============ 3. Applicability Domain ============
    print("رسم 3: applicability domain")
    thr=cfg["AD_TANIMOTO_THRESHOLD"]
    inside=max_sim_te>=thr
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(max_sim_te[inside],np.abs(std_res)[inside],s=24,alpha=.7,c=C_TR,label="Inside AD")
    ax.scatter(max_sim_te[~inside],np.abs(std_res)[~inside],s=34,alpha=.85,c=C_CL,
               marker="^",label="Outside AD")
    ax.axvline(thr,c="k",ls="--",lw=1,label=f"AD threshold={thr}")
    ax.axhline(3,c="gray",ls=":",lw=.9,label="|std residual|=3")
    ax.set(xlabel="Max Tanimoto similarity to nearest training compound",
           ylabel="|Standardized residual|",
           title=f"Applicability Domain  |  coverage={100*inside.mean():.1f}%")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/3_applicability_domain.png"); plt.close()

    # ============ 4. Benchmark 9 combos ============
    print("رسم 4: benchmark")
    b=bench.copy(); b["combo"]=b["features"]+"\n"+b["model"]
    b=b.sort_values("Q2_cv",ascending=True)
    fig,(a1,a2)=plt.subplots(1,2,figsize=(12,5))
    cols=[C_TE if (r.features==metrics["best_features"] and r.model==metrics["best_model"])
          else C_TR for r in b.itertuples()]
    a1.barh(b["combo"],b["Q2_cv"],color=cols); a1.axvline(0.7,c="k",ls="--",lw=.9)
    a1.set(xlabel="Q²(CV)",title="Q2 (CV) — all combinations (orange = winner)")
    a2.barh(b["combo"],b["RMSE_cv"],color=cols)
    a2.set(xlabel="RMSE(CV)",title="RMSE (CV) — lower = better")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/4_benchmark_combos.png"); plt.close()

    # ============ 5. Activity cliffs ============
    print("رسم 5: activity cliffs")
    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.8))
    if te_cliff.sum()>=3 and (~te_cliff).sum()>=3:
        r_on=rmse(yte[te_cliff],pte[te_cliff]); r_off=rmse(yte[~te_cliff],pte[~te_cliff])
        a1.bar(["Off cliffs","On cliffs"],[r_off,r_on],color=[C_TR,C_CL])
        for i,v in enumerate([r_off,r_on]): a1.text(i,v+.01,f"{v:.3f}",ha="center")
        a1.set(ylabel="RMSE",title=f"Model limit on activity cliffs\n(n_cliff_test={int(te_cliff.sum())})")
    a2.scatter(yte[~te_cliff],pte[~te_cliff],s=18,alpha=.5,c=C_TR,label="non-cliff")
    a2.scatter(yte[te_cliff],pte[te_cliff],s=42,alpha=.9,c=C_CL,marker="D",
               edgecolor="k",linewidth=.4,label="cliff")
    lim=[y.min()-.5,y.max()+.5]; a2.plot(lim,lim,"k--",lw=1)
    a2.set(xlabel="Experimental",ylabel="Predicted",title="Test set colored by cliff",xlim=lim,ylim=lim)
    a2.legend(fontsize=9); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/5_activity_cliffs.png"); plt.close()

    # ============ 6. Feature importance ============
    print("رسم 6: feature importance")
    if hasattr(model,"feature_importances_"):
        imp=model.feature_importances_
        support=pp["variance"].get_support()
        # أسماء الأعمدة الأصلية بحسب التمثيل
        if fs=="ECFP": names=np.array([f"FP_{i}" for i in range(X_fp.shape[1])])
        elif fs=="Descriptors":
            X_de,dn=qb.descriptor_matrix(mols,cfg); names=np.array(dn)
        else:
            X_de,dn=qb.descriptor_matrix(mols,cfg)
            names=np.array([f"FP_{i}" for i in range(X_fp.shape[1])]+list(dn))
        kept=names[support]
        order=np.argsort(imp)[::-1][:20]
        fig,ax=plt.subplots(figsize=(8,6))
        ax.barh(range(20)[::-1],imp[order],color=C_TR)
        ax.set_yticks(range(20)[::-1]); ax.set_yticklabels(kept[order],fontsize=8)
        ax.set(xlabel="Importance",title=f"Top 20 features ({fs} + {metrics['best_model']})")
        fig.tight_layout(); fig.savefig(f"{FIGDIR}/6_feature_importance.png"); plt.close()
    else:
        print("   (الموديل لا يوفّر feature_importances_ — تخطّي)")

    # ============ 7. Y-scramble ============
    print("رسم 7: Y-scramble")
    fig,ax=plt.subplots(figsize=(6.5,4.8))
    ys_mean=metrics.get("Yscramble_R2_mean",0); ys_max=metrics.get("Yscramble_R2_max",0)
    real=metrics["Q2_cv"]
    ax.bar(["Real model\n(Q2 CV)","Scrambled\n(mean)","Scrambled\n(max)"],
           [real,ys_mean,ys_max],color=[C_TR,C_CL,"#8C8C8C"])
    for i,v in enumerate([real,ys_mean,ys_max]): ax.text(i,v+(.01 if v>=0 else -.03),f"{v:.3f}",ha="center")
    ax.axhline(0,c="k",lw=.8)
    ax.set(ylabel="R²",title="Y-randomization: real >> scrambled (not chance)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/7_y_scramble.png"); plt.close()

    # ============ 8. لوحة الملخص ============
    print("رسم 8: summary panel")
    fig,ax=plt.subplots(figsize=(7,5)); ax.axis("off")
    rows=[("Best features",metrics["best_features"]),("Best model",metrics["best_model"]),
          ("Q²(CV)",metrics["Q2_cv"]),("R²(train)",metrics["R2_train"]),
          ("R²(test)",metrics["R2_test"]),("RMSE(test)",metrics["RMSE_test"]),
          ("MAE(test)",metrics["MAE_test"]),
          ("RMSE on cliffs",metrics.get("RMSE_test_on_cliffs","-")),
          ("RMSE off cliffs",metrics.get("RMSE_test_off_cliffs","-")),
          ("Y-scramble R²(mean)",metrics["Yscramble_R2_mean"]),
          ("AD coverage",metrics["AD_test_coverage"]),
          ("n_train / n_test",f'{metrics["n_train"]} / {metrics["n_test"]}')]
    tbl=ax.table(cellText=[[k,str(v)] for k,v in rows],
                 colLabels=["Metric","Value"],cellLoc="left",loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1,1.6)
    for j in range(2): tbl[0,j].set_facecolor(C_TR); tbl[0,j].set_text_props(color="white",weight="bold")
    ax.set_title("Model performance summary",pad=18,weight="bold")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/8_summary.png"); plt.close()

    print("\n✅ خلص. الصور في:", FIGDIR)
    for f in sorted(os.listdir(FIGDIR)): print("   -", f)
    print("جدول التنبؤات:", f"{OUT}/predictions_test.csv")

if __name__=="__main__":
    main()
