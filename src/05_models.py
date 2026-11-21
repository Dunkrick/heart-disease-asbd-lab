#!/usr/bin/env python3
"""Fit requested KNN, K-means, and hierarchical models with numeric reports."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

def save_metrics(model, x_train, x_test, y_train, y_test, name):
    """Fit KNN, score held-out rows, and save confusion/ROC images; inputs are split arrays and model label."""
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = model.predict_proba(x_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    values = {"model": name, "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred, zero_division=0),
              "recall": recall_score(y_test, pred, zero_division=0), "f1": f1_score(y_test, pred, zero_division=0),
              "roc_auc": roc_auc_score(y_test, score), "tn": cm[0, 0], "fp": cm[0, 1], "fn": cm[1, 0], "tp": cm[1, 1]}
    return values, cm, y_test, score

def cv_mean_sd(k, x_train, y_train):
    """Return stratified five-fold KNN accuracy mean and sample SD; k/data specify fixed model and training set."""
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), x_train, y_train,
                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), scoring="accuracy")
    return float(scores.mean()), float(scores.std(ddof=1))

def readable_tree_rules(tree, feature_names, clean):
    """Render tree paths in original numeric units and category labels; tree/schema/data define branch decoding."""
    numeric = {column: clean[column].astype(float).to_numpy() for column in ("age", "thalach", "oldpeak")}
    scale = {column: (float(values.mean()), float(values.std(ddof=0))) for column, values in numeric.items()}
    readable_categories = {"X_cp__4": ("cp!=asymptomatic", "cp=asymptomatic"),
                          "X_exang__0": ("exang=yes", "exang=no"),
                          "X_sex__1": ("sex=female", "sex=male")}
    lines = []
    structure = tree.tree_
    def visit(node, conditions):
        """Walk one sklearn tree node; node indexes tree arrays and conditions is current root-to-node path."""
        if structure.children_left[node] == structure.children_right[node]:
            predicted = int(tree.classes_[int(np.argmax(structure.value[node][0]))])
            lines.append(f"IF {' AND '.join(conditions) if conditions else 'TRUE'} THEN class={predicted}")
            return
        feature = feature_names[structure.feature[node]]
        threshold = float(structure.threshold[node])
        if feature in readable_categories:
            left_label, right_label = readable_categories[feature]
            visit(structure.children_left[node], conditions + [left_label])
            visit(structure.children_right[node], conditions + [right_label])
        elif feature.startswith("X_") and feature[2:] in scale:
            column = feature[2:]
            mean_value, std_value = scale[column]
            raw_threshold = threshold * std_value + mean_value
            visit(structure.children_left[node], conditions + [f"{column} <= {raw_threshold:.2f}"])
            visit(structure.children_right[node], conditions + [f"{column} > {raw_threshold:.2f}"])
        else:
            visit(structure.children_left[node], conditions + [f"{feature} <= {threshold:.2f}"])
            visit(structure.children_right[node], conditions + [f"{feature} > {threshold:.2f}"])
    visit(0, [])
    return "\n".join(lines)

def main():
    """Run requested supervised and unsupervised analyses; fixed seed and inputs define reproducibility."""
    out = ROOT / "out"; (out / "figs").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(ROOT / "data/model_matrix_sel.csv")
    x = selected.drop(columns="y").to_numpy(float); y = selected.y.to_numpy(int)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ks = list(range(1, 26, 2)); cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), xtr, ytr, cv=cv, scoring="accuracy").mean() for k in ks]
    best_k = ks[int(np.argmax(cv_scores))]
    plt.figure(); plt.plot(ks, cv_scores, marker="o"); plt.xlabel("k"); plt.ylabel("5-fold CV accuracy"); plt.title("KNN cross-validation accuracy"); plt.tight_layout(); plt.savefig(out / "figs/knn_cv.png", dpi=300); plt.close()
    primary, cm, yt, probs = save_metrics(KNeighborsClassifier(n_neighbors=best_k), xtr, xte, ytr, yte, "selected")
    primary["best_k"] = best_k
    plt.figure(); ConfusionMatrixDisplay(cm).plot(cmap="Blues", values_format="d"); plt.title("KNN confusion matrix"); plt.tight_layout(); plt.savefig(out / "figs/knn_confusion.png", dpi=300); plt.close()
    fpr, tpr, _ = roc_curve(yt, probs); plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("KNN ROC curve"); plt.tight_layout(); plt.savefig(out / "figs/knn_roc.png", dpi=300); plt.close()
    full = pd.read_csv(ROOT / "data/model_matrix.csv"); xf = full.drop(columns="y").to_numpy(float); yf = full.y.to_numpy(int)
    ftr, fte, fytr, fyte = train_test_split(xf, yf, test_size=.2, stratify=yf, random_state=SEED)
    ablation, _, _, _ = save_metrics(KNeighborsClassifier(n_neighbors=best_k), ftr, fte, fytr, fyte, "all_features")
    slope_columns = [column for column in selected.columns if column.startswith("X_slope__")]
    selected_without_slope = selected.drop(columns=slope_columns)
    xns = selected_without_slope.drop(columns="y").to_numpy(float)
    ntr, nte, nytr, nyte = train_test_split(xns, y, test_size=.2, stratify=y, random_state=SEED)
    no_slope, _, _, _ = save_metrics(KNeighborsClassifier(n_neighbors=best_k), ntr, nte, nytr, nyte, "selected_minus_slope")
    selected_without_missing_slope = selected.drop(columns=["X_slope__-1"])
    xnm = selected_without_missing_slope.drop(columns="y").to_numpy(float)
    mtr, mte, mytr, myte = train_test_split(xnm, y, test_size=.2, stratify=y, random_state=SEED)
    no_missing_slope, _, _, _ = save_metrics(KNeighborsClassifier(n_neighbors=best_k), mtr, mte, mytr, myte, "selected_minus_slope_missing_only")
    metrics = pd.DataFrame([primary, ablation, no_slope, no_missing_slope]); metrics.to_csv(out / "tables/model_metrics.csv", index=False); print(metrics.to_string(index=False))
    cv_rows = []
    for label, train_x, train_y in (("selected", xtr, ytr), ("all_features", ftr, fytr),
                                    ("selected_minus_slope", ntr, nytr), ("selected_minus_slope_missing_only", mtr, mytr)):
        mean, sd = cv_mean_sd(best_k, train_x, train_y)
        cv_rows.append({"model": label, "k": best_k, "cv_accuracy_mean": mean, "cv_accuracy_sd": sd})
    cv_table = pd.DataFrame(cv_rows); cv_table.to_csv(out / "tables/knn_cv_chosen_k.csv", index=False); print(cv_table.to_string(index=False))
    majority_class = int(np.bincount(ytr).argmax())
    majority_accuracy = float(np.mean(yte == majority_class))
    print("majority_baseline", {"class": majority_class, "test_accuracy": majority_accuracy})
    original_for_slope = pd.read_csv(ROOT / "data/clean.csv")
    original_for_slope["slope_missing"] = original_for_slope["slope"].astype(float).eq(-1)
    original_for_slope["disease"] = original_for_slope["num_bin"].astype(int)
    slope_source = original_for_slope.groupby("source", dropna=False).slope_missing.mean().rename("slope_minus1_share").reset_index()
    slope_source.to_csv(out / "tables/slope_missing_by_source.csv", index=False); print(slope_source.to_string(index=False))
    slope_disease = original_for_slope.groupby("slope_missing").disease.agg(["size", "mean"]).reset_index()
    slope_disease = slope_disease.rename(columns={"size": "n", "mean": "disease_rate"})
    slope_disease["slope_group"] = slope_disease.slope_missing.map({True: "slope=-1", False: "slope!=-1"})
    slope_disease = slope_disease[["slope_group", "n", "disease_rate"]]
    print(slope_disease.to_string(index=False))
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(xtr, ytr)
    tree_rules = readable_tree_rules(tree, list(selected.drop(columns="y").columns), original_for_slope)
    print("EXTRA DecisionTreeClassifier(max_depth=3, random_state=42) rules:")
    print(tree_rules)
    tree_rules_path = out / "rules" / "tree_rules.txt"; tree_rules_path.parent.mkdir(parents=True, exist_ok=True)
    tree_rules_path.write_text(tree_rules + "\n", encoding="utf-8")
    tree_accuracy = accuracy_score(yte, tree.predict(xte))
    print("EXTRA tree_test_accuracy", tree_accuracy)
    fig, ax = plt.subplots(figsize=(18, 9)); plot_tree(tree, feature_names=list(selected.drop(columns="y").columns), class_names=["no", "yes"], filled=True, rounded=True, ax=ax)
    ax.set_title("Decision tree, max depth 3"); fig.tight_layout(); fig.savefig(out / "figs/tree.png", dpi=300); plt.close(fig)
    inertias=[]; silhouettes=[]
    for k in range(2, 9):
        km=KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
        inertias.append(km.inertia_); silhouettes.append(silhouette_score(x, km.labels_))
    curve=pd.DataFrame({"k":range(2,9),"inertia":inertias,"silhouette":silhouettes}); curve.to_csv(out / "tables/kmeans_selection.csv",index=False); print(curve.to_string(index=False))
    fig, ax=plt.subplots(); ax.plot(curve.k,curve.inertia,marker="o"); ax.set(xlabel="k",ylabel="Inertia",title="K-means elbow"); fig.tight_layout(); fig.savefig(out / "figs/kmeans_elbow.png",dpi=300); plt.close(fig)
    fig, ax=plt.subplots(); ax.plot(curve.k,curve.silhouette,marker="o"); ax.set(xlabel="k",ylabel="Silhouette score",title="K-means silhouette"); fig.tight_layout(); fig.savefig(out / "figs/kmeans_silhouette.png",dpi=300); plt.close(fig)
    chosen=int(curve.loc[curve.silhouette.idxmax(),"k"]); km=KMeans(n_clusters=chosen,n_init=10,random_state=SEED).fit(x)
    pc=PCA(n_components=2,random_state=SEED).fit_transform(x); fig,ax=plt.subplots(); sc=ax.scatter(pc[:,0],pc[:,1],c=km.labels_,cmap="tab10",s=18); ax.set(xlabel="PC1",ylabel="PC2",title=f"K-means PCA projection, k={chosen}"); fig.colorbar(sc,ax=ax,label="Cluster"); fig.tight_layout(); fig.savefig(out / "figs/kmeans_pca.png",dpi=300); plt.close(fig)
    original=pd.read_csv(ROOT / "data/clean.csv"); num_cols=[c for c in ["age","thalach","oldpeak"] if c in original]; cat_cols=[c for c in ["sex","cp","exang","slope"] if c in original]
    profile=original[num_cols+cat_cols].copy(); profile["cluster"]=km.labels_; profile["y"]=y
    rows=[]
    for cluster, group in profile.groupby("cluster"):
        row={"cluster":cluster,"n":len(group),"disease_share":group.y.mean()}
        row.update({f"median_{c}":group[c].median() for c in num_cols}); row.update({f"mode_{c}":group[c].mode().iloc[0] for c in cat_cols})
        counts=group.y.value_counts(); row["purity"]=counts.max()/len(group); rows.append(row)
    pd.DataFrame(rows).to_csv(out / "tables/cluster_profile.csv",index=False)
    purity=sum(r["purity"]*r["n"] for r in rows)/len(profile); ari=adjusted_rand_score(y,km.labels_)
    print("selected_k",chosen,"purity",purity,"ARI",ari); print(pd.DataFrame(rows).to_string(index=False))
    cluster_source = pd.crosstab(pd.Series(km.labels_, name="cluster"), original["source"].rename("source"))
    cluster_source.to_csv(out / "tables/kmeans_cluster_source.csv"); print(cluster_source.to_string())
    slope_cluster = pd.DataFrame({"cluster": km.labels_, "slope_missing": original["slope"].astype(float).eq(-1)})
    slope_cluster = slope_cluster.groupby("cluster").slope_missing.mean().rename("slope_minus1_share").reset_index()
    slope_cluster.to_csv(out / "tables/kmeans_slope_missing_by_cluster.csv", index=False); print(slope_cluster.to_string(index=False))
    disease_source = pd.DataFrame({"cluster": km.labels_, "source": original["source"], "disease": y})
    disease_cluster_source = disease_source.groupby(["cluster", "source"]).disease.mean().unstack("source")
    disease_cluster_source.to_csv(out / "tables/kmeans_disease_rate_cluster_source.csv"); print(disease_cluster_source.to_string())
    # Hierarchical clustering uses stratified sample to bound quadratic distance work.
    sample_idx=[]
    for cls in np.unique(y):
        idx=np.flatnonzero(y==cls); sample_idx.extend(np.random.default_rng(SEED+int(cls)).choice(idx,min(len(idx),75),replace=False))
    sample_idx=np.array(sample_idx[:150]); z=linkage(x[sample_idx],method="ward"); fig,ax=plt.subplots(figsize=(12,6)); dendrogram(z,ax=ax,no_labels=True); ax.set(title="Ward hierarchical dendrogram",xlabel="Sampled patients",ylabel="Ward distance"); fig.tight_layout(); fig.savefig(out / "figs/dendrogram.png",dpi=300); plt.close(fig)

if __name__ == "__main__": main()
