import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
import xgboost as xgb
from feature_selector import FeatureSelector
from scipy import interp
from sklearn.base import RegressorMixin
from sklearn.calibration import CalibratedClassifierCV

# from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings
from structlog.stdlib import get_logger


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return (round(m, 3), round(m - h, 3), round(m + h, 3))


def roauc_plot(tprs, aucs, model_name, ofile):
    """ROCAUC PLOTS"""
    fig, ax = plt.subplots()
    # plotting random line
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    # plotting auc results
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_current_model = np.mean(tprs, axis=0)
    std_tpr_current_model = np.std(tprs, axis=0)
    mean_tpr_current_model[-1] = 1.0
    mean_auc_enet = auc(mean_fpr, mean_tpr_current_model)
    std_auc_enet = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr_current_model,
        color="b",
        label=f"Mean {model_name} ROC (AUC = {mean_auc_enet} $\pm$ {std_auc_enet}",
        lw=2,
        alpha=0.8,
    )
    tprs_current_model_upper = np.minimum(mean_tpr_current_model + std_tpr_current_model, 1)
    tprs_current_model_lower = np.maximum(mean_tpr_current_model - std_tpr_current_model, 0)
    ax.fill_between(
        mean_fpr,
        tprs_current_model_lower,
        tprs_current_model_upper,
        color="grey",
        alpha=0.2,
        # label=r'$\pm$ 1 std. dev.'
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Receiver operating characteristic averaged over 50 cross-validations",
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.savefig(ofile, dpi=300)


@ignore_warnings(category=ConvergenceWarning)
def model_generator(
    cf_data: pd.DataFrame,
    cf_label: pd.DataFrame,
    model: RegressorMixin,
    model_name: str,
    ofile: Path,
    inner_cv_splits: int = 50,
    outer_cv_splits: int = 20,
    test_size: float = 0.2,
    njobs: int = -1,
    random_seed: int = 2021,
) -> Dict[str, Any]:
    """Takes cystic fibrosis data, train ML models using nested cross-validation.

    Params:
        cf_data: pd.dataframe of the data (SNV IDs as row indices and samples as columns).
        cf_label: a data frame containing labels for prediction ( 0 or 1).
        model: Scikitlearn instance of a ML model.
        model_name: Model name for plotting.
        ofile: Path to output plot.
        inner_cv_splits: Number of splits for inner cross-validation.
        outer_cv_splits: Number of splits for outer cross-validation.
        test_size: percentage of test size,
        njobs: int, num. of processors to be used.
        random-seed: int, random seed for reproducibility of the results.

    Returns:
        AUROC evaluating ML model
    """
    samples = cf_data.columns.to_list()
    Y = cf_label.loc[samples, "group"]

    X = cf_data
    X = np.array(X.T)

    cv_outer = StratifiedShuffleSplit(
        n_splits=outer_cv_splits,
        test_size=test_size,
        random_state=random_seed,
    )

    def inner_cv_feature_selector(X, Y):
        fs = FeatureSelector(data=X, labels=Y)
        fs.identify_low_importance(
            task="classification",
            eval_metric="auc",
            n_iterations=inner_cv_splits,
            early_stopping=False,
            cumulative_importance=0.99,
        )
        low_importance_features = fs.ops["low_importance"]
        return low_importance_features

    tprs_lgr, aucs_lgr, f1_lgr, rec_lgr, pre_lgr, bacc_lgr, acc_lgr = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    tprs_xgb, aucs_xgb, f1_xgb, rec_xgb, pre_xgb, bacc_xgb, acc_xgb = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    tprs_rf, aucs_rf, f1_rf, rec_rf, pre_rf, bacc_rf, acc_rf = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    tprs_svc, aucs_svc, f1_svc, rec_svc, pre_svc, bacc_svc, acc_svc = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    tprs_enet, aucs_enet, f1_enet, rec_enet, pre_enet, bacc_enet, acc_enet = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    mean_fpr = np.linspace(0, 1, 100)
    for train_index, test_index in cv_outer.split(X, Y):
        x_train, x_test = X[train_index, :], X[test_index, :]
        y_train, y_test = Y[train_index], Y[test_index]
        # Identify low-importance features in train-set by inner cross-validation
        low_importance_features = inner_cv_feature_selector(x_train, x_test)
        important_features = set(cf_data.columns.to_list()) - set(low_importance_features)
        # Create new  dataset only with important features estimated from train dataset cross-validation
        x_train_important_fs, x_test_important_fs = (
            x_train.loc[:, important_features],
            x_test.loc[:, important_features],
        )

        # FIT MODELS
        # logistic regression
        lgr = LogisticRegression()
        lgr.fit(x_train_important_fs, y_train)
        y_pred_lgr = lgr.predict_proba(x_test_important_fs)[:, 1]
        fpr_lgr, tpr_lgr, thresholds_lgr = roc_curve(y_test, y_pred_lgr)
        auc_lgr = auc(fpr_lgr, tpr_lgr)
        interp_lgr_tpr = interp(mean_fpr, fpr_lgr, tpr_lgr)
        interp_lgr_tpr[0] = 0.0
        tprs_lgr.append(interp_lgr_tpr)
        aucs_lgr.append(auc_lgr)
        y_pred_lgr = lgr.predict(x_test_important_fs)
        f1_lgr.append(f1_score(y_test, y_pred_lgr, average="weighted"))
        pre_lgr.append(precision_score(y_test, y_pred_lgr, average="weighted"))
        rec_lgr.append(recall_score(y_test, y_pred_lgr, average="weighted"))
        bacc_lgr.append(balanced_accuracy_score(y_test, y_pred_lgr))
        acc_lgr.append(accuracy_score(y_test, y_pred_lgr))

        # Elastic net
        enet = SGDClassifier(loss="log", penalty="elasticnet")
        enet.fit((x_train_important_fs, y_train))
        y_pred_enet = enet.predict_proba(x_test_important_fs)[:, 1]
        fpr_enet, tpr_enet, thresholds_enet = roc_curve(y_test, y_pred_enet)
        auc_enet = auc(fpr_enet, tpr_enet)
        interp_enet_tpr = interp(mean_fpr, fpr_enet, tpr_enet)
        interp_enet_tpr[0] = 0.0
        tprs_enet.append(interp_enet_tpr)
        aucs_enet.append(auc_enet)
        y_pred_enet = enet.predict(x_test_important_fs)
        f1_enet.append(f1_score(y_test, y_pred_enet, average="weighted"))
        pre_enet.append(precision_score(y_test, y_pred_enet, average="weighted"))
        rec_enet.append(recall_score(y_test, y_pred_enet, average="weighted"))
        bacc_enet.append(balanced_accuracy_score(y_test, y_pred_enet))
        acc_enet.append(accuracy_score(y_test, y_pred_enet))

        # 2)Random Forests
        rf = RandomForestClassifier()
        rf.fit(x_train_important_fs, y_train)
        y_pred_rf = rf.predict_proba(x_test_important_fs)[:, 1]
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
        auc_rf = auc(fpr_rf, tpr_rf)
        interp_rf_tpr = interp(mean_fpr, fpr_rf, tpr_rf)
        interp_rf_tpr[0] = 0.0
        tprs_rf.append(interp_rf_tpr)
        aucs_rf.append(auc_rf)
        y_pred_rf = rf.predict(x_test_important_fs)
        f1_rf.append(f1_score(y_test, y_pred_rf, average="weighted"))
        pre_rf.append(precision_score(y_test, y_pred_rf, average="weighted"))
        rec_rf.append(recall_score(y_test, y_pred_rf, average="weighted"))
        bacc_rf.append(balanced_accuracy_score(y_test, y_pred_rf))
        acc_rf.append(accuracy_score(y_test, y_pred_rf))

        # XGBoost
        xgmod = xgb.XGBClassifier()
        xgmod.fit(x_train_important_fs, y_train)
        y_pred_xgb = xgmod.predict_proba(x_test_important_fs)[:, 1]
        fpr_xgb, tpr_xgb, thresholds_xg = roc_curve(y_test, y_pred_xgb)
        auc_xgb = auc(fpr_xgb, tpr_xgb)
        interp_xgb_tpr = interp(mean_fpr, fpr_xgb, tpr_xgb)
        interp_xgb_tpr[0] = 0.0
        tprs_xgb.append(interp_xgb_tpr)
        aucs_xgb.append(auc_xgb)
        y_pred_xgb = xgmod.predict(x_test_important_fs)
        f1_xgb.append(f1_score(y_test, y_pred_xgb, average="weighted"))
        pre_xgb.append(precision_score(y_test, y_pred_xgb, average="weighted"))
        rec_xgb.append(recall_score(y_test, y_pred_xgb, average="weighted"))
        bacc_xgb.append(balanced_accuracy_score(y_test, y_pred_xgb))
        acc_xgb.append(accuracy_score(y_test, y_pred_xgb))

        # SVC
        svc = CalibratedClassifierCV(base_estimator=LinearSVC())
        svc.fit(x_train_important_fs, y_train)
        y_pred_svc = svc.predict_proba(x_test_important_fs)[:, 1]
        fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, y_pred_svc)
        auc_svc = auc(fpr_svc, tpr_svc)
        interp_svc_tpr = interp(mean_fpr, fpr_svc, tpr_svc)
        interp_svc_tpr[0] = 0.0
        tprs_svc.append(interp_svc_tpr)
        aucs_svc.append(auc_svc)
        y_pred_svc = svc.predict(x_test_important_fs)
        f1_svc.append(f1_score(y_test, y_pred_svc, average="weighted"))
        pre_svc.append(precision_score(y_test, y_pred_svc, average="weighted"))
        rec_svc.append(recall_score(y_test, y_pred_svc, average="weighted"))
        bacc_svc.append(balanced_accuracy_score(y_test, y_pred_svc))
        acc_svc.append(accuracy_score(y_test, y_pred_svc))
    # write out performance values.
    # writing model performances across different metrics in the output
    aucs_lgr_m, f1_lgr_m, rec_lgr_m, pre_lgr_m, bacc_lgr_m, acc_lgr_m = (
        mean_confidence_interval(aucs_lgr),
        mean_confidence_interval(f1_lgr),
        mean_confidence_interval(rec_lgr),
        mean_confidence_interval(pre_lgr),
        mean_confidence_interval(bacc_lgr),
        mean_confidence_interval(acc_lgr),
    )
    aucs_xgb_m, f1_xgb_m, rec_xgb_m, pre_xgb_m, bacc_xgb_m, acc_xgb_m = (
        mean_confidence_interval(aucs_xgb),
        mean_confidence_interval(f1_xgb),
        mean_confidence_interval(rec_xgb),
        mean_confidence_interval(pre_xgb),
        mean_confidence_interval(bacc_xgb),
        mean_confidence_interval(acc_xgb),
    )
    aucs_rf_m, f1_rf_m, rec_rf_m, pre_rf_m, bacc_rf_m, acc_rf_m = (
        mean_confidence_interval(aucs_rf),
        mean_confidence_interval(f1_rf),
        mean_confidence_interval(rec_rf),
        mean_confidence_interval(pre_rf),
        mean_confidence_interval(bacc_rf),
        mean_confidence_interval(acc_rf),
    )
    aucs_svc_m, f1_svc_m, rec_svc_m, pre_svc_m, bacc_svc_m, acc_svc_m = (
        mean_confidence_interval(aucs_svc),
        mean_confidence_interval(f1_svc),
        mean_confidence_interval(rec_svc),
        mean_confidence_interval(pre_svc),
        mean_confidence_interval(bacc_svc),
        mean_confidence_interval(acc_svc),
    )
    aucs_enet_m, f1_enet_m, rec_enet_m, pre_enet_m, bacc_enet_m, acc_enet_m = (
        mean_confidence_interval(aucs_enet),
        mean_confidence_interval(f1_enet),
        mean_confidence_interval(rec_enet),
        mean_confidence_interval(pre_enet),
        mean_confidence_interval(bacc_enet),
        mean_confidence_interval(acc_enet),
    )
    # writing the evaluation in a dataframe
    df_lr = pd.DataFrame(
        {
            "AUROC": aucs_lgr_m,
            "bACC": bacc_lgr_m,
            "Accuracy": acc_lgr_m,
            "F1": f1_lgr_m,
            "Precision": pre_lgr_m,
            "Recall": rec_lgr_m,
        },
        index=pd.Series(["Logistric_regression", "lrCI-", "lrCI+"], name="Tag"),
    )
    df_xgb = pd.DataFrame(
        {
            "AUROC": aucs_xgb_m,
            "bACC": bacc_xgb_m,
            "Accuracy": acc_xgb_m,
            "F1": f1_xgb_m,
            "Precision": pre_xgb_m,
            "Recall": rec_xgb_m,
        },
        index=pd.Series(["XGBoost", "xgbCI-", "xgbCI+"], name="Tag"),
    )
    df_rf = pd.DataFrame(
        {
            "AUROC": aucs_rf_m,
            "bACC": bacc_rf_m,
            "Accuracy": acc_rf_m,
            "F1": f1_rf_m,
            "Precision": pre_rf_m,
            "Recall": rec_rf_m,
        },
        index=pd.Series(["Random_forest", "rfCI-", "rfCI+"], name="Tag"),
    )
    df_svc = pd.DataFrame(
        {
            "AUROC": aucs_svc_m,
            "bACC": bacc_svc_m,
            "Accuracy": acc_svc_m,
            "F1": f1_svc_m,
            "Precision": pre_svc_m,
            "Recall": rec_svc_m,
        },
        index=pd.Series(["SVM", "svmCI-", "svmCI+"], name="Tag"),
    )
    df_enet = pd.DataFrame(
        {
            "AUROC": aucs_enet_m,
            "bACC": bacc_enet_m,
            "Accuracy": acc_enet_m,
            "F1": f1_enet_m,
            "Precision": pre_enet_m,
            "Recall": rec_enet_m,
        },
        index=pd.Series(["SVM", "svmCI-", "svmCI+"], name="Tag"),
    )
    frames = [df_lr, df_xgb, df_rf, df_svc, df_enet]
    result = pd.concat(frames).T

    df = pd.DataFrame(
        {
            "lr_auc": aucs_lgr,
            "lr_bacc": bacc_lgr,
            "lr_acc": acc_lgr,
            "lr_f1": f1_lgr,
            "lr_pre": pre_lgr,
            "lr_rec": rec_lgr,
            "svc_auc": aucs_svc,
            "svc_bacc": bacc_svc,
            "svc_acc": acc_svc,
            "svc_f1": f1_svc,
            "svc_pre": pre_svc,
            "svc_rec": rec_svc,
            "rf_auc": aucs_rf,
            "rf_bacc": bacc_rf,
            "rf_acc": acc_rf,
            "rf_f1": f1_rf,
            "rf_pre": pre_rf,
            "rf_rec": rec_rf,
            "xgb_auc": aucs_xgb,
            "xgb_bacc": bacc_xgb,
            "xgb_acc": acc_xgb,
            "xgb_f1": f1_xgb,
            "xgb_pre": pre_xgb,
            "xgb_rec": rec_xgb,
            "enet_auc": aucs_enet,
            "enet_bacc": bacc_enet,
            "enet_acc": acc_enet,
            "enet_f1": f1_enet,
            "enet_pre": pre_enet,
            "enet_rec": rec_enet,
        }
    )
    # estimate rocauc and plot the results
    tpr_auc_model_set = {
        [tprs_lgr, aucs_lgr, "Logistic regression"],
        [tprs_enet, aucs_enet, "Elastic net"],
        [tprs_svc, aucs_svc, "SVM"],
        [tprs_rf, aucs_rf, "RandomForest"],
        [tprs_xgb, aucs_xgb, "XGBoost"],
    }
    for current_model in tpr_auc_model_set:
        roauc_plot(
            current_model[0], current_model[1], current_model[2], Path(ofile).with_suffix(f".{current_model}.tiff")
        )
