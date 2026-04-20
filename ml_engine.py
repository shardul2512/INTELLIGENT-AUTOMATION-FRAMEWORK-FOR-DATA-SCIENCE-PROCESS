"""
ML Engine: H2O AutoML · SHAP (shap library) · MLflow tracking
No normal/manual ML — purely AutoML-driven.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow
import os
import tempfile
from datetime import datetime

# Ensure Java from Homebrew is in PATH so H2O can find it
if "/opt/homebrew/opt/openjdk/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"/opt/homebrew/opt/openjdk/bin:{os.environ.get('PATH', '')}"

try:
    import h2o
    from h2o.automl import H2OAutoML
    HAS_H2O = True
except Exception:
    HAS_H2O = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ── MLflow setup ──────────────────────────────────────────────────────────────
MLFLOW_DIR = os.path.join(tempfile.gettempdir(), "ds_workbench_mlflow")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
EXPERIMENT_NAME = "DS_Workbench_AutoML"


def get_or_create_experiment():
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return exp


def _sanitize_col_names(df: pd.DataFrame):
    """H2O dislikes special characters in column names — strip them."""
    rename = {}
    for c in df.columns:
        clean = (str(c).replace(" ", "_").replace("(", "").replace(")", "")
                       .replace("/", "_").replace("\\", "_").replace(".", "_")
                       .replace("[", "").replace("]", "").replace(",", ""))
        if clean != c:
            rename[c] = clean
    return df.rename(columns=rename), rename


# ── SHAP computation ──────────────────────────────────────────────────────────

def compute_shap_from_h2o(best_model, hf_test_feats, features: list, col_rename: dict):
    """
    Priority chain:
      1. True Shapley via H2O predict_contributions() (GBM, XGBoost in H2O)
      2. shap.TreeExplainer on an sklearn-compatible surrogate (LightGBM)
         trained on the H2O model's predictions — only if `shap` is installed
      3. H2O variable importance proxy
    Returns a dict with keys: ok, shap_values, expected_value,
                               feature_names, mean_abs_shap, source
    """
    inv = {v: k for k, v in col_rename.items()}

    # ── Attempt 1: H2O native Shapley ────────────────────────────────────────
    try:
        contribs   = best_model.predict_contributions(hf_test_feats)
        contrib_df = contribs.as_data_frame()
        bias = 0.0
        if "BiasTerm" in contrib_df.columns:
            bias       = float(contrib_df["BiasTerm"].mean())
            contrib_df = contrib_df.drop(columns=["BiasTerm"])
        feat_cols = [c for c in features if c in contrib_df.columns]
        sv        = contrib_df[feat_cols].values.astype(float)
        mean_abs  = (pd.Series(np.abs(sv).mean(axis=0), index=feat_cols)
                       .rename(index=lambda x: inv.get(x, x))
                       .sort_values(ascending=False))
        orig_feat_cols = [inv.get(f, f) for f in feat_cols]
        return {
            "ok": True,
            "shap_values": sv,
            "expected_value": bias,
            "feature_names": orig_feat_cols,
            "mean_abs_shap": mean_abs,
            "source": "shapley_contributions",
        }
    except Exception:
        pass

    # ── Attempt 2: shap.TreeExplainer on LightGBM surrogate ──────────────────
    if HAS_SHAP:
        try:
            import lightgbm as lgb
            from sklearn.preprocessing import LabelEncoder

            test_pd  = hf_test_feats.as_data_frame()
            feat_pd  = test_pd[[c for c in features if c in test_pd.columns]].copy()
            preds_h2o = best_model.predict(hf_test_feats).as_data_frame()

            # Encode categoricals for LightGBM surrogate
            cat_cols  = feat_pd.select_dtypes(include="object").columns.tolist()
            encoders  = {}
            for cc in cat_cols:
                le           = LabelEncoder()
                feat_pd[cc]  = le.fit_transform(feat_pd[cc].astype(str))
                encoders[cc] = le

            # Determine surrogate target
            if "predict" in preds_h2o.columns:
                y_surr = preds_h2o["predict"].values
                surr_type = "classification"
            else:
                y_surr = preds_h2o.iloc[:, 0].values.astype(float)
                surr_type = "regression"

            if surr_type == "classification":
                try:
                    le_y = LabelEncoder()
                    y_surr = le_y.fit_transform(y_surr.astype(str))
                    surr_lgb = lgb.LGBMClassifier(n_estimators=100, random_state=42,
                                                   verbose=-1)
                except Exception:
                    surr_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42,
                                                  verbose=-1)
                    y_surr   = y_surr.astype(float)
            else:
                surr_lgb = lgb.LGBMRegressor(n_estimators=100, random_state=42,
                                              verbose=-1)

            surr_lgb.fit(feat_pd.values, y_surr)
            explainer = shap.TreeExplainer(surr_lgb)
            sv        = explainer.shap_values(feat_pd.values)

            # For multi-class LightGBM, sv is a list → use class 1
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]

            orig_feats = [inv.get(f, f) for f in feat_pd.columns.tolist()]
            mean_abs   = (pd.Series(np.abs(sv).mean(axis=0), index=orig_feats)
                            .sort_values(ascending=False))
            ev = float(explainer.expected_value
                       if not isinstance(explainer.expected_value, (list, np.ndarray))
                       else explainer.expected_value[1]
                       if len(explainer.expected_value) > 1
                       else explainer.expected_value[0])
            return {
                "ok": True,
                "shap_values": sv,
                "expected_value": ev,
                "feature_names": orig_feats,
                "mean_abs_shap": mean_abs,
                "source": "shap_surrogate",
            }
        except Exception:
            pass

    # ── Attempt 3: variable importance proxy ──────────────────────────────────
    try:
        vi = best_model.varimp(use_pandas=True)
        if vi is not None and not vi.empty:
            fi = (vi.set_index("variable")["relative_importance"]
                    .reindex([f for f in vi["variable"] if f in features])
                    .dropna()
                    .rename(index=lambda x: inv.get(x, x))
                    .sort_values(ascending=False))
            n        = min(200, hf_test_feats.nrows)
            sv_proxy = np.tile(fi.values, (n, 1)).astype(float)
            return {
                "ok": True,
                "shap_values": sv_proxy,
                "expected_value": 0.0,
                "feature_names": list(fi.index),
                "mean_abs_shap": fi,
                "source": "varimp_proxy",
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "SHAP/importance not available for this model type"}


# ── Main AutoML entry point ───────────────────────────────────────────────────

def train_h2o(df: pd.DataFrame, target: str, features: list,
              max_models: int, max_secs: int, task: str):
    """
    Run H2O AutoML with SHAP explanations and MLflow experiment tracking.
    No normal/manual ML — AutoML only.

    Returns a flat result dict with:
      ok, leaderboard, metrics, shap, run_id, model_id, model_type, task,
      features, target, feature_importance
    """
    if not HAS_H2O:
        return {"ok": False, "error": "h2o not installed. Run: pip install h2o"}

    h2o_started = False
    try:
        # ── Init H2O ──────────────────────────────────────────────────────────
        h2o.init(nthreads=-1, max_mem_size="2g", verbose=False)
        h2o_started = True

        # ── Prepare data ──────────────────────────────────────────────────────
        df_ml = df[features + [target]].dropna().copy()

        # Bool → int
        for c in df_ml.select_dtypes(include="bool").columns:
            df_ml[c] = df_ml[c].astype(int)

        # Sanitize column names
        df_ml, col_rename = _sanitize_col_names(df_ml)
        clean_features = [col_rename.get(f, f) for f in features]
        clean_target   = col_rename.get(target, target)

        # Auto-detect task
        if task == "auto":
            is_clf = (df_ml[clean_target].dtype == object) or \
                     (df_ml[clean_target].nunique() < 15)
        else:
            is_clf = (task == "classification")

        # H2O frames with 80/20 split
        hf = h2o.H2OFrame(df_ml)
        if is_clf:
            hf[clean_target] = hf[clean_target].asfactor()
        splits   = hf.split_frame(ratios=[0.8], seed=42)
        hf_train = splits[0]
        hf_test  = splits[1]

        # ── H2O AutoML ────────────────────────────────────────────────────────
        aml = H2OAutoML(
            max_models=max_models,
            max_runtime_secs=max_secs,
            seed=42,
            verbosity=None,
            exclude_algos=["DeepLearning"],
        )
        aml.train(x=clean_features, y=clean_target, training_frame=hf_train)

        lb   = aml.leaderboard.as_data_frame()
        best = aml.leader
        perf = best.model_performance(hf_test)

        result = {
            "model_id":    best.model_id,
            "model_type":  best.algo,
            "leaderboard": lb.head(10).to_dict("records"),
            "task":        "classification" if is_clf else "regression",
            "features":    features,
            "target":      target,
        }

        # ── Metrics ───────────────────────────────────────────────────────────
        if is_clf:
            try:
                result["auc"] = float(perf.auc())
            except Exception:
                pass
            try:
                result["logloss"] = float(perf.logloss())
            except Exception:
                pass
            try:
                # Try binomial accuracy (list of [threshold, accuracy])
                acc = perf.accuracy()
                if acc and isinstance(acc, list):
                    result["accuracy"] = float(max(x[1] for x in acc))
            except Exception:
                pass
            if "accuracy" not in result:
                try:
                    # Fallback for multinomial
                    result["accuracy"] = float(1 - perf.mean_per_class_error())
                except Exception:
                    pass
        else:
            try:
                result["rmse"] = float(perf.rmse())
            except Exception:
                pass
            try:
                result["mae"]  = float(perf.mae())
            except Exception:
                pass
            try:
                result["r2"]   = float(perf.r2())
            except Exception:
                pass

        # ── SHAP (must happen before H2O shutdown) ────────────────────────────
        result["shap"] = compute_shap_from_h2o(
            best, hf_test[clean_features], clean_features, col_rename
        )

        # ── Variable importance ───────────────────────────────────────────────
        try:
            inv = {v: k for k, v in col_rename.items()}
            vi  = best.varimp(use_pandas=True)
            if vi is not None and not vi.empty:
                fi = (vi.set_index("variable")["relative_importance"]
                         .rename(index=lambda x: inv.get(x, x))
                         .sort_values(ascending=False))
                result["feature_importance"] = fi.to_dict()
        except Exception:
            pass

        # ── MLflow tracking ───────────────────────────────────────────────────
        exp      = get_or_create_experiment()
        run_name = f"H2O_{best.algo}_{datetime.now().strftime('%H%M%S')}"
        with mlflow.start_run(experiment_id=exp.experiment_id,
                              run_name=run_name) as run:
            mlflow.log_param("engine",       "H2O AutoML")
            mlflow.log_param("best_model",   best.model_id)
            mlflow.log_param("best_algo",    best.algo)
            mlflow.log_param("max_models",   max_models)
            mlflow.log_param("max_secs",     max_secs)
            mlflow.log_param("target",       target)
            mlflow.log_param("task",         result["task"])
            mlflow.log_param("n_features",   len(features))
            mlflow.log_param("n_train",      hf_train.nrows)
            mlflow.log_param("n_test",       hf_test.nrows)
            shap_src = result.get("shap", {}).get("source", "none")
            mlflow.log_param("shap_source",  shap_src)

            for k in ("auc", "logloss", "accuracy", "rmse", "mae", "r2"):
                if k in result and isinstance(result[k], (int, float)):
                    mlflow.log_metric(k, result[k])

            # Log top-10 SHAP importances as MLflow metrics
            shap_data = result.get("shap", {})
            if shap_data.get("ok") and "mean_abs_shap" in shap_data:
                top10 = shap_data["mean_abs_shap"].head(10)
                for feat, val in top10.items():
                    safe_feat = feat.replace(" ", "_")[:50]
                    mlflow.log_metric(f"shap_{safe_feat}", float(val))

            result["run_id"]   = run.info.run_id
            result["run_name"] = run_name

        result["ok"] = True
        return result

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        if h2o_started:
            try:
                h2o.cluster().shutdown(prompt=False)
            except Exception:
                pass


# ── MLflow run history ────────────────────────────────────────────────────────

def get_mlflow_runs(n: int = 20):
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            return []
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=n,
        )
        return runs.to_dict("records") if not runs.empty else []
    except Exception:
        return []
