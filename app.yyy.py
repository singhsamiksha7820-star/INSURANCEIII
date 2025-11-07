import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


st.set_page_config(
    page_title="Insurance Policy Status Dashboard",
    layout="wide"
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    # Drop identifiers to avoid overfitting on IDs / names
    work_df = df.drop(columns=["POLICY_NO", "PI_NAME"])

    target_col = "POLICY_STATUS"
    y = work_df[target_col]
    X = work_df.drop(columns=[target_col])

    class_names = ["Approved Death Claim", "Repudiate Death"]
    y_bin = label_binarize(y, classes=class_names).ravel()

    # Train-test split
    X_train, X_test, y_train, y_test, yb_train, yb_test = train_test_split(
        X,
        y,
        y_bin,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    numeric_features = ["SUM_ASSURED", "PI_AGE", "PI_ANNUAL_INCOME"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = "passthrough"
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosted Trees": GradientBoostingClassifier(random_state=42),
    }

    results = []
    roc_curves = {}
    cm_train_dict = {}
    cm_test_dict = {}
    fitted_models = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")

        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        cm_train = confusion_matrix(y_train, y_train_pred, labels=class_names)
        cm_test = confusion_matrix(y_test, y_test_pred, labels=class_names)
        cm_train_dict[name] = cm_train
        cm_test_dict[name] = cm_test

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average="weighted", zero_division=0
        )

        y_test_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(yb_test, y_test_proba)
        auc = roc_auc_score(yb_test, y_test_proba)
        roc_curves[name] = (fpr, tpr, auc)

        results.append(
            {
                "Algorithm": name,
                "CV Mean Accuracy (Train, cv=5)": cv_scores.mean(),
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,
                "Precision (weighted)": precision,
                "Recall (weighted)": recall,
                "F1-score (weighted)": f1,
                "AUC (Repudiate Death=1)": auc,
            }
        )

    results_df = pd.DataFrame(results)

    # Prepare feature importance data for top features
    feature_importance_info = {}
    for name, pipe in fitted_models.items():
        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]

        num_features = numeric_features
        cat_ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = list(cat_ohe.get_feature_names_out(categorical_features))
        all_feature_names = num_features + cat_feature_names

        importances = model.feature_importances_
        fi_df = (
            pd.DataFrame({"feature": all_feature_names, "importance": importances})
            .sort_values(by="importance", ascending=False)
        )
        feature_importance_info[name] = fi_df

    # Return everything needed for dashboard
    return {
        "models": fitted_models,
        "results_df": results_df,
        "roc_curves": roc_curves,
        "cm_train": cm_train_dict,
        "cm_test": cm_test_dict,
        "class_names": class_names,
        "feature_importances": feature_importance_info,
        "feature_columns": X.columns.tolist(),
    }


def plot_confusion_matrix(cm, classes, title: str):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def main():
    st.title("📊 Insurance Policy Status Analytics & Prediction")

    st.sidebar.header("Data & Filters")

    data_path = "Insurance_excel.xlsx"
    try:
        df = load_data(data_path)
    except Exception as e:
        st.error(f"Could not load data from {data_path}. Please make sure the file is in the app folder. Error: {e}")
        st.stop()

    st.sidebar.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    tabs = st.tabs(
        [
            "📈 Insights Dashboard",
            "🤖 Model Performance",
            "📂 Predict New Data",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1: INSIGHTS DASHBOARD
    # ------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Portfolio Overview & Risk Insights")

        # Filters (treat PI_OCCUPATION as job role, and PI_AGE as slider)
        df_filt = df.copy()

        job_roles = sorted(df_filt["PI_OCCUPATION"].dropna().unique())
        selected_roles = st.multiselect(
            "Filter by Occupation (Job Role)",
            job_roles,
            default=job_roles,
        )

        if selected_roles:
            df_filt = df_filt[df_filt["PI_OCCUPATION"].isin(selected_roles)]

        age_min = int(df["PI_AGE"].min())
        age_max = int(df["PI_AGE"].max())
        age_range = st.slider(
            "Filter by Policy Holder Age",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
        )
        df_filt = df_filt[
            (df_filt["PI_AGE"] >= age_range[0]) & (df_filt["PI_AGE"] <= age_range[1])
        ]

        st.markdown(
            f"**Filtered rows:** {df_filt.shape[0]} (out of {df.shape[0]} total policies)"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sum Assured (Filtered)", f"{df_filt['SUM_ASSURED'].sum():,}")
        with col2:
            approval_rate = (
                (df_filt["POLICY_STATUS"] == "Approved Death Claim").mean() * 100
            )
            st.metric("Approval Rate (Filtered)", f"{approval_rate:0.1f}%")
        with col3:
            repudiated_rate = (
                (df_filt["POLICY_STATUS"] == "Repudiate Death").mean() * 100
            )
            st.metric("Repudiated Rate (Filtered)", f"{repudiated_rate:0.1f}%")

        st.markdown("### 1. Zone vs Policy Status (Stacked Bar)")
        zone_status = (
            df_filt.groupby(["ZONE", "POLICY_STATUS"])
            .size()
            .reset_index(name="count")
        )
        chart1 = (
            alt.Chart(zone_status)
            .mark_bar()
            .encode(
                x=alt.X("ZONE:N", title="Zone"),
                y=alt.Y("count:Q", title="Number of Policies"),
                color=alt.Color("POLICY_STATUS:N", title="Policy Status"),
                tooltip=["ZONE", "POLICY_STATUS", "count"],
            )
        )
        st.altair_chart(chart1, use_container_width=True)

        st.markdown("### 2. Approval Rate by Payment Mode")
        pay_mode = (
            df_filt.groupby(["PAYMENT_MODE", "POLICY_STATUS"])
            .size()
            .unstack(fill_value=0)
        )
        pay_mode["Total"] = pay_mode.sum(axis=1)
        pay_mode["Approval_Rate_%"] = (
            pay_mode.get("Approved Death Claim", 0) / pay_mode["Total"]
        ) * 100
        pay_mode_reset = pay_mode.reset_index()
        chart2 = (
            alt.Chart(pay_mode_reset)
            .mark_bar()
            .encode(
                x=alt.X("PAYMENT_MODE:N", title="Payment Mode"),
                y=alt.Y("Approval_Rate_%:Q", title="Approval Rate (%)"),
                tooltip=["PAYMENT_MODE", "Approval_Rate_%"],
            )
        )
        st.altair_chart(chart2, use_container_width=True)

        st.markdown("### 3. Age vs Sum Assured (Scatter)")
        chart3 = (
            alt.Chart(df_filt)
            .mark_circle(opacity=0.6)
            .encode(
                x=alt.X("PI_AGE:Q", title="Policy Holder Age"),
                y=alt.Y("SUM_ASSURED:Q", title="Sum Assured"),
                color=alt.Color("POLICY_STATUS:N", title="Policy Status"),
                size=alt.Size("PI_ANNUAL_INCOME:Q", title="Annual Income", legend=None),
                tooltip=[
                    "PI_AGE",
                    "SUM_ASSURED",
                    "PI_ANNUAL_INCOME",
                    "POLICY_STATUS",
                    "PI_OCCUPATION",
                ],
            )
        )
        st.altair_chart(chart3, use_container_width=True)

        st.markdown("### 4. Claim Reason vs Policy Status (Heatmap)")
        top_reasons = (
            df_filt["REASON_FOR_CLAIM"]
            .value_counts()
            .head(10)
            .index
        )
        df_reason = df_filt[df_filt["REASON_FOR_CLAIM"].isin(top_reasons)]
        reason_status = (
            df_reason.groupby(["REASON_FOR_CLAIM", "POLICY_STATUS"])
            .size()
            .reset_index(name="count")
        )
        chart4 = (
            alt.Chart(reason_status)
            .mark_rect()
            .encode(
                x=alt.X("REASON_FOR_CLAIM:N", title="Reason for Claim"),
                y=alt.Y("POLICY_STATUS:N", title="Policy Status"),
                color=alt.Color("count:Q", title="Number of Policies"),
                tooltip=["REASON_FOR_CLAIM", "POLICY_STATUS", "count"],
            )
        )
        st.altair_chart(chart4, use_container_width=True)

        st.markdown("### 5. Sum Assured Distribution by Policy Status (Box Plot)")
        chart5 = (
            alt.Chart(df_filt)
            .mark_boxplot()
            .encode(
                x=alt.X("POLICY_STATUS:N", title="Policy Status"),
                y=alt.Y("SUM_ASSURED:Q", title="Sum Assured"),
                color=alt.Color("POLICY_STATUS:N", legend=None),
            )
        )
        st.altair_chart(chart5, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 2: MODEL PERFORMANCE
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Model Comparison: DT vs RF vs GBRT")

        if st.button("Run all 3 models"):
            train_output = train_models(df)
            results_df = train_output["results_df"]
            roc_curves = train_output["roc_curves"]
            cm_train_dict = train_output["cm_train"]
            cm_test_dict = train_output["cm_test"]
            class_names = train_output["class_names"]
            feature_importances = train_output["feature_importances"]

            st.markdown("#### Performance Metrics")
            st.dataframe(results_df.round(4))

            st.markdown("#### ROC Curves")
            fig, ax = plt.subplots()
            colors = ["tab:blue", "tab:orange", "tab:green"]
            for (name, (fpr, tpr, auc)), color in zip(roc_curves.items(), colors):
                ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)
            ax.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves - Policy Status (Repudiate Death = Positive)")
            ax.legend()
            st.pyplot(fig)

            st.markdown("#### Confusion Matrices")
            model_choice = st.selectbox(
                "Select algorithm",
                list(cm_train_dict.keys()),
            )
            cm_type = st.radio("View", ["Training", "Test"], horizontal=True)

            if cm_type == "Training":
                cm_sel = cm_train_dict[model_choice]
                fig_cm = plot_confusion_matrix(
                    cm_sel, class_names, f"{model_choice} - Training Confusion Matrix"
                )
            else:
                cm_sel = cm_test_dict[model_choice]
                fig_cm = plot_confusion_matrix(
                    cm_sel, class_names, f"{model_choice} - Test Confusion Matrix"
                )
            st.pyplot(fig_cm)

            st.markdown("#### Top 15 Feature Importances (Selected Model)")
            fi_model = st.selectbox(
                "Select model for feature importance",
                list(feature_importances.keys()),
                index=1,
            )
            fi_df = feature_importances[fi_model].head(15).iloc[::-1]
            fig_fi, ax_fi = plt.subplots()
            ax_fi.barh(fi_df["feature"], fi_df["importance"])
            ax_fi.set_xlabel("Importance")
            ax_fi.set_title(f"Top 15 Features - {fi_model}")
            plt.tight_layout()
            st.pyplot(fig_fi)
        else:
            st.info("Click the button above to train all three models and view metrics.")

    # ------------------------------------------------------------------
    # TAB 3: PREDICT NEW DATA
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Upload New Dataset & Predict Policy Status")

        st.write(
            "Upload a new insurance dataset (CSV or Excel) with the same structure as the original data."
        )

        uploaded_file = st.file_uploader(
            "Upload file", type=["csv", "xlsx"], key="upload_predict"
        )

        if uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".csv"):
                new_df = pd.read_csv(uploaded_file)
            else:
                new_df = pd.read_excel(uploaded_file)

            st.write("Preview of uploaded data:")
            st.dataframe(new_df.head())

            # Train / load models (cached)
            train_output = train_models(df)
            models_dict = train_output["models"]
            feature_columns = train_output["feature_columns"]

            missing_cols = set(feature_columns) - set(new_df.columns)
            if missing_cols:
                st.error(
                    f"The uploaded file is missing the following required columns: {sorted(missing_cols)}"
                )
            else:
                # Prepare features in the same way as training
                X_new = new_df[feature_columns].copy()

                # Use Random Forest as the champion model
                rf_model = models_dict["Random Forest"]

                preds = rf_model.predict(X_new)
                proba = rf_model.predict_proba(X_new)[:, 1]

                result_df = new_df.copy()
                result_df["PREDICTED_POLICY_STATUS"] = preds
                result_df["PROBA_REPUDIATE"] = proba

                st.markdown("#### Predictions")
                st.dataframe(result_df.head())

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions as CSV",
                    data=csv_bytes,
                    file_name="insurance_predictions.csv",
                    mime="text/csv",
                )

        else:
            st.info("Upload a new dataset to generate predictions.")


if __name__ == "__main__":
    main()
