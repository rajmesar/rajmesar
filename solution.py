import pandas as pd
import numpy as np


def explanatory_analysis(charges_data_path, personal_data_path, plan_data_path):
    charges_data = pd.read_csv(charges_data_path)
    personal_data = pd.read_csv(personal_data_path)
    plan_data = pd.read_csv(plan_data_path)

    monthly = pd.to_numeric(charges_data["monthlyCharges"], errors="coerce")

    # Compute trimmed mean based on the full column length (including missing values).
    monthly_non_missing = np.sort(monthly.dropna().to_numpy())
    if monthly_non_missing.size == 0:
        monthly_charges_mean = 0
    else:
        trim_each_side = int(np.floor(len(monthly) * 0.1))
        if 2 * trim_each_side >= monthly_non_missing.size:
            trimmed = monthly_non_missing
        else:
            trimmed = monthly_non_missing[trim_each_side: monthly_non_missing.size - trim_each_side]
        monthly_charges_mean = int(np.rint(trimmed.mean()))

    charges_data["monthlyCharges"] = monthly.fillna(monthly_charges_mean)

    charges_data["tenure"] = pd.to_numeric(charges_data["tenure"], errors="coerce")
    total = pd.to_numeric(charges_data["totalCharges"], errors="coerce")
    charges_data["totalCharges"] = total.fillna(
        charges_data["monthlyCharges"] * charges_data["tenure"]
    )

    bins = [0, 24, 48, 60, np.inf]
    labels = ["group1", "group2", "group3", "group4"]
    charges_data["tenureBinned"] = pd.cut(
        charges_data["tenure"], bins=bins, labels=labels, right=True, include_lowest=False
    )

    churn_numeric = pd.to_numeric(charges_data["churn"], errors="coerce")
    if churn_numeric.notna().any():
        churn_pct = int(np.rint(churn_numeric.mean() * 100))
    else:
        churn_map = {"yes": 1, "no": 0, "true": 1, "false": 0}
        churn_series = (
            charges_data["churn"].astype(str).str.strip().str.lower().map(churn_map)
        )
        churn_pct = int(np.rint(churn_series.mean() * 100)) if churn_series.notna().any() else 0

    data_merged = charges_data.merge(personal_data, on="customerID", how="inner")
    data_merged = data_merged.merge(plan_data, on="customerID", how="left")

    age_numeric = pd.to_numeric(data_merged["age"], errors="coerce")
    pct_age_above_60 = int(np.rint((age_numeric.gt(60)).mean() * 100))

    internet_service_counts = (
        data_merged["internetService"].value_counts(dropna=False).to_dict()
        if "internetService" in data_merged.columns
        else {}
    )

    return {
        "monthly_charges_mean": monthly_charges_mean,
        "charges_data_updated": charges_data,
        "churn_pct": churn_pct,
        "data_merged": data_merged,
        "pct_age_above_60": pct_age_above_60,
        "internet_service_counts": internet_service_counts,
    }
