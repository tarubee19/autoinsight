"""Generate three synthetic datasets with *known, planted* drivers.

Why synthetic?  Because then we can test the pipeline objectively: "did the agents recover the
drivers we planted?" (see scripts/evaluate.py). The data is also deliberately messy (numbers
stored as text, blanks, duplicates, constants, IDs, outliers) so the cleaning agent has real work.

Run:  python data/generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "sample"

# Ground truth used by the evaluation harness: features that truly drive each target.
PLANTED_DRIVERS = {
    "telco_churn": ["Contract", "tenure", "InternetService", "TechSupport", "MonthlyCharges", "PaymentMethod"],
    "hr_attrition": ["OverTime", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany", "BusinessTravel"],
    "house_prices": ["sqft", "neighborhood", "year_built", "bathrooms", "garage"],
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def telco_churn(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[0.44, 0.34, 0.22])
    tenure = np.where(
        contract == "Month-to-month", rng.gamma(2.0, 9.0, n), rng.gamma(4.0, 11.0, n)
    ).clip(0, 72).round().astype(int)
    base = np.select([internet == "Fiber optic", internet == "DSL"], [78, 52], default=21)
    monthly = (base + rng.normal(0, 7, n) + rng.choice([0, 8, 15], n, p=[0.6, 0.25, 0.15])).clip(18, 119).round(2)
    tech = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n, p=[0.3, 0.7]))
    security = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], n, p=[0.32, 0.68]))
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21]
    )
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    logit = (
        -1.45
        + 1.55 * (contract == "Month-to-month") - 0.95 * (contract == "Two year")
        - 0.034 * tenure
        + 0.85 * (internet == "Fiber optic") - 0.35 * (internet == "No")
        + 0.011 * (monthly - 65)
        - 0.55 * (tech == "Yes") - 0.25 * (security == "Yes")
        + 0.45 * (payment == "Electronic check")
        + 0.2 * (paperless == "Yes") + 0.25 * senior
        + rng.normal(0, 0.35, n)
    )
    churn = np.where(rng.random(n) < _sigmoid(logit), "Yes", "No")
    total = (monthly * tenure + rng.normal(0, 15, n)).clip(0).round(2)
    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}" for _ in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": senior,
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "InternetService": internet,
        "OnlineSecurity": security,
        "TechSupport": tech,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total.astype(object),
        "Country": "USA",
        "Churn": churn,
    })
    # --- realistic mess -------------------------------------------------------------------------
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # blanks stored as text (as in the public Telco file)
    df["TotalCharges"] = df["TotalCharges"].astype(str)
    df.loc[rng.choice(n, int(0.02 * n), replace=False), "MonthlyCharges"] = np.nan
    df.loc[rng.choice(n, int(0.015 * n), replace=False), "PaymentMethod"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "MonthlyCharges"] = 9999.0  # entry errors / outliers
    df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)  # duplicate rows
    df.columns = [c if c != "gender" else " gender " for c in df.columns]  # stray whitespace in a header
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def hr_attrition(n: int = 1500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 60, n)
    years = (rng.gamma(2.2, 3.2, n)).clip(0, 35).round().astype(int)
    income = (2500 + 180 * years + 90 * (age - 20) + rng.normal(0, 900, n)).clip(1500, 20000).round()
    overtime = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    travel = rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n, p=[0.7, 0.18, 0.12])
    satisfaction = rng.integers(1, 5, n)
    distance = rng.integers(1, 30, n)
    logit = (
        -1.6 + 1.35 * (overtime == "Yes") - 0.00028 * (income - 6000) - 0.42 * (satisfaction - 2.5)
        - 0.07 * years + 0.55 * (travel == "Travel_Frequently") + 0.012 * (distance - 9) + rng.normal(0, 0.4, n)
    )
    attr = np.where(rng.random(n) < _sigmoid(logit), "Yes", "No")
    df = pd.DataFrame({
        "EmployeeNumber": np.arange(1, n + 1),
        "Age": age,
        "Department": rng.choice(["Sales", "Research & Development", "Human Resources"], n, p=[0.3, 0.62, 0.08]),
        "MonthlyIncome": income,
        "OverTime": overtime,
        "YearsAtCompany": years,
        "JobSatisfaction": satisfaction,
        "DistanceFromHome": distance,
        "BusinessTravel": travel,
        "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n, p=[0.32, 0.46, 0.22]),
        "Over18": "Y",
        "StandardHours": 80,
        "Attrition": attr,
    })
    df.loc[rng.choice(n, int(0.04 * n), replace=False), "MonthlyIncome"] = np.nan
    df.loc[rng.choice(n, int(0.03 * n), replace=False), "JobSatisfaction"] = np.nan
    return df


def house_prices(n: int = 2000, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sqft = rng.normal(1900, 650, n).clip(450, 6000).round()
    hood = rng.choice(["Riverside", "Old Town", "Hillcrest", "Eastgate", "Harbor"], n, p=[0.25, 0.2, 0.15, 0.25, 0.15])
    hood_effect = pd.Series({"Riverside": 40, "Old Town": 15, "Hillcrest": 85, "Eastgate": -20, "Harbor": 60})[hood].to_numpy()
    year = rng.integers(1925, 2024, n)
    baths = rng.choice([1, 1.5, 2, 2.5, 3, 3.5], n, p=[0.15, 0.1, 0.35, 0.2, 0.15, 0.05])
    garage = rng.choice([0, 1, 2, 3], n, p=[0.1, 0.3, 0.5, 0.1])
    lot = rng.gamma(2.0, 2500, n).round()
    price = (
        45 + 0.115 * sqft + hood_effect + 0.55 * (year - 1925) + 14 * baths + 11 * garage + 0.0006 * lot
        + rng.normal(0, 22, n)
    ) * 1000
    df = pd.DataFrame({
        "listing_id": [f"L{100000 + i}" for i in range(n)],
        "sqft": sqft,
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": baths,
        "year_built": year,
        "neighborhood": hood,
        "garage": garage,
        "lot_size": lot,
        "price": price.round(-2),
    })
    df.loc[rng.choice(n, int(0.06 * n), replace=False), "lot_size"] = np.nan
    df.loc[rng.choice(n, int(0.02 * n), replace=False), "year_built"] = np.nan
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "telco_churn": telco_churn(), "hr_attrition": hr_attrition(), "house_prices": house_prices()
    }.items():
        path = OUT / f"{name}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path}  {frame.shape}")


if __name__ == "__main__":
    main()
