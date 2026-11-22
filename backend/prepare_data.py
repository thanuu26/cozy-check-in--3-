"""Build the training CSV for the Cozy Check-In model.

Real columns come from the Kaggle "student lifestyle / stress" CSV. The page also asks for
MEALS and MOOD (and screen time), which that dataset does not have, so those are FILLED WITH
FAKE DATA generated here (seeded, so it is repeatable). Say this in your report.

Usage
  python prepare_data.py --csv student_lifestyle.csv     # your downloaded Kaggle file
  python prepare_data.py --standin                       # synthetic stand-in, to test the pipeline

Output: student_data_filled.csv  (columns: sleep, study, screen, meals, activity, mood, wellbeing, source columns)
"""
import argparse
import csv
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# Likely column names in the Kaggle file (case-insensitive). EDIT if your CSV differs.
ALIASES = {
    "study": ["study_hours_per_day", "study_hours"],
    "sleep": ["sleep_hours_per_day", "sleep_hours"],
    "social": ["social_hours_per_day", "social_hours"],
    "extra": ["extracurricular_hours_per_day", "extracurricular_hours"],
    "physical": ["physical_activity_hours_per_day", "physical_activity_hours"],
    "stress": ["stress_level"],
}
STRESS_MAP = {"low": 0, "moderate": 1, "medium": 1, "high": 2}

def load_kaggle_csv(file_path):
    """Load the downloaded Kaggle CSV, including its quoted-row format."""

    df = pd.read_csv(
        file_path,
        sep=",",
        quoting=csv.QUOTE_NONE,
        encoding="utf-8-sig"
    )

    # Remove surrounding quotation marks from column names
    df.columns = df.columns.str.strip('"')

    # Remove surrounding quotation marks from text values
    for col in df.select_dtypes(include="str").columns:
        df[col] = df[col].str.strip('"')

    return df

def find(df, key):
    lower = {c.lower(): c for c in df.columns}
    for name in ALIASES[key]:
        if name in lower:
            return lower[name]
    raise SystemExit(f"Could not find a column for '{key}'. Columns in your file: {list(df.columns)}\n"
                     f"Add its name to ALIASES in prepare_data.py.")


def standin(n=2000):
    """Synthetic data calibrated to the summary statistics printed in the Kaggle notebook
    (student_lifestyle_dataset.csv, 2000 rows): study and sleep ~uniform 5-10 h, extracurricular
    uniform 0-4 h, social mean 2.7 h, physical activity mean 4.3 h (max 13), and stress split
    14.8% Low / 33.7% Moderate / 51.5% High, driven mostly by study hours (r about 0.74), then
    sleep (about -0.3) and physical activity (about -0.2). NOT the real rows."""
    study = RNG.uniform(5, 10, n).round(1)
    sleep = RNG.uniform(5, 10, n).round(1)
    extra = RNG.uniform(0, 4, n).round(1)
    social = RNG.normal(2.7, 1.95, n).clip(0, 6).round(1)
    physical = RNG.normal(4.3, 2.55, n).clip(0, 13).round(1)
    z = lambda x: (x - x.mean()) / x.std()
    latent = z(study) - 0.4 * z(sleep) - 0.3 * z(physical) + RNG.normal(0, 0.45, n)
    lo, hi = np.quantile(latent, [0.148, 0.485])
    stress = np.where(latent < lo, "Low", np.where(latent < hi, "Moderate", "High"))
    return pd.DataFrame({"Study_Hours_Per_Day": study, "Extracurricular_Hours_Per_Day": extra,
                         "Sleep_Hours_Per_Day": sleep, "Social_Hours_Per_Day": social,
                         "Physical_Activity_Hours_Per_Day": physical, "Stress_Level": stress})


def add_low_study(study, sleep, physical, stress, n_extra):
    """The Kaggle data has no student studying under 5 h, but the page allows 0-12 h. Append FAKE rows with
    study 0-5 h (other habits resampled from real rows). Their stress label uses the same rule as the
    real data: more study -> more stress, less sleep/activity -> more stress, same Low/Moderate/High cut-offs."""
    z = lambda x, ref: (x - ref.mean()) / ref.std()
    lat = lambda st, sl, ph: z(st, study) - 0.4 * z(sl, sleep) - 0.3 * z(ph, physical)
    lo, hi = np.quantile(lat(study, sleep, physical), [(stress == 0).mean(), (stress <= 1).mean()])
    idx = RNG.integers(0, len(study), n_extra)
    st = RNG.uniform(0, 5, n_extra).round(1)
    latent = lat(st, sleep[idx], physical[idx]) + RNG.normal(0, 0.45, n_extra)
    return st, idx, np.where(latent < lo, 0, np.where(latent < hi, 1, 2))


def build(raw, low_study_rows=800):
    c = {k: find(raw, k) for k in ALIASES}
    study = raw[c["study"]].astype(float).to_numpy()
    sleep = raw[c["sleep"]].astype(float).to_numpy()
    social = raw[c["social"]].astype(float).to_numpy()
    physical = raw[c["physical"]].astype(float).to_numpy()
    stress = raw[c["stress"]]
    if not pd.api.types.is_numeric_dtype(stress):
        stress = stress.astype(str).str.strip().str.lower().map(STRESS_MAP)
        if stress.isna().any():
            raise SystemExit("Stress_Level has values other than Low/Moderate/High. Edit STRESS_MAP.")
    stress = stress.astype(int).to_numpy()
    fake_low = np.zeros(len(raw), dtype=bool)

    if low_study_rows:                                   # FAKE rows so the model covers 0-5 h of study
        st, idx, s2 = add_low_study(study, sleep, physical, stress, low_study_rows)
        study = np.concatenate([study, st]); sleep = np.concatenate([sleep, sleep[idx]])
        social = np.concatenate([social, social[idx]]); physical = np.concatenate([physical, physical[idx]])
        stress = np.concatenate([stress, s2]); fake_low = np.concatenate([fake_low, np.ones(low_study_rows, bool)])
    n = len(study)

    # --- FAKE columns (not in the Kaggle data) ---
    mood_mean = np.array([4.0, 3.0, 2.0])[stress] + 0.08 * (physical - 4)      # calmer / more active -> better mood
    mood = np.clip(np.round(mood_mean + RNG.normal(0, 0.9, n)), 1, 5)
    meals = np.clip(np.round(3.0 - 0.15 * stress + 0.12 * (sleep - 7) + RNG.normal(0, 0.95, n)), 0, 5)
    skipped_all = RNG.random(n) < np.array([0.02, 0.04, 0.07])[stress]          # some days with NO meals (0 option)
    meals = np.where(skipped_all, 0, meals) + 0.0                               # meals run 0-5 like the page
    screen = np.clip(social + 0.3 * study + RNG.normal(1.5, 1.0, n), 0, 14)   # phone/laptop leisure + study screens

    df = pd.DataFrame({
        "sleep": sleep.clip(0, 12), "study": study.clip(0, 12), "screen": screen.round(1),
        "meals": meals, "activity": (physical * 60).clip(0, 180).round(0), "mood": mood,
    })
    # --- wellbeing target (my definition: edit it and explain it in your report) ---
    meal_curve = np.array([0, .35, .7, 1, .9, .7])[meals.astype(int)]         # same shape as the page: 3 meals is best
    study_curve = np.where(study < 2, .5 + .25 * study, 1.0)                  # page: almost no study (<2 h) is slightly low
    base = np.array([82.0, 60.0, 34.0])[stress]
    wb = (base + 2.0 * np.clip(sleep - 7, -3, 2) + 3.0 * np.minimum(physical, 1.5)
          - 1.5 * np.clip(study - 8, 0, None)
          - 10.0 * (1 - study_curve)                                          # 0 h of study costs about 5 points
          + 18.0 * (meal_curve - 0.7)                                         # skipping meals lowers the score
          + 3.0 * (mood - 3)                                                  # mood moves the score directly
          - 0.8 * np.clip(screen - 6, 0, None)                                # very long screen days lower it
          + RNG.normal(0, 3, n))
    df["wellbeing"] = wb.clip(0, 100).round(1)
    df["stress_level"] = stress
    df["fake_low_study_row"] = fake_low
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--out", default="student_data_filled.csv")
    ap.add_argument("--low-study-rows", type=int, default=800, help="fake rows with 0-5 h of study (0 to skip)")
    a = ap.parse_args()
    if not a.csv and not a.standin:
        raise SystemExit("Give --csv <file> or --standin")
    raw = standin() if a.standin else load_kaggle_csv(a.csv)
    df = build(raw, a.low_study_rows)
    df.to_csv(a.out, index=False)
    print(f"Wrote {a.out}: {len(df)} rows. Source: {'SYNTHETIC STAND-IN' if a.standin else a.csv}")
    print("Real columns: sleep, study, activity, stress -> wellbeing. FAKE: meals, mood, screen, and the extra 0-5 h study rows.")


if __name__ == "__main__":
    main()
