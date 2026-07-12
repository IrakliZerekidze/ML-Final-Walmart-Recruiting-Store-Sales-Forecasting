from pathlib import Path
import numpy as np
import pandas as pd

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

SUPER_BOWL = pd.to_datetime(["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"])
LABOR_DAY = pd.to_datetime(["2010-09-10", "2011-09-09", "2012-09-07", "2013-09-06"])
THANKSGIVING = pd.to_datetime(["2010-11-26", "2011-11-25", "2012-11-23", "2013-11-29"])
CHRISTMAS = pd.to_datetime(["2010-12-31", "2011-12-30", "2012-12-28", "2013-12-27"])


# Metric

def weighted_mae(y_true, y_pred, is_holiday):
    """Competition metric: holiday weeks weighted 5x."""
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


# Pipeline steps

def load_raw(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv.zip")
    test = pd.read_csv(data_dir / "test.csv.zip")
    features = pd.read_csv(data_dir / "features.csv.zip")
    stores = pd.read_csv(data_dir / "stores.csv")
    for df in (train, test, features):
        df["Date"] = pd.to_datetime(df["Date"])
    return train, test, features, stores


def merge_sources(train, test, features, stores):
    train_full = (
        train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
             .merge(stores, on="Store", how="left")
    )
    test_full = (
        test.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
            .merge(stores, on="Store", how="left")
    )
    return (
        train_full.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True),
        test_full.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True),
    )


def fill_grid(train_full, features, stores, freq="W-FRI", verbose=True):
    """Ensure every (Store, Dept) has a row for every week in the train range."""
    all_dates = pd.date_range(train_full["Date"].min(), train_full["Date"].max(), freq=freq)
    pairs = train_full[["Store", "Dept"]].drop_duplicates()
    full_index = pairs.merge(pd.DataFrame({"Date": all_dates}), how="cross")

    if verbose:
        gap = len(full_index) - len(train_full)
        print(f"Expected rows if no gaps: {len(full_index)}")
        print(f"Actual rows: {len(train_full)}")
        print(f"Missing (Store,Dept,Date) combos filled in: {gap}")

    keep_cols = ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
    train_full = full_index.merge(train_full[keep_cols], on=["Store", "Dept", "Date"], how="left")
    train_full["was_grid_filled"] = train_full["Weekly_Sales"].isna().astype(int)
    
    holiday_lookup = features[["Date", "IsHoliday"]].drop_duplicates()
    train_full = train_full.drop(columns=["IsHoliday"]).merge(holiday_lookup, on="Date", how="left")

    features_no_holiday = features.drop(columns=["IsHoliday"])
    train_full = train_full.merge(features_no_holiday, on=["Store", "Date"], how="left")
    train_full = train_full.merge(stores, on="Store", how="left")
    return train_full


def add_markdown_flags(df):
    df = df.copy()
    for col in MARKDOWN_COLS:
        df[col + "_exists"] = df[col].notna().astype(int)
        df[col] = df[col].fillna(0)
    return df


def flag_negative_sales(df):
    df = df.copy()
    if "Weekly_Sales" in df.columns:
        df["is_negative_sales"] = (df["Weekly_Sales"] < 0).astype(int)
        df["Weekly_Sales_clipped"] = df["Weekly_Sales"].clip(lower=0)
    return df


def _flag_holiday(df, dates, name):
    df[name] = df["Date"].isin(dates).astype(int)
    return df


def add_calendar_features(df):
    df = df.copy()
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Week_sin"] = np.sin(2 * np.pi * df["WeekOfYear"] / 52)
    df["Week_cos"] = np.cos(2 * np.pi * df["WeekOfYear"] / 52)
    df = _flag_holiday(df, SUPER_BOWL, "is_superbowl")
    df = _flag_holiday(df, LABOR_DAY, "is_labor_day")
    df = _flag_holiday(df, THANKSGIVING, "is_thanksgiving")
    df = _flag_holiday(df, CHRISTMAS, "is_christmas")
    return df


def time_split(df, months=3):
    df = df.sort_values("Date")
    cutoff_date = df["Date"].max() - pd.DateOffset(months=months)
    train_part = df[df["Date"] <= cutoff_date].copy()
    valid_part = df[df["Date"] > cutoff_date].copy()
    return train_part, valid_part


# Orchestrator

def run_pipeline(data_dir, out_dir, months_valid=3, save=True):
    """Runs preprocessing with leakage-safe validation split."""
    train, test, features, stores = load_raw(data_dir)
    train_full_raw, test_full = merge_sources(train, test, features, stores)

    # Split BEFORE fill_grid to avoid future Store-Dept existence leakage
    train_part_raw, valid_part = time_split(train_full_raw, months=months_valid)

    # Fill grid only on the training part for validation experiments
    train_part = fill_grid(train_part_raw, features, stores)

    # Full train is still useful for final model training/inference later
    train_full = fill_grid(train_full_raw, features, stores)

    train_part = add_markdown_flags(train_part)
    valid_part = add_markdown_flags(valid_part)
    train_full = add_markdown_flags(train_full)
    test_full = add_markdown_flags(test_full)

    train_part = flag_negative_sales(train_part)
    valid_part = flag_negative_sales(valid_part)
    train_full = flag_negative_sales(train_full)

    train_part = add_calendar_features(train_part)
    valid_part = add_calendar_features(valid_part)
    train_full = add_calendar_features(train_full)
    test_full = add_calendar_features(test_full)

    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_part.to_parquet(out_dir / "train_part.parquet", index=False)
        valid_part.to_parquet(out_dir / "valid_part.parquet", index=False)
        train_full.to_parquet(out_dir / "train_full.parquet", index=False)
        test_full.to_parquet(out_dir / "test_full.parquet", index=False)
        print("Saved parquet files to", out_dir)

    return train_part, valid_part, train_full, test_full

def load_processed(out_dir):
    """Load already-saved parquet outputs (skip re-running the pipeline)."""
    out_dir = Path(out_dir)
    train_part = pd.read_parquet(out_dir / "train_part.parquet")
    valid_part = pd.read_parquet(out_dir / "valid_part.parquet")
    train_full = pd.read_parquet(out_dir / "train_full.parquet")
    test_full = pd.read_parquet(out_dir / "test_full.parquet")
    return train_part, valid_part, train_full, test_full
