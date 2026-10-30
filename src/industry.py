from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCES = {
    "patents": {
        "file": "patents.xlsx",
        "url": "https://raw.githubusercontent.com/PetraMoser/Pirates_and_Patents/c5c38cc4e59a803fcc8df800a29091eb780dbc98/data/Brit_patents_app_sealed1617-1938.xlsx",
        "page": "https://github.com/PetraMoser/Pirates_and_Patents",
        "sha256": "9729df588b53bc5183f313a98db517f34819f088971a49e56702278283814722",
    },
    "coal": {
        "file": "coal.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=A01211GBA418NNBR",
        "page": "https://fred.stlouisfed.org/series/A01211GBA418NNBR",
        "sha256": "cc0c471cb4257fff098d346c12ebbe1b9ef0d26d8d5a3d0db34f1370e81333f0",
    },
    "rail": {
        "file": "rail.csv",
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=A02085GBA374NNBR",
        "page": "https://fred.stlouisfed.org/series/A02085GBA374NNBR",
        "sha256": "b0d49ee96f7a865441cf533720f44847d680caa7ef2968c8201673c79915f586",
    },
}
LABELS = {"sealed": "Sealed patents", "applications": "Patent applications",
          "coal": "Coal output", "rail": "New railway mileage"}


def validate_frame(frame, value_columns, allow_negative=()):
    """Reject malformed keys and values while retaining genuine missing cells."""
    if "year" not in frame or not set(value_columns).issubset(frame):
        raise ValueError("Required columns are missing")
    years = pd.to_numeric(frame.year, errors="raise")
    if years.isna().any() or not np.isfinite(years).all() or (years % 1 != 0).any():
        raise ValueError("Years must be finite integers")
    if years.duplicated().any() or not years.is_monotonic_increasing:
        raise ValueError("Years must be unique and ordered")
    for column in value_columns:
        values = pd.to_numeric(frame[column], errors="raise").dropna()
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {column}")
        if column not in allow_negative and (values < 0).any():
            raise ValueError(f"Negative values in {column}")
    return frame


def read_source(path, name):
    path = Path(path)
    if name == "patents":
        if not zipfile.is_zipfile(path):
            raise ValueError("The patent response is not an XLSX workbook")
        frame = pd.read_excel(path, sheet_name="Sheet1")
        if list(frame.columns) != ["year", "applications", "sealed"]:
            raise ValueError("Unexpected patent workbook schema")
        validate_frame(frame, ["applications", "sealed"])
    else:
        frame = pd.read_csv(path, na_values=["."])
        series = {"coal": "A01211GBA418NNBR", "rail": "A02085GBA374NNBR"}[name]
        if list(frame.columns) != ["observation_date", series]:
            raise ValueError(f"Unexpected {name} CSV schema")
        dates = pd.to_datetime(frame.observation_date, errors="raise")
        if not ((dates.dt.month == 1) & (dates.dt.day == 1)).all():
            raise ValueError("Expected annual observations dated January 1")
        frame = pd.DataFrame({"year": dates.dt.year,
                              name: pd.to_numeric(frame[series], errors="raise")})
        # Later railway closures can create genuine negative net additions.
        validate_frame(frame, [name], allow_negative=("rail",))
    frame["year"] = frame.year.astype(int)
    return frame


def download_sources(root, refresh=False):
    """GET actual bytes, check schema/checksum, and replace only valid files."""
    root = Path(root)
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "HistoricalIndustryResearch/1.0"
    session.mount("https://", HTTPAdapter(max_retries=Retry(
        total=2, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])))
    checks = []
    for name, spec in SOURCES.items():
        path = raw / spec["file"]
        record = {"source": name, "url": spec["url"], "page": spec["page"],
                  "checked_utc": datetime.now(timezone.utc).isoformat()}
        if path.exists() and not refresh:
            payload = path.read_bytes()
            record.update(mode="bundled_snapshot", http_status=None)
        else:
            response = session.get(spec["url"], timeout=(15, 60))
            response.raise_for_status()
            payload = response.content
            record.update(mode="network_get", http_status=response.status_code,
                          final_url=response.url,
                          content_type=response.headers.get("Content-Type", ""))
        digest = hashlib.sha256(payload).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"{name}: source bytes changed; review the new data before use")
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            temporary.write_bytes(payload)
            # Parse the response rather than treating HTTP 200 as sufficient.
            frame = read_source(temporary, name)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        record.update(bytes=len(payload), sha256=digest, rows=len(frame),
                      schema_valid=True)
        checks.append(record)
    (raw / "source_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return pd.DataFrame(checks)


def load_panel(root):
    raw = Path(root) / "data/raw"
    frames = {name: read_source(raw / spec["file"], name)
              for name, spec in SOURCES.items()}
    panel = pd.DataFrame({"year": np.arange(1802, 1903)})
    for frame in frames.values():
        panel = panel.merge(frame, on="year", how="left", validate="one_to_one")
    panel = panel.set_index("year")
    # A complete calendar prevents shifts from bridging unobserved years.
    for column in ["sealed", "coal"]:
        if (panel[column].dropna() <= 0).any():
            raise ValueError("Log growth requires strictly positive observations")
        panel[column + "_growth"] = 100 * np.log(panel[column]).diff()
    if (panel.rail.dropna() < 0).any():
        raise ValueError("Railway log1p changes require nonnegative values in the study window")
    panel["rail_change"] = 100 * np.log1p(panel.rail).diff()
    coverage = []
    for column in ["sealed", "applications", "coal", "rail"]:
        valid = panel[column].dropna()
        start, end = int(valid.index.min()), int(valid.index.max())
        coverage.append({"indicator": LABELS[column], "first_year": start,
                         "last_year": end, "observed_years": len(valid),
                         "missing_inside_span": int(panel.loc[start:end, column].isna().sum())})
    return panel, pd.DataFrame(coverage), frames


def cagr(series, start, end):
    """Annualized endpoint growth; the exponent uses elapsed calendar years."""
    if end <= start or pd.isna(series.loc[start]) or pd.isna(series.loc[end]):
        raise ValueError("CAGR requires two observed endpoints in chronological order")
    if series.loc[start] <= 0 or series.loc[end] <= 0:
        raise ValueError("CAGR requires positive endpoints")
    return 100 * ((series.loc[end] / series.loc[start]) ** (1 / (end - start)) - 1)


def summarize_growth(panel):
    windows = [("Sealed patents", "sealed", 1802, 1851),
               ("Sealed patents", "sealed", 1853, 1902),
               ("Coal output", "coal", 1854, 1878),
               ("Coal output", "coal", 1878, 1902)]
    growth = pd.DataFrame([{"indicator": label, "start": start, "end": end,
                           "start_value": panel.loc[start, col],
                           "end_value": panel.loc[end, col],
                           "cagr_pct": cagr(panel[col], start, end)}
                          for label, col, start, end in windows])
    local = []
    for start, end in [(1845, 1851), (1853, 1859)]:
        for col in ["sealed", "rail"]:
            values = panel.loc[start:end, col].dropna()
            local.append({"indicator": LABELS[col], "start": start, "end": end,
                          "n": len(values), "annual_mean": values.mean()})
    return growth, pd.DataFrame(local)


def segmented_fit(years, values, breakpoint=None):
    """OLS with a possible level jump and slope change in log sealed patents."""
    years, y = np.asarray(years, float), np.log(np.asarray(values, float))
    time = years - years.min()
    columns = [np.ones(len(years)), time]
    if breakpoint is not None:
        columns += [(years >= breakpoint).astype(float),
                    np.maximum(years - breakpoint, 0)]
    X = np.column_stack(columns)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    sse = float(np.sum((y - fitted) ** 2))
    bic = len(y) * np.log(max(sse / len(y), 1e-15)) + X.shape[1] * np.log(len(y))
    return {"beta": beta, "fitted": fitted, "sse": sse, "bic": bic,
            "rmse_log": np.sqrt(sse / len(y)), "n": len(y)}


def analyze_reform(panel):
    sample = panel.loc[1802:1882, "sealed"].drop(index=1852).dropna()
    base = segmented_fit(sample.index, sample.values)
    fixed = segmented_fit(sample.index, sample.values, 1853)
    scan = []
    for year in range(1817, 1869):
        # A candidate denotes the first observed post-break year. Do not
        # assign separate dates to the same partition across an omitted year.
        if year in sample.index and (sample.index < year).sum() >= 15 and (sample.index >= year).sum() >= 15:
            result = segmented_fit(sample.index, sample.values, year)
            scan.append({"first_post_year": year, "bic_fixed_date": result["bic"]})
    scan = pd.DataFrame(scan)
    best_year = int(scan.loc[scan.bic_fixed_date.idxmin(), "first_post_year"])
    best = segmented_fit(sample.index, sample.values, best_year)
    # One additional parameter penalizes selection of the breakpoint itself.
    models = pd.DataFrame([
        {"model": "Single trend", "first_post_year": np.nan,
         "parameters": 2, "bic": base["bic"], "rmse_log": base["rmse_log"]},
        {"model": "Reform date", "first_post_year": 1853,
         "parameters": 4, "bic": fixed["bic"], "rmse_log": fixed["rmse_log"]},
        {"model": "Scanned date", "first_post_year": best_year,
         "parameters": 5, "bic": best["bic"] + np.log(len(sample)),
         "rmse_log": best["rmse_log"]},
    ])
    summary = {"n": len(sample), "best_first_post_year": best_year,
               "jump_pct": float(100 * np.expm1(fixed["beta"][2])),
               "pre_trend_pct": float(100 * np.expm1(fixed["beta"][1])),
               "post_trend_pct": float(100 * np.expm1(fixed["beta"][1] + fixed["beta"][3]))}
    return sample, base, fixed, scan, models, summary


def residualize(values, years):
    X = np.column_stack([np.ones(len(years)), np.asarray(years) - min(years)])
    return values - X @ np.linalg.lstsq(X, values, rcond=None)[0]


def correlation_analysis(panel, exclude_years=()):
    sample = panel.loc[1854:1902].drop(index=list(exclude_years), errors="ignore")
    rows, lags = [], []
    for target, changes in [("coal", "coal_growth"), ("rail", "rail_change")]:
        levels = sample[["sealed", target]].dropna()
        x = np.log(levels.sealed.to_numpy())
        y = np.log(levels[target].to_numpy()) if target == "coal" else np.log1p(levels[target].to_numpy())
        for name, a, b in [("Log levels", x, y),
                           ("Detrended log levels", residualize(x, levels.index),
                            residualize(y, levels.index))]:
            rows.append({"target": LABELS[target], "representation": name,
                         "n": len(a), "r": float(np.corrcoef(a, b)[0, 1])})
        changes_frame = sample[["sealed_growth", changes]].dropna()
        rows.append({"target": LABELS[target], "representation": "Annual log changes",
                     "n": len(changes_frame),
                     "r": changes_frame.corr().iloc[0, 1]})
        for lag in range(-5, 6):
            # Shift on the full calendar before filtering exclusions.
            aligned = pd.DataFrame({"patents": panel.sealed_growth.shift(lag),
                                    "target": panel[changes]}).loc[1854:1902]
            aligned = aligned.drop(index=list(exclude_years), errors="ignore").dropna()
            lags.append({"target": LABELS[target], "patent_lead_years": lag,
                         "n": len(aligned), "r": aligned.corr().iloc[0, 1]})
    return pd.DataFrame(rows), pd.DataFrame(lags)


def forecast_features(panel, target):
    sample = pd.DataFrame({"target": panel[target], "own_lag": panel[target].shift(1),
                           "time": panel.index.to_numpy() - 1854,
                           "patent_lag": panel.sealed_growth.shift(1)}, index=panel.index)
    return sample.loc[1854:1902].dropna()


def expanding_evaluation(panel, min_train=20):
    predictions = []
    for target in ["coal_growth", "rail"]:
        sample = forecast_features(panel, target)
        for i in range(min_train, len(sample)):
            train, test = sample.iloc[:i], sample.iloc[[i]]
            assert train.index.max() < test.index.min()
            for name, features in [("Historical mean", []),
                                   ("Own lag + time", ["own_lag", "time"]),
                                   ("Own lag + time + patents", ["own_lag", "time", "patent_lag"])]:
                if features:
                    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
                    model.fit(train[features], train.target)
                    estimate = float(model.predict(test[features])[0])
                else:
                    estimate = float(train.target.mean())
                predictions.append({"target": target, "model": name,
                                    "year": int(test.index[0]), "train_end": int(train.index[-1]),
                                    "train_n": len(train), "actual": float(test.target.iloc[0]),
                                    "prediction": estimate})
    predictions = pd.DataFrame(predictions)
    metrics = []
    for (target, name), group in predictions.groupby(["target", "model"], sort=False):
        error = group.prediction - group.actual
        metrics.append({"target": target, "model": name, "test_n": len(group),
                        "first_test": group.year.min(), "last_test": group.year.max(),
                        "mae": np.abs(error).mean(), "rmse": np.sqrt(np.mean(error ** 2))})
    return predictions, pd.DataFrame(metrics)


def save_table(frame, root, name):
    path = Path(root) / "outputs/tables" / (name + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
