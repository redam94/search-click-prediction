import numpy as np
import xarray as xr
import pandas as pd
from scipy.special import expit

from search_click_prediction.transforms import hill


def generate_data_search_data(
    n_days: int = 400, n_geos: int = 10, seed: int = 42, start_date: str = "2020-01-01"
) -> xr.Dataset:
    np.random.seed(seed)
    date = pd.date_range(start_date, periods=n_days, freq="D")
    time = xr.DataArray(np.arange(n_days), coords={"date": date}, dims=["date"])
    geos = ["geo_" + str(i) for i in range(n_geos)]
    lam_offset = xr.DataArray(
        np.random.normal(size=n_geos) * 100, coords={"geo": geos}, dims=["geo"]
    )

    weekly_seasonal_comp = 50 * np.sin(2 * np.pi * time / 7)
    yearly_seasonal_comp = (
        150 * np.sin(2 * np.pi * time / 365) - 50 * np.sin(4 * np.pi * time / 365)
    ) * np.exp(lam_offset * 0.0001)
    lam = (
        300
        + weekly_seasonal_comp
        + yearly_seasonal_comp
        + lam_offset
        + 20 * np.random.random(size=(n_days, n_geos))
    )  # Varies over time

    search_volume = xr.DataArray(
        np.random.poisson(lam=lam * 10),
        coords={"date": date, "geo": geos},
        dims=["date", "geo"],
    )  # Daily search volume

    budget = np.clip(
        (yearly_seasonal_comp + 300)
        + 5 * np.random.normal(size=(n_days, n_geos)).cumsum(axis=0),
        0,
        None,
    )  # Varies over time

    half_sat = np.mean(budget)
    slope = 2.5

    budget[60:67] = budget[60:67] * 1.2  # Double budget for a week
    budget[300:307] = budget[300:307] * 1.2  # Double budget for a week
    budget_expit = expit(hill(budget, half_sat, slope) - 2)
    budget[30:37] = 0  # Turn off budget for a week
    budget[200:207] = 0
    budget_expit = budget_expit.where(budget > 0.01, 0)
    impression_rate = (budget_expit) + 0 * lam_offset  # Varies over time
    impressions = xr.DataArray(
        np.random.binomial(search_volume, impression_rate),
        coords={"date": date, "geo": geos},
        dims=["date", "geo"],
    )  # Impressions from search volume

    yearly_seasonal_comp_ctr = (
        0.02
        * (yearly_seasonal_comp - yearly_seasonal_comp.min())
        / (yearly_seasonal_comp.max() - yearly_seasonal_comp.min())
    )

    consumer_trends = 0.01 * np.cos(np.pi * 2 * time / 7) + 0.02 * expit(
        0.1 * np.random.normal(size=n_days)
    )  # Varies over time

    click_rate = (
        0.04 + yearly_seasonal_comp_ctr + consumer_trends + 0 * lam_offset
    )  # Varies over time

    observed_clicks = xr.DataArray(
        np.random.binomial(impressions, click_rate),
        coords={"date": date, "geo": geos},
        dims=["date", "geo"],
    )  # Clicks can only come from impressions

    dayofyear = xr.DataArray(
        date.dayofyear.to_numpy(), coords={"date": date}, dims=["date"]
    )

    data = xr.Dataset(
        {
            "search_volume": search_volume,
            "impressions": impressions,
            "observed_clicks": observed_clicks,
            "dayofyear": dayofyear,
            "time": time,
            "budget": budget,
        }
    )
    return data
