"""Hydrology-standard regression and interval metrics."""
import numpy as np


def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    d = np.sum((obs - obs.mean()) ** 2)
    return float(1 - np.sum((obs - sim) ** 2) / d) if d > 0 else float("nan")


def kge(obs, sim):
    """Kling-Gupta Efficiency (2009): 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if obs.std() == 0 or obs.mean() == 0:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def regression_metrics(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[ok], sim[ok]
    if len(obs) < 3:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "NSE": np.nan, "KGE": np.nan, "PBIAS_%": np.nan}
    err = sim - obs
    r = np.corrcoef(obs, sim)[0, 1] if obs.std() > 0 and sim.std() > 0 else np.nan
    return {"RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(np.abs(err))),
            "R2": float(r ** 2) if np.isfinite(r) else np.nan, "NSE": nse(obs, sim), "KGE": kge(obs, sim),
            "PBIAS_%": float(100 * err.sum() / obs.sum()) if obs.sum() != 0 else np.nan}


def interval_metrics(obs, lo, hi, alpha):
    """Coverage (PICP), mean width (MPIW) and Winkler/interval score (lower = better)."""
    obs, lo, hi = map(lambda v: np.asarray(v, float), (obs, lo, hi))
    cover = (obs >= lo) & (obs <= hi)
    width = hi - lo
    winkler = width + (2 / alpha) * ((lo - obs) * (obs < lo) + (obs - hi) * (obs > hi))
    return {"PICP": float(cover.mean()), "MPIW": float(width.mean()), "Winkler": float(winkler.mean())}
