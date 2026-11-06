"""Simulation and scale estimators for the self-affine signals project.

The conventions and usable fitting bands are documented in the notebook.
"""

import numpy as np
from scipy import signal, stats
import pywt


def _slope(x, y):
    x, y = np.asarray(x), np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def davies_harte(n, h, rng):
    if n < 2 or not 0 < h < 1:
        raise ValueError("Require n >= 2 and 0 < H < 1")
    k = np.arange(n, dtype=float)
    gamma = np.empty(n)
    gamma[0] = 1
    gamma[1:] = 0.5 * ((k[1:] + 1) ** (2 * h) - 2 * k[1:] ** (2 * h)
                       + (k[1:] - 1) ** (2 * h))
    gamma_n = 0.5 * ((n+1)**(2*h)-2*n**(2*h)+(n-1)**(2*h))
    return _embed_covariance(gamma, gamma_n, rng)


def _embed_covariance(gamma, middle, rng):
    n = len(gamma)
    c = np.r_[gamma, middle, gamma[:0:-1]]
    eig = np.fft.rfft(c).real
    if eig.min() < -1e-8:
        raise ValueError("Circulant embedding is not positive semidefinite")
    z = rng.standard_normal(2 * n)
    return np.fft.irfft(np.sqrt(np.maximum(eig, 0)) * np.fft.rfft(z),
                        n=2 * n)[:n]


def cauchy_process(n, h_local, h_global, rng):
    """Generalized Cauchy: C(k)=(1+|k|^(2Hloc))^[-(1-Hglob)/Hloc]."""
    if not 0 < h_local < 1 or not 0 < h_global < 1:
        raise ValueError("H parameters must be in (0,1)")
    k = np.arange(n, dtype=float)
    a, b = 2*h_local, 2-2*h_global
    covariance = (1 + k**a)**(-b/a)
    return _embed_covariance(covariance, (1+n**a)**(-b/a), rng)


def spectral_synthesis(n, alpha, rng):
    """Periodic Fourier-filtered Gaussian sequence; DC is removed."""
    z = rng.standard_normal(n)
    f = np.fft.rfftfreq(n)
    shape = np.zeros_like(f)
    shape[1:] = f[1:]**(-alpha/2)
    x = np.fft.irfft(np.fft.rfft(z)*shape, n=n)
    return (x-x.mean())/x.std()


def lorentzian_synthesis(n, rng, low_rate=1/1024, high_rate=1/2):
    """Gaussian synthesis with a continuum mixture of Lorentzian spectra.

    Integral d(lambda)/(lambda^2+f^2) gives an interior 1/f band and
    a low-frequency plateau. Frequencies are in cycles/sample.
    """
    if not 0 < low_rate < high_rate <= 0.5:
        raise ValueError("Require 0 < low_rate < high_rate <= 0.5")
    f = np.fft.rfftfreq(n)
    spectrum = np.empty_like(f)
    spectrum[0] = 1/low_rate - 1/high_rate
    spectrum[1:] = (np.arctan(high_rate/f[1:]) -
                    np.arctan(low_rate/f[1:]))/f[1:]
    spectrum[0] = 0  # remove the random constant component
    z = rng.standard_normal(n)
    x = np.fft.irfft(np.fft.rfft(z)*np.sqrt(spectrum), n=n)
    return (x-x.mean())/x.std()


def _scales(n, lower=8, upper_fraction=8, count=12):
    top = max(lower + 1, n // upper_fraction)
    return np.unique(np.geomspace(lower, top, count).astype(int))


def dfa(x, scales=None, order=1, return_curve=False):
    x = np.asarray(x, float)
    n = len(x)
    if scales is None:
        scales = _scales(n, 16, 8)
    y = np.cumsum(x - x.mean())
    used, values = [], []
    for s in scales:
        s = int(s)
        ns = n // s
        if s <= order + 2 or ns < 4:
            continue
        blocks = np.concatenate((y[:ns*s].reshape(ns, s),
                                 y[n-ns*s:].reshape(ns, s)), axis=0)
        t = np.linspace(-1, 1, s)
        basis = np.polynomial.polynomial.polyvander(t, order)
        fitted = (blocks @ np.linalg.pinv(basis).T) @ basis.T
        fluct = np.sqrt(np.mean((blocks - fitted) ** 2))
        if fluct > 0:
            used.append(s)
            values.append(fluct)
    exponent = _slope(np.log(used), np.log(values))
    return (exponent, np.array(used), np.array(values)) if return_curve else exponent


def spectral_alpha(x, method="periodogram"):
    x = np.asarray(x, float)
    n = len(x)
    if method == "periodogram":
        f, p = signal.periodogram(x, detrend="linear", window="boxcar")
    elif method == "welch":
        f, p = signal.welch(x, nperseg=min(n // 4, 4096),
                            detrend="linear", noverlap=None)
    else:
        raise ValueError(method)
    band = (f >= max(4/n, 1/4096)) & (f <= 1/8) & (p > 0)
    return -_slope(np.log(f[band]), np.log(p[band]))


def acf_beta(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    fft = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(abs(fft)**2, n=2*n)[:n]
    acf /= np.maximum(n - np.arange(n), 1)
    acf /= acf[0]
    lags = np.unique(np.geomspace(2, min(n//16, 512), 30).astype(int))
    vals = acf[lags]
    # The positive-tail estimator is defined for persistent fGn only.
    return -_slope(np.log(lags[vals > 0]), np.log(vals[vals > 0]))


def aggregate_beta(x):
    x = np.asarray(x, float)
    sizes = _scales(len(x), 4, 16, 10)
    used, variances = [], []
    for s in sizes:
        ns = len(x) // s
        if ns < 8:
            continue
        v = np.var(x[:ns*s].reshape(ns, s).mean(axis=1), ddof=1)
        if v > 0:
            used.append(s)
            variances.append(v)
    return -_slope(np.log(used), np.log(variances))


def structure_h(path):
    path = np.asarray(path, float)
    lags = _scales(len(path), 1, 32, 12)
    sd = [np.std(path[k:] - path[:-k], ddof=1) for k in lags]
    return _slope(np.log(lags), np.log(sd))


def rs_h(x):
    x = np.asarray(x, float)
    sizes = _scales(len(x), 16, 16, 10)
    r = []
    for s in sizes:
        ns = len(x)//s
        blocks = x[:ns*s].reshape(ns, s)
        profile = np.cumsum(blocks - blocks.mean(axis=1)[:,None], axis=1)
        sd = blocks.std(axis=1, ddof=1)
        good = sd > 0
        r.append(np.mean(np.ptp(profile[good], axis=1)/sd[good]) if good.any() else np.nan)
    return _slope(np.log(sizes), np.log(r))


def wavelet_alpha(x):
    coeffs = pywt.wavedec(np.asarray(x, float), "db4", mode="periodization")
    j = np.arange(1, len(coeffs))
    details = coeffs[:0:-1]
    good = [(jj, np.var(d, ddof=1)) for jj, d in zip(j, details)
            if len(d) >= 16 and jj >= 2]
    if len(good) < 4:
        return np.nan
    jj, v = np.array(good).T
    return _slope(jj*np.log(2), np.log(v))


def wavelet_alpha_fbm(path):
    """Wavelet variance slope on nonstationary fBm with edge coefficients removed."""
    details = pywt.wavedec(np.asarray(path, float), "db4", mode="symmetric")[:0:-1]
    good = [(j, np.var(d[4:-4], ddof=1)) for j, d in enumerate(details, 1)
            if j >= 2 and len(d) >= 32]
    if len(good) < 4:
        return np.nan
    j, variance = np.array(good).T
    return _slope(j*np.log(2), np.log(variance))


def higuchi_dimension(x, kmax=64):
    x = np.asarray(x, float)
    n = len(x)
    ks = np.unique(np.geomspace(2, min(kmax, n//16), 16).astype(int))
    lengths = []
    for k in ks:
        vals = []
        for m in range(k):
            seq = x[m::k]
            if len(seq) < 3:
                continue
            vals.append(np.abs(np.diff(seq)).sum() * (n-1) / ((len(seq)-1)*k*k))
        lengths.append(np.mean(vals))
    return -_slope(np.log(ks), np.log(lengths))


def box_dimension(path):
    y = np.asarray(path, float)
    n = len(y)
    y = (y-y.min())/(np.ptp(y) or 1)
    bins = np.unique(np.geomspace(4, min(128, n//8), 12).astype(int))
    counts = []
    for b in bins:
        edges = np.linspace(0, n, b+1).astype(int)
        total = 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            piece = y[lo:max(lo+1,hi)]
            total += max(1, int(np.floor(piece.max()*b) - np.floor(piece.min()*b) + 1))
        counts.append(total)
    return _slope(np.log(bins), np.log(counts))


def estimate_pair(noise, path, h):
    out = {
        "H_dfa": dfa(noise), "H_rs": rs_h(noise),
        "H_structure": structure_h(path),
        "beta_acf": acf_beta(noise) if h > 0.5 else np.nan,
        "beta_aggregate": aggregate_beta(noise),
        "alpha_periodogram": spectral_alpha(noise),
        "alpha_welch": spectral_alpha(noise, "welch"),
        "alpha_wavelet": wavelet_alpha(noise),
        "alpha_wavelet_fbm": wavelet_alpha_fbm(path),
        "alpha_fbm": spectral_alpha(path),
        "D_higuchi": higuchi_dimension(path),
        "D_box": box_dimension(path),
    }
    return out


def summarize(df):
    records = []
    for keys, g in df.groupby(["process", "H", "N"], dropna=False):
        proc, h, n = keys
        targets = {"H_dfa": h, "H_rs": h, "H_structure": h,
                   "beta_acf": 2-2*h, "beta_aggregate": 2-2*h,
                   "alpha_periodogram": 2*h-1, "alpha_welch": 2*h-1,
                   "alpha_wavelet": 2*h-1, "alpha_fbm": 2*h+1,
                   "alpha_wavelet_fbm": 2*h+1,
                   "D_higuchi": 2-h, "D_box": 2-h}
        for name, target in targets.items():
            v = g[name].dropna().to_numpy()
            if len(v) == 0:
                continue
            records.append(dict(process=proc, H=h, N=n, estimator=name,
                                count=len(v), target=target, mean=v.mean(),
                                bias=v.mean()-target, sd=v.std(ddof=1) if len(v)>1 else np.nan,
                                rmse=np.sqrt(np.mean((v-target)**2))))
    import pandas as pd
    return pd.DataFrame(records)


def all_relation_residuals(df, rng, reps=5000):
    """Paired residuals for every distinct scaling identity used in the study.

    Percentile bootstrap intervals describe sampling variation; two-sided
    one-sample t tests supply p-values corrected over the complete family.
    """
    import pandas as pd
    rows = []
    for (proc, h, n), g in df.groupby(["process", "H", "N"]):
        tests = {
            "beta_H": g.beta_aggregate-(2-2*g.H_dfa),
            "alpha_g_beta": g.alpha_welch-(1-g.beta_aggregate),
            "alpha_B_g": g.alpha_fbm-(g.alpha_welch+2),
            "D_H": g.D_higuchi-(2-g.H_dfa),
            "D_alpha_B": g.D_higuchi-(5-g.alpha_fbm)/2,
            "D_beta": g.D_higuchi-(1+g.beta_aggregate/2),
            "wavelet_g": g.alpha_wavelet-g.alpha_welch,
            "wavelet_B": g.alpha_wavelet_fbm-g.alpha_fbm,
        }
        for relation, series in tests.items():
            x = series.dropna().to_numpy()
            if len(x) < 3:
                continue
            draws = x[rng.integers(0, len(x), (reps, len(x)))].mean(axis=1)
            p = (float(stats.ttest_1samp(x, 0).pvalue)
                 if np.std(x, ddof=1)>0 else float(x.mean()==0))
            rows.append(dict(process=proc, H=h, N=n, relation=relation,
                             count=len(x), mean=x.mean(),
                             lo=np.quantile(draws, .025), hi=np.quantile(draws, .975),
                             p_t=p))
    out = pd.DataFrame(rows)
    if len(out):
        out["p_bonferroni"] = np.minimum(1., out.p_t * len(out))
        out["reject_05"] = out.p_bonferroni < .05
    return out


def primary_relation_residuals(all_relations):
    """Select the three task examples without changing their test family."""
    names = {"alpha_g_beta": "alpha=1-beta", "D_H": "D=2-H",
             "beta_H": "beta=2-2H"}
    selected = all_relations.loc[all_relations.relation.isin(names)].copy()
    selected["relation"] = selected.relation.map(names)
    return selected
