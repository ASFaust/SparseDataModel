import numpy as np

def norm_ppf(p: np.ndarray | float) -> np.ndarray | float:
    """
    Inverse CDF (ppf) of the standard normal distribution
    using the rational approximation from P. J. Acklam (AS 241),
    with infinities for p = 0 or 1.
    Absolute error ≲1e-9 for p in (0,1).

    Parameters
    ----------
    p : float or array-like
        Probabilities in [0, 1].

    Returns
    -------
    z : float or ndarray
        Quantiles such that Φ(z) = p, with z = -inf for p=0 and +inf for p=1.
    """
    p_arr = np.asarray(p, dtype=float)
    # prepare output array
    z = np.empty_like(p_arr)

    # invalid outside [0,1]
    invalid = (p_arr < 0) | (p_arr > 1)
    if np.any(invalid):
        raise ValueError("p must be in the interval [0, 1].")

    # edge cases
    z[p_arr == 0] = -np.inf
    z[p_arr == 1] =  np.inf

    # mask for valid interior points (0 < p < 1)
    mask = (p_arr > 0) & (p_arr < 1)
    if np.any(mask):
        pm = p_arr[mask]

        # coefficients
        a = np.array([-3.969683028665376e+01,  2.209460984245205e+02,
                      -2.759285104469687e+02,  1.383577518672690e+02,
                      -3.066479806614716e+01,  2.506628277459239e+00])
        b = np.array([-5.447609879822406e+01,  1.615858368580409e+02,
                      -1.556989798598866e+02,  6.680131188771972e+01,
                      -1.328068155288572e+01])
        c = np.array([-7.784894002430293e-03, -3.223964580411365e-01,
                      -2.400758277161838e+00, -2.549732539343734e+00,
                       4.374664141464968e+00,  2.938163982698783e+00])
        d = np.array([ 7.784695709041462e-03,  3.224671290700398e-01,
                       2.445134137142996e+00,  3.754408661907416e+00])

        plow, phigh = 0.02425, 1 - 0.02425
        zm = np.empty_like(pm)

        # region 1: lower tail
        m1 = pm < plow
        if np.any(m1):
            q = np.sqrt(-2*np.log(pm[m1]))
            zm[m1] = -(
                (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
            )

        # region 2: central
        m2 = (~m1) & (pm <= phigh)
        if np.any(m2):
            q = pm[m2] - 0.5
            r = q*q
            zm[m2] = (
                (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
            ) / (
                (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
            )

        # region 3: upper tail
        m3 = pm > phigh
        if np.any(m3):
            q = np.sqrt(-2*np.log(1 - pm[m3]))
            zm[m3] = (
                (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
            )

        z[mask] = zm

    # return scalar if input was scalar
    if np.isscalar(p):
        return float(z)
    return z


if __name__ == "__main__":
    # Example usage
    p_values = [0.01, 0.5, 0.99, 0.9999]
    results = norm_ppf(p_values)
    print("p-values:", p_values)
    print("Quantiles:", results)

    # Test edge cases
    print("Quantile for p=0:", norm_ppf(0))  # should be -inf
    print("Quantile for p=1:", norm_ppf(1))  # should be inf