def plot_pdf(data, weights=None, ax=None, linear_bins=False,
             x_scale='log', y_scale='log', **kwargs):
    """
    Plots the probability density function (PDF) to a new figure or to axis ax
    if provided.

    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    linear_bins : bool, optional
        Whether to use linearly spaced bins (True) or logarithmically
        spaced bins (False). False by default.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    edges, hist = pdf(data, weights=weights, linear_bins=linear_bins, **kwargs)
    bin_centers = (edges[1:]+edges[:-1])/2.0
    from numpy import nan
    hist[hist==0] = nan
    if not ax:
        import matplotlib.pyplot as plt
        plt.plot(bin_centers, hist, **kwargs)
        ax = plt.gca()
    else:
        ax.plot(bin_centers, hist, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    return ax

def pdf(data, weights=None, linear_bins=False, **kwargs):
    """
    Returns the probability density function (normalized histogram) of the
    data.

    Parameters
    ----------
    data : list or array
    xmin : float, optional
        Minimum value of the PDF. If None, uses the smallest value in the data.
    xmax : float, optional
        Maximum value of the PDF. If None, uses the largest value in the data.
    linear_bins : float, optional
        Whether to use linearly spaced bins, as opposed to logarithmically
        spaced bins (recommended for log-log plots).

    Returns
    -------
    bin_edges : array
        The edges of the bins of the probability density function.
    probabilities : array
        The portion of the data that is within the bin. Length 1 less than
        bin_edges, as it corresponds to the spaces between them.
    """
    from numpy import logspace, histogram, floor, unique,asarray
    from math import ceil, log10
    data = asarray(data)
    xmax = max(data)
    xmin = min(data)

    if xmin<1:  #To compute the pdf also from the data below x=1, the data, xmax and xmin are rescaled dividing them by xmin.
        xmax2=xmax/xmin
        xmin2=1
    else:
        xmax2=xmax
        xmin2=xmin

    if 'bins' in kwargs.keys():
        bins = kwargs.pop('bins')
    elif linear_bins:
        bins = range(int(xmin2), ceil(xmax2)+1)
    else:
        log_min_size = log10(xmin2)
        log_max_size = log10(xmax2)
        number_of_bins = ceil((log_max_size-log_min_size)*10)
        bins = logspace(log_min_size, log_max_size, num=number_of_bins)
        bins[:-1] = floor(bins[:-1])
        bins[-1] = ceil(bins[-1])
        bins = unique(bins)

    if xmin<1: #Needed to include also data x<1 in pdf.
        hist, edges = histogram(data/xmin, bins, density=True, weights=weights)
        edges=edges*xmin # transform result back to original
        hist=hist/xmin # rescale hist, so that np.sum(hist*edges)==1
    else:
        hist, edges = histogram(data, bins, density=True, weights=weights)
        #from numpy import diff
        #hist = hist * diff(edges)
    return edges, hist

def plot_cdf(data, weights=None, ax=None, survival=False,
             x_scale='log', y_scale='log', **kwargs):
    """
    Plots the cumulative distribution function (CDF) of the data to a new
    figure or to axis ax if provided.

    Parameters
    ----------
    data : list or array
    ax : matplotlib axis, optional
        The axis to which to plot. If None, a new figure is created.
    survival : bool, optional
        Whether to plot a CDF (False) or CCDF (True). False by default.

    Returns
    -------
    ax : matplotlib axis
        The axis to which the plot was made.
    """
    bins, CDF = cdf(data, weights=weights, survival=survival, **kwargs)
    if not ax:
        import matplotlib.pyplot as plt
        plt.plot(bins, CDF, **kwargs)
        ax = plt.gca()
    else:
        ax.plot(bins, CDF, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    return ax

def cdf(data, weights=None, survival=False, **kwargs):
    """
    The cumulative distribution function (CDF) of the data.

    Parameters
    ----------
    data : list or array, optional
    survival : bool, optional
        Whether to calculate a CDF (False) or CCDF (True). False by default.

    Returns
    -------
    X : array
        The sorted, unique values in the data.
    probabilities : array
        The portion of the data that is less than or equal to X.
    """
    return cumulative_distribution_function(data, weights=weights,survival=survival, **kwargs)

def cumulative_distribution_function(data, weights=None, survival=False, **kwargs):
    """
    The cumulative distribution function (CDF) of the data.

    Parameters
    ----------
    data : list or array, optional
    survival : bool, optional
        Whether to calculate a CDF (False) or CCDF (True). False by default.

    Returns
    -------
    X : array
        The sorted, unique values in the data.
    probabilities : array
        The portion of the data that is less than to X.
    """

    from numpy import array, vstack, lexsort, vsplit
    data = array(data)
    if not data.any():
        from numpy import nan
        return array([nan]), array([nan])
    n = float(len(data))
    from numpy import sort
    if weights is None:
        data = sort(data)
    else:
        weights = array(weights)
        weights = weights/sum(weights)
        weights_data = vstack((weights, data))
        weights_data = weights_data[:,lexsort(weights_data)]
        weights, data = [x.flatten() for x in vsplit(weights_data, 2)]
    all_unique = not( any( data[:-1]==data[1:] ) )

    if all_unique:
        from numpy import arange
        CDF = arange(n)/n
    else:
#This clever bit is a way of using searchsorted to rapidly calculate the
#CDF of data with repeated values comes from Adam Ginsburg's plfit code,
#specifically https://github.com/keflavich/plfit/commit/453edc36e4eb35f35a34b6c792a6d8c7e848d3b5#plfit/plfit.py
        from numpy import searchsorted, unique, cumsum
        if weights is None:
            CDF = searchsorted(data, data,side='left')/n
        else:
            CDF = cumsum(weights)
            CDF = CDF - weights
        unique_data, unique_indices = unique(data, return_index=True)
        data=unique_data
        CDF = CDF[unique_indices]
    if survival:
        CDF = 1-CDF
    return data, CDF