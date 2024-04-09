#antes del ref

def pairwise_distances(X, Y=None, metric='euclidean', n_jobs=None, **kwds):
    if metric == "precomputed":
        X, _ = check_pairwise_arrays(
            X, Y, precomputed=True, force_all_finite=force_all_finite
        )

        whom = (
            "pairwise_distances. Precomputed distance "
            " need to have non-negative values."
        )
        check_non_negative(X, whom=whom)
        return X
    elif metric in PAIRWISE_DISTANCE_FUNCTIONS:
        func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
    elif callable(metric):
        func = partial(
            _pairwise_callable,
            metric=metric,
            force_all_finite=force_all_finite,
            **kwds,
        )
    else:
        if issparse(X) or issparse(Y):
            raise TypeError("scipy distance metrics do not support sparse matrices.")

        dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else "infer_float"

        if dtype == bool and (X.dtype != bool or (Y is not None and Y.dtype != bool)):
            msg = "Data was converted to boolean for metric %s" % metric
            warnings.warn(msg, DataConversionWarning)

        X, Y = check_pairwise_arrays(
            X, Y, dtype=dtype, force_all_finite=force_all_finite
        )

        # precompute data-derived metric params
        params = _precompute_metric_params(X, Y, metric=metric, **kwds)
        kwds.update(**params)

        if effective_n_jobs(n_jobs) == 1 and X is Y:
            return distance.squareform(distance.pdist(X, metric=metric, **kwds))
        func = partial(distance.cdist, metric=metric, **kwds)

    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
    
    #despues del ref
   def pairwise_distances(
    X,
    Y=None,
    metric="euclidean",
    *,
    n_jobs=None,
    force_all_finite=True,
    **kwds,
):
    # Si la métrica se calcula previamente, verifique y devuelva la matriz de distancia
    if metric == "precomputed":
        X, _ = check_pairwise_arrays(X, Y, precomputed=True, force_all_finite=force_all_finite)
        check_non_negative(X, whom="precomputed distance matrix")
        return X

    # Compruebe si las matrices de entrada son escasas y convierta el tipo de datos si es necesario
    if isinstance(metric, str):
        if metric in PAIRWISE_DISTANCE_FUNCTIONS:
            func = PAIRWISE_DISTANCE_FUNCTIONS[metric]
        else:
            raise ValueError(f"Invalid metric: {metric}")
    elif callable(metric):
        func = partial(
            _pairwise_callable,
            metric=metric,
            force_all_finite=force_all_finite,
            **kwds,
        )
    else:
        raise TypeError("Metric must be a string or callable")

    # Check if the input arrays are sparse and convert data type if necessary
    if issparse(X) or issparse(Y):
        raise TypeError("scipy distance metrics do not support sparse matrices.")

    dtype = bool if metric in PAIRWISE_BOOLEAN_FUNCTIONS else "infer_float"
    X, Y = check_pairwise_arrays(X, Y, dtype=dtype, force_all_finite=force_all_finite)

    # Precalcular parámetros métricos y actualizar argumentos de palabras clave
    params = _precompute_metric_params(X, Y, metric=metric, **kwds)
    kwds.update(**params)

    # Calcular distancias por pares usando la función apropiada
    if effective_n_jobs(n_jobs) == 1 and X is Y:
        return distance.squareform(distance.pdist(X, metric=metric, **kwds))
    return _parallel_pairwise(X, Y, func, n_jobs, **kwds)
    
    '''Se realizó optimizaciones en el código para la verificación de métricas, simplificación de la conversión de tipos de datos, precalculo de parámetros y mejora en el cálculo de distancias, eliminando pasos innecesarios y redundantes para aumentar la eficiencia'''