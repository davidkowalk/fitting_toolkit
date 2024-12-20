# fitting_toolkit.utils

This module contains general utility functions.

## fitting_toolkit.utils.array

Generates numpy array from arguments

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| *x       | any      | Elements of array

| Returns  | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| x        | numpy.ndarray | Array generated from arguments.

The following commands are equivalen

```py
numpy.array([1, 2, 3, 4])
fitting_toolkit.utils.array(1, 2, 3, 4)
```

## fitting_toolkit.utils.args_to_dict

Returns key word arguments as dictionary.
```
args_to_dict(**kwargs)
```


## fitting_toolkit.utils.get_sigma_probability

```py
get_sigma_probability(n: float = 1)
```

To get probability to fall into n-sigma interval call assuming a normal distribution.

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| n        | float    | Number of sigmas in interval

| Returns | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| p        | float    | Probability of falling into sigma interval. $P(\mu - n*\sigma < X < \mu + n*\sigma) $


## fitting_toolkit.utils.generate_thresholds

```py
generate_thresholds(data, lower_frac=1/6, upper_frac=5/6)`
```

Given a bootstrapped distribution, generate a custom confidence interval.

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| data     | np.array | At a defined point x give the y value for all points in parameter space.
| lower_frac | float  | Fraction of data below lower threshold
| upper_frac | float  | Fraction of data below upper threshold


| Returns | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| lower_threshold | float | Point defined by `lower_frac` 
| upper_threshold | float | Point defined by `upper_frac`