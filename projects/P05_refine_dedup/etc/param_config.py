import numpy as np

# ranges for grid search
PARAMS_RANGES_CONFIG = {
    "primary": {
        "eps": np.arange(0.30, 0.35, 0.01),
        "min_samples": [2, 3, 4, 5],
    },
    "noise": {
        "eps": np.arange(0.41, 0.46, 0.01),
        "min_samples": [2, 3],
    },
    "big": {
        "eps": np.arange(0.25, 0.30, 0.01),
        "min_samples": [2, 3],
    },
}

# BEST_CONFIG = {
#     "primary": {
#         "eps": 0.3,
#         "min_samples": 5,
#     },
#     "noise": {
#         "eps": 0.2,
#         "min_samples": 3,
#     },
#     "big": {
#         "eps": 0.1,
#         "min_samples": 2,
#     },
# }
