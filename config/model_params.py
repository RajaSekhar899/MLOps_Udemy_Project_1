from scipy.stats import randint, uniform

LIGHTGBM_PARAMS = {
    'boosting_type': ['gbdt', 'dart', 'goss'],
    'num_leaves': randint(20, 100),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.2),
    'n_estimators': randint(50, 250)    
}

RANDOM_SEARCH_PARAMS = {
    'n_iter': 4,
    'cv': 2,
    'verbose': 2,
    'random_state': 42,
    'n_jobs': -1,
    'scoring': 'accuracy'
}