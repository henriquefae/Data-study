from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path('.').resolve()))
from inverse_modeling import inverse_yield, estimate_empirical_reference

def regression_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {'rmse': rmse, 'mae': float(mean_absolute_error(y_true, y_pred)), 'r2': float(r2_score(y_true, y_pred))}

def kernel_from_tuning_row(row):
    optimized = str(row['optimized_kernel'])
    constant_match = re.search(r'([0-9.]+)\*\*2', optimized)
    length_scale_match = re.search(r'length_scale=([0-9.]+)', optimized)
    noise_match = re.search(r'noise_level=([0-9.]+)', optimized)
    constant_value = float(constant_match.group(1)) ** 2
    length_scale = float(length_scale_match.group(1))
    noise_level = float(noise_match.group(1))
    if row['kernel_name'] == 'rbf':
        base_kernel = RBF(length_scale=length_scale, length_scale_bounds='fixed')
    elif row['kernel_name'] == 'matern_1p5':
        base_kernel = Matern(length_scale=length_scale, nu=1.5, length_scale_bounds='fixed')
    elif row['kernel_name'] == 'matern_2p5':
        base_kernel = Matern(length_scale=length_scale, nu=2.5, length_scale_bounds='fixed')
    else:
        raise ValueError(row['kernel_name'])
    return ConstantKernel(constant_value=constant_value, constant_value_bounds='fixed') * base_kernel + WhiteKernel(noise_level=noise_level, noise_level_bounds='fixed')

DATA_PATH = Path('prepared/yields_weather_by_crop.csv')
GP_TUNING_PATH = Path('prepared/gp_hyperparameter_tuning_results.csv')
CROP = 'RED SPRING WHEAT'
TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
climate_feature_cols = [c for c in df.columns if re.match(r'^\d{2}_Q[12]_', c)]
fixed_crop_df = df.loc[df['Crop'] == CROP, ['Mean_Yield', 'Crop'] + climate_feature_cols].dropna().reset_index(drop=True)
X = fixed_crop_df[climate_feature_cols].copy()
y = fixed_crop_df['Mean_Yield'].to_numpy(dtype=float)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

gp_tuning_results = pd.read_csv(GP_TUNING_PATH).sort_values(['val_rmse', 'val_mae', 'val_r2'], ascending=[True, True, False]).reset_index(drop=True)
best_gp_row = gp_tuning_results.iloc[0].copy()

print('Best GP row:')
print(best_gp_row.to_dict())

gp_kernel = kernel_from_tuning_row(best_gp_row)
gp_model = GaussianProcessRegressor(kernel=gp_kernel, alpha=float(best_gp_row['alpha']), normalize_y=True, optimizer=None, random_state=RANDOM_STATE)
gp_model.fit(X_train, y_train)

y_train_pred = gp_model.predict(X_train)
y_test_pred = gp_model.predict(X_test)
metrics = pd.DataFrame([{'split': 'train', **regression_metrics(y_train, y_train_pred)}, {'split': 'test', **regression_metrics(y_test, y_test_pred)}])
print('\nMetrics:')
print(metrics.to_string(index=False))

permutation = permutation_importance(gp_model, X_test, y_test, n_repeats=20, random_state=RANDOM_STATE, scoring='neg_root_mean_squared_error')
importance_df = pd.DataFrame({'feature': climate_feature_cols, 'importance_mean': permutation.importances_mean, 'importance_std': permutation.importances_std}).sort_values('importance_mean', ascending=False).reset_index(drop=True)
print('\nTop 15 features:')
print(importance_df.head(15).to_string(index=False))

reference = estimate_empirical_reference(X_train)
reference_mean = reference['mean']
reference_cov = reference['cov']
lower = np.quantile(X_train, 0.01, axis=0)
upper = np.quantile(X_train, 0.99, axis=0)
bounds = list(zip(lower, upper))
train_quantiles = pd.Series(y_train).quantile([0.25,0.5,0.75])
targets = {'low': float(train_quantiles.loc[0.25]), 'median': float(train_quantiles.loc[0.50]), 'high': float(train_quantiles.loc[0.75])}

for label, y_target in targets.items():
    result = inverse_yield(model=gp_model, y_target=y_target, feature_names=climate_feature_cols, bounds=bounds, reference_mean=reference_mean, reference_cov=reference_cov, lambda_mahal=0.0, lambda_l2=0.001, n_restarts=8, random_state=RANDOM_STATE)
    print(f"\nScenario {label}: success={result['success']} target={result['y_target']:.4f} pred={result['y_pred']:.4f} abs_error={abs(result['y_pred']-result['y_target']):.4f}")
    if result['x_opt'] is not None:
        raw_solution = scaler.inverse_transform(result['x_opt'].reshape(1,-1)).ravel()
        raw_ref = scaler.inverse_transform(reference_mean.reshape(1,-1)).ravel()
        df_out = pd.DataFrame({'feature': climate_feature_cols, 'raw_delta': raw_solution-raw_ref, 'abs_delta': np.abs(raw_solution-raw_ref)})
        print(df_out.sort_values('abs_delta', ascending=False).head(10).to_string(index=False))
