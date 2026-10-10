import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import average
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from catboost import CatBoostClassifier, Pool
import optuna
from optuna_integration import CatBoostPruningCallback

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df['mass_per_minute'] = df['model_mass_g'] / df['estimated_time_min']
df_test['mass_per_minute'] = df_test['model_mass_g'] / df_test['estimated_time_min']
df['minute_per_mass'] = df['estimated_time_min'] / df['model_mass_g']
df_test['minute_per_mass'] = df_test['estimated_time_min'] / df_test['model_mass_g'] 

df['area_per_mass'] = df['contact_area_cm2'] / df['model_mass_g']
df['mass_per_area'] = df['model_mass_g'] / df['contact_area_cm2']
df_test['area_per_mass'] = df_test['contact_area_cm2'] / df_test['model_mass_g']
df_test['mass_per_area'] = df_test['model_mass_g'] / df_test['contact_area_cm2']

df['overhang_danger'] = df['overhang_angle_deg'] * df['print_speed_mm_s']
df_test['overhang_danger'] = df_test['overhang_angle_deg'] * df_test['print_speed_mm_s']

df['mass_per_infill'] = df['model_mass_g'] * df['infill_percent']
df_test['mass_per_infill'] = df_test['model_mass_g'] * df_test['infill_percent']

df['mean_cooling_percent'] = df.groupby('material')['cooling_percent'].transform('mean')
df['deviation_cooling_percent'] = df['cooling_percent'] - df['mean_cooling_percent']
df_test['mean_cooling_percent'] = df_test.groupby('material')['cooling_percent'].transform('mean')
df_test['deviation_cooling_percent'] = df_test['cooling_percent'] - df_test['mean_cooling_percent']

df['mean_nozzle_temp'] = df.groupby('material')['nozzle_temp_c'].transform('mean')
df['deviation_nozzle_temp'] = df['nozzle_temp_c'] - df['mean_nozzle_temp']
df_test['mean_nozzle_temp'] = df_test.groupby('material')['nozzle_temp_c'].transform('mean')
df_test['deviation_nozzle_temp'] = df_test['nozzle_temp_c'] - df_test['mean_nozzle_temp']

df['mean_bed_temp'] = df.groupby('material')['bed_temp_c'].transform('mean')
df['deviation_bed_temp'] = df['bed_temp_c'] - df['mean_bed_temp']
df_test['mean_bed_temp'] = df_test.groupby('material')['bed_temp_c'].transform('mean')
df_test['deviation_bed_temp'] = df_test['bed_temp_c'] - df_test['mean_bed_temp']

df['delta_ambient_temp'] = df['bed_temp_c'] - df['ambient_temp_c']
df_test['delta_ambient_temp'] = df_test['bed_temp_c'] - df_test['ambient_temp_c']

df = df.drop(columns=['mean_nozzle_temp', 'mean_bed_temp'])
df_test = df_test.drop(columns=['mean_nozzle_temp', 'mean_bed_temp'])

X = df.drop(columns=['target', 'id'])
Y = df['target']

modes = df['bed_surface'].mode().iloc[0]
X['bed_surface'] = X['bed_surface'].fillna(modes)
df['bed_surface'] = df['bed_surface'].fillna(modes)
df_test['bed_surface'] = df_test['bed_surface'].fillna(modes)

cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(cat_features)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=54)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)
test_pool = Pool(df_test, cat_features=cat_features)

def objective(trial):
    param_optuna = {
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 20),
        'random_strength': trial.suggest_float('random_strength', 0.0, 2.0),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count': trial.suggest_int('border_count', 128, 254),
        'loss_function': 'Logloss',
        'random_seed': 54,
        'verbose': 1,
        'task_type': 'CPU'
    }

    pruning_callback = CatBoostPruningCallback(trial, 'TotalF1:average=Macro')
    model = CatBoostClassifier(**param_optuna, eval_metric='TotalF1:average=Macro')
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=50,
        callbacks=[pruning_callback],
        verbose=1,
    )
    preds = model.predict(X_val)
    preds = np.squeeze(preds)
    score = f1_score(y_val, preds, average='macro')
    return score

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=100)
)

study.optimize(objective, n_trials=30, timeout=1200)

print(f"Лучшие параметры: {study.best_params}")
print(f"Лучший Macro F1: {study.best_value}")

model = CatBoostClassifier(
    **study.best_params,
    iterations=2000,
    random_seed=54,
    early_stopping_rounds=50,
    loss_function='Logloss',
    verbose=100,
    task_type='CPU',
)

model.fit(
    train_pool,
    eval_set=val_pool
)

preds = model.predict(test_pool)
submission = pd.DataFrame({
    'id': df_test['id'],
    'target': preds,
})
submission.to_csv('submissions.csv', index=False)