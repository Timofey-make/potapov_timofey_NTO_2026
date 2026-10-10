import pandas as pd
from scipy.cluster.hierarchy import average
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from catboost import CatBoostClassifier, Pool

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

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

cat_features = X.select_dtypes(include=['str', 'category']).columns.tolist()
print(cat_features)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=54)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)
test_pool = Pool(df_test, cat_features=cat_features)

macro_f1_scorer = make_scorer(f1_score, average='macro')
model = CatBoostClassifier(
    verbose=0,
    random_seed=54,
    loss_function='Logloss'
)

param_grid = {
    'iterations': [2000],
    'learning_rate': [0.03, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1,5,10,20],
    'random_strength': [0.0, 1.0, 2.0],
    'bagging_temperature': [0.0, 0.5, 1.0],
    'border_count': [128, 254],
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=2,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1,
)

grid_search.fit=cat_features, (X_train, y_train, cat_featuresearly_stopping_rounds=100)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшая точность: {grid_search.best_score_}")

model = CatBoostClassifier(
    **grid_search.best_params_,
    random_seed=54,
    early_stopping_rounds=50,
    loss_function='Logloss',
    verbose=1,
    task_type='GPU',
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