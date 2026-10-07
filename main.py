import pandas as pd
from scipy.cluster.hierarchy import average
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from catboost import CatBoostClassifier, Pool

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

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
    'iterations': [600, 1000, 1500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=2,
    scoring=macro_f1_scorer,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train, cat_features=cat_features)

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


