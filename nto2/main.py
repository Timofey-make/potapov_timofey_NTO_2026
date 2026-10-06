import pandas as pd
from sklearn.model_selection import train_test_split
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

model = CatBoostClassifier(
    iterations=600,
    learning_rate=0.03,
    depth=6,
    eval_metric='F1',
    random_seed=54,
    early_stopping_rounds=100,
    task_type='GPU',
)

model.fit(
    train_pool,
    eval_set=val_pool,
    verbose=100
)

preds = model.predict(test_pool)
submission = pd.DataFrame({
    'id': df_test['id'],
    'target': preds,
})
submission.to_csv('submissions.csv', index=False)


