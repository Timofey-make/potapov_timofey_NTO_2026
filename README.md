<h1>Зависимости:</h1>
<ul>
    <li>catboost==1.2.10</li>
    <li>scikit-learn==1.9.1</li>
    <li>pandas==3.0.6</li>
    <li>python==3.10</li>
    <li>numpy==2.4.2</li>
    <li>optuna==5.0.0</li>
    <li>optuna_integration==5.0.0</li>
</ul>
<h1>Инструкция по запуску:</h1>

- В директории должны быть файлы `test.csv`, `train.csv`, `main.py`</li>
- Установите нужные библиотеки
```bash
    pip install catboost
    pip install scikit-learn
    pip install optuna
    pip install optuna_integration
    pip install pandas
    pip install numpy
```
- Запустите `main.py`
``` bash
    python3 main.py
```