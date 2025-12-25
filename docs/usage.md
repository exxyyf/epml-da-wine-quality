## Train model

```bash
python -m epml_da.modeling.train --model-name rf
```
## Система сравнения экспериментов

Для анализа экспериментов реализована утилита compare_runs.py, которая:

- выбирает top-N запусков по метрике

- поддерживает фильтрацию по параметрам

- экспортирует результаты в CSV

Пример использования:

```bash
python compare_runs.py \
  --experiment wine-quality-exp-baseline-models \
  --metric f1 \
  --top-n 5
```

## Визуализация результатов

С помощью mv_dashboard.py реализован dashboard на Streamlit:

- таблица версий моделей

- графики метрик по версиям

- просмотр параметров моделей

Запуск:

```bash
streamlit run epml_da/modeling/mv_dashboard.py
```
