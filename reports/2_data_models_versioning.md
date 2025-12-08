## Версионирование данных с DVC

Устанавливаем и инициализируем DVC

```
poetry add dvc
dvc init
```
![dvc init](figures/dvc%20init.png)

После инициализации DVC создает два файла:
- `.dvcignore`
- директорию `.dvc/`

Создаем remote хранилище для данных, которое будет располагаться локально

```
mkdir -p .dvc/storage
dvc remote add -d localstore .dvc/storage
```
![dvc remote](figures/dvc%20set%20remote.png)

Config обновился

```
[core]
    remote = localstore
['remote "localstore"']
    url = storage
```

Обновляем .gitinore, чтобы он пропускал .dvc файлы

```
!.dvc
!data.dvc
```

И вместо полного игнора директории дата

```
*.csv
*.parquet
*.xlsx
```

Если data/ в .gitignore, то DVC не может отслеживать файлы, которые генерирует там

Добавляем сырые данные для отслеживания

```
dvc add data/raw/WineQT.csv
```

У нас появился файл WineQT.csv.dvc - теперь DVC отслеживает эти данные

![dvc track](figures/dvc%20add.png)

Обновляем .gitignore внутри .dvc, чтобы git не видел наш local remote
Обязательно пушим изменения dvc push

![dvc push](figures/dvc%20push.png)


Автоматизируем версионирование с помощью

```
dvc config core.autostage true
```

config теперь выглядит так:

```
[core]

remote = localstore

autostage = true

['remote "localstore"']

url = storage
```

DVC будет автоматически делать похожие операции

```
git add data/raw.csv.dvc
git add .gitignore
```

Далее создаем тестовый пайплайн обработки данных, где создаем таргет, удаляем ненужные столбцы, получаем data/processed/dataset.csv, добавляем его в dvc

```
python -m epml_da.dataset
dvc add data/processed/dataset.csv
```


Сделаем автоматизированный пайплайн предобработки с нашим dataset.py, используя dvc stage

```
dvc stage add -n preprocess_data \
	-d data/raw/WineQT.csv \
	-o data/processed/processed_data.csv \
	python -m epml_da.dataset
```


Возникла ошибка пересечения, так как датасет был уже добавлен в dvc, поэтому меняем имя датасета, создающегося при предобработке данных на processed_data.csv

![dvc error](figures/dvc%20stage%20error.png)


Получаем dvc.yaml, в котором есть наш пайплайн

```
stages:

preprocess_data:

cmd: python -m epml_da.dataset

deps:

- data/raw/WineQT.csv

outs:

- data/processed/processed_data.csv
```

Запускаем с помощью команды dvc repro

```
dvc repro preprocess_data
```

![dvc repro](figures/dvc%20repro.png)

Получили dvc.lock, который автоматически уже застейджен для гита
Пушим наши изменения в гит и в dvc, настройка завершена

![dvc](figures/dvc%20stage.png)

![dvc lock](figures/dvc%20lock.png)

![dvc hash](figures/dvc%20hash.png)

## Версионирование моделей с MLflow

Устанавливаем Mlflow

```
poetry add mlflow
```

Создаем базовый код для обучения и оценки качества модели на обработанных данных с предыдущего шага train.py и конфиг параметров моделей params.yaml. В этом коде будет создаваться эксперимент wine-quality-exp-1, регистрироваться модель, и логироваться артефакты (модель в формате .pkl и метрики в формате .json)

Запускаем обучение и валидацию

```
python -m epml_da.modeling.train
```

Mlflow создает локальную директорию mlruns/, в которой хранит артефакты и mlflow.db с метаданными.
Также создается директория models/. Их добавляем в .gitignore.

Запускаем обучение и валидацию двух других моделей - lr, mlp. Для этого изменяем параметр model в params.yaml, запускаем заново команду.

Немного изменяем параметры моделей через params.yaml и прогоняем эксперимент заново, чтобы у моделей появилось несколько версий.

![mlflow v](figures/mlflow%20train.png)

Командой запускаем интерфейс, где видно эксперимент и метрики моделей

```
mlflow ui
```

![mlflow ui](figures/mlflow%20experiment.png)
![mlflow m](figures/mlflow%20metrics.png)

Метаданные для моделей настроены.

### Система для отображения версий модели

В Mlflow можно посмотреть метрики разных версий зарегистрированных моделей в разрезе запусков. Для этого нужно перейти в Model Registry, выбрать модель, перейти в нее, выбрать несколько версий и нажать compare.

![mlflow vers](figures/mlflow%20model%20versioning.png)

Также можно создать свою систему сравнения версий.

Создаем modeling/fetch_model_versions.py, где мы ищем по названию зарегистрированной модели ее параметры и метрики, сохраняем в датафрейм. Версия модели будет задаваться в конфиге model_version.yaml.

```
python -m epml_da.modeling.fetch_model_versions
```
Далее создаем modeling/mv_dashboard.py, это скрипт для запуска дашборда на Streamlit, который отображает данные модели.

Для работы нужен streamlit

```
poetry add streamlit
```

Дашборд можно запустить по команде:

```
streamlit run epml_da/modeling/mv_dashboard.py
```
![dashboard](figures/custom%20model%20versioning.png)

## Docker
В докерфайле изменили команду на train, чтобы выполнялся весь пайплайн обучения и валидации модели.

![docker run](figures/docker%20container%20run.png)
