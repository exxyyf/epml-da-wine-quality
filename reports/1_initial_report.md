Отчет о первой проделанной работе по проекту.
Для работы над проектом был выбран датасет Wine Quality.

## Настройка базовой структуры проекта

### Установка python и шаблона CCDS

С помощью miniconda было сделано новое окружение, туда установлен **python 3.11**, загружен шаблон из cookiecutter-data-science и создана базовая структура проекта

```
conda create -n epml-da python=3.11
conda activate epml-da

pip install cookiecutter-data-science
ccds
```

### Работа с Poetry

Далее как основной менеджер зависимостей был установлен **Poetry** (version 2.2.1)

```
curl -sSL https://install.python-poetry.org | python3 -
```

**Poetry** уже использован в шаблоне ccds, поэтому там сразу был файл pyproject.toml, с частично предустановленными пакетами для DS. Отдельно был установлен seaborn.

Все пакеты устанавливаются через данную команду

```
poetry add {package_name}
```

Важно -- miniconda НЕ используется для установки пакетов, всё должно проходить только через **Poetry**

```
# С помощью этой команды выводим основную информацию про наше окружение
poetry env info
```

![poetry env info](figures/poetry%20env%20info.png)

## Настройка Git репозитория

На удаленном хосте github.com был создан репозиторий проекта
https://github.com/exxyyf/epml-da-wine-quality

Далее локальный репозиторий был связан с удаленным, сделан initial commit с базовой структурой всех файлов.

Удобно, что ccds предоставляет уже готовый .gitignore файл, учитывающий особенности операционной системы.

```
git init
git remote add origin git@github.com:exxyyf/epml-da-wine-quality.git
git add .
git push -u origin main
```

## Настройка pre-commit hooks

Установили через Poetry

```
poetry add pre-commit
pre-commit install
```

Создали специальный файл конфигуратор, где указали black, ruff. Затем еще добавили базовые проверки (заключительные строки, пробелы)

```yaml
# .pre-commit-config.yaml

repos:

- repo: https://github.com/pre-commit/pre-commit-hooks

rev: v4.6.0

hooks:

- id: trailing-whitespace

- id: end-of-file-fixer

- id: check-yaml

- id: check-added-large-files

- repo: https://github.com/psf/black

rev: 24.3.0

hooks:

- id: black

- repo: https://github.com/astral-sh/ruff-pre-commit

rev: v0.4.4

hooks:

- id: ruff

args: ["--fix"]
```

Протестировали на всех файлах проекта. Так как ccds уже добавил некоторый базовый код с заглушками, проверки были проведены на .py файлах.

```
pre-commit run --all-files
```

![pre commit hooks in action](figures/pre%20commit%20hooks%20in%20action.png)

## Создание Dockerfile для контейнеризации

1. Для начала была создана отдельная ветка для подбора конфигурации Dockerfile -- dev-docker
2. Далее был создан файл .dockerignore, в который включены файлы ноутбуков, окружений, переменных и тд.
3. После этого создан файл Dockerignore.

```yaml
# -----------------------------

# 1. Base image

# -----------------------------

FROM python:3.11-slim AS base



ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1



# -----------------------------

# 2. System deps

# -----------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \

curl \

build-essential \

git \

&& rm -rf /var/lib/apt/lists/*



# -----------------------------

# 3. Install Poetry

# I have matched this to my project Poetry version

# -----------------------------

ENV POETRY_VERSION=2.2.1

RUN curl -sSL https://install.python-poetry.org | python3 -



# Poetry will be installed into /root/.local/bin

ENV PATH="/root/.local/bin:$PATH"



# -----------------------------

# 4. Disable Poetry virtualenvs

# Dependencies installed directly into system Python of the container

# -----------------------------

RUN poetry config virtualenvs.create false



# -----------------------------

# 5. Set workdir

# -----------------------------

WORKDIR /app



# -----------------------------

# 6. Copy only dependency files first

# -----------------------------

COPY pyproject.toml poetry.lock* ./

COPY LICENSE ./

COPY README.md ./



# -----------------------------

# 7. Copy project code

# -----------------------------

COPY epml_da ./epml_da

COPY models ./models



# -----------------------------

# 8. Install dependencies

# -----------------------------

RUN poetry install --no-interaction --no-ansi




# I am not sure what to do with data, might need to add processed later

# COPY data/processed ./data/processed



# -----------------------------

# 9. Default command is calling model to predict

# -----------------------------

CMD ["python", "-m", "epml_da.modeling.predict"]
```

- При создании Dockerfile я руководствовалась тем, что мой проект создан под python 3.11.

- Далее я установила curl, чтобы установить Poetry, и некоторые другие помощники.

- Указала, чтобы Poetry не делал виртуальных окружений других, так как Docker - это уже по сути своей отдельная изолированная система.

- Далее я скопировала файлы лицензии и ридми - не думала, что их нужно копировать, но они зафиксированы в pyproject.toml, без них не собирался образ, поэтому я их тоже указала.

- Следующим шагом скопировала код проекта (самого модуля) и модели.

- Установила все пакеты через Poetry

- Для основной команды при запуске моего контейнера выбрала запуск предсказания модели. Кажется, это адекватный смысл использования контейнеризированного проекта. Были мысли еще запустить юпитерлаб, но остановилась на этом.

- Возник вопрос, что делать с данными для предсказания. По идее, они должны быть в data/processed, и наверно их тоже нужно включать в образ? Либо как-то еще добавлять, указывать путь, делать для этого команду в CLI. Честно, здесь затрудняюсь.

Итог:

**образ успешно собрался, Dockerfile вышел рабочий, и он даже запустился!** То есть, команда RUN успешно прогнала предикт модели с базовым кодом из cookiecutter.

![docker image built](figures/docker%20image.png)

![docker image build interface](figures/docker%20image%20success.png)

![docker run success](figures/docker%20run%20success.png)

## Настройка веток в Git

За основу логики ветвления в данном проекте был взят [The Data Science Lifecycle Process](https://github.com/dslp/dslp/blob/main/branching/branch-types.md)
Ветка develop не была создана, так как проект небольшой, все PR будут делаться в main.
Следующие типы веток будут использоваться:

1. **Data branches** - для обработки данных
2. **Explore branches** - для исследования данных
3. **Experiment branches** - для экспериментов с моделями
4. **Model branches** - для финальных моделей
5. **Feature branches** - для того, чтобы добавлять репорты по заданиям

плюс еще одна ветка **dev-docker** для того, чтобы контейнеризировать проект

Создавать ветки я буду при выполнении каких-либо задач - например, если буду делать эксперимент с данными, то сделаю
```
git checkout -b "experiment/[number]-basic-data-eda"
```

Сейчас создам только ветку feature с первым отчётом по проделанной работе

```
git checkout -b "feature/1-initial-report"
```
