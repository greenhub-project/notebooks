# GreenHub's data pipeline notebooks

> A collection of notebooks and helper scripts to run statistical analysis of GreenHub's data repository

This repository uses [Pipenv](https://docs.pipenv.org/) to manage its dependencies and virtualenv.

## Prerequisites
```shell
$ pip install --user pipenv
```

## Install
```shell
$ cd repo_folder
$ pipenv install
$ pipenv shell # activate virtualenv
```

### Minimal install
Alternatively, you can install only the bare minimum dependencies on your local machine. This will only install the requirements for the `src/` files.
```shell
$ pip install --user -r src/requirements.txt
```

## Editor configuration
Don't forget to setup the virtualenv path for `python` and `pylint` on your editor.

### Visual Studio Code
Example `settings.json` for Visual Studio Code running on Unix-like OS:
```json
{
  "python.pythonPath": "/home/john/.local/share/virtualenvs/notebooks-44m-0JdC/bin/python",
  "python.linting.pylintPath": "/home/john/.local/share/virtualenvs/notebooks-44m-0JdC/bin/pylint"
}
```