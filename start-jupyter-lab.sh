#!/bin/bash

export GIT_PYTHON_REFRESH=quiet
export PYTHONPATH=$(pwd)
export PATH=$(pwd)/vendor/Modest/linux:/home/vscode/.local/bin:$PATH

jupyter lab --config=config/jupyter_lab_config.py
