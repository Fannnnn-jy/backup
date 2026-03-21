#!/bin/bash

eval "$(mamba shell hook --shell bash)"
mamba activate actrec
uv pip install -e .

exec "$@"
