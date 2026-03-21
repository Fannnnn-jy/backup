# ActRec

## Installation

Normal Installation:
```
conda create -n actrec-new python=3.11 -y && conda activate actrec-new
pip install -e .
```

Docker Installation:
```
printf "DOCKER_UID=$(id -u $USER)\nDOCKER_GID=$(id -g $USER)\nDOCKER_USER=$USER\n" > .env
docker compose up --build -d && docker exec -it metasim bash
```

For developer, please
1) Install pre-commit hooks:
    ```
    pip install pre-commit
    pre-commit install
    ```

2) Setup vscode settings:
    ```
    cp .vscode/template/settings.jsonc .vscode/settings.json
    ```
    At the same time, make sure you've installed the `ruff` extension in vscode.

## Usage

To test sapien render system:
```
python actrec/scripts/dev_ms_parallel_freecam.py
```
