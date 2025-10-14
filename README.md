Instalar pyenv en windows:

https://github.com/pyenv-win/pyenv-win

Abrir powershell y ejecutar:

Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

Cerrar powershell y volver a abrirla y ejecutar esto:

pyenv --version

pyenv install 3.11.8

pyenv global 3.11.8

Parado en la raiz del proyecto ejecutar:

En windows:

python -m venv .venv

Para activarlo ejecutar esto en powershell:

.\.venv\Scripts\activate.ps1

En mac:

python3 -m venv .venv

source .venv/bin/activate

which python

pip install -r requirements.txt

Chusmear esto de Open3D:

https://realsenseai.com/news-insights/insights/open3d/
