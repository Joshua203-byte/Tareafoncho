# Arranca el servicio con la configuración de .env (que NO está en git).
#
# Uso (PowerShell, desde la raíz del repo):
#     .\scripts\run.ps1            # puerto 8000
#     .\scripts\run.ps1 -Port 8001
#
# .env es un archivo de líneas CLAVE=valor (ver .env.example). Este script copia cada
# una al entorno de ESTA sesión de PowerShell, así el token vive solo en el entorno y
# nunca en el código. Escucha solo en 127.0.0.1: el túnel (ngrok) llega desde la misma
# máquina y así la API no queda expuesta a toda la red local.
param([int]$Port = 8000)

$root = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $root ".env"
if (-not (Test-Path $envFile)) {
    throw "Falta .env: copia .env.example a .env y pon tu token de desarrollo."
}

Get-Content $envFile | ForEach-Object {
    # Ignora comentarios y líneas en blanco; el resto es CLAVE=valor.
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
        Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
    }
}
if (-not $env:QA_API_TOKEN) { throw "QA_API_TOKEN está vacío en .env." }

# Usa el entorno virtual del proyecto si existe; si no, el python del PATH.
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }

Set-Location $root
# --workers 1: el modelo vive en un solo proceso (ver service/app.py).
& $python -m uvicorn service.app:app --host 127.0.0.1 --port $Port --workers 1
