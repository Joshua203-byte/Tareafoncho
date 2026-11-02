# Abre un túnel público hacia el servicio local (puerto 8000 por defecto).
#
# Uso (PowerShell, con el servicio ya corriendo: ver scripts/run.ps1):
#     .\scripts\tunnel.ps1                       # túnel principal: ngrok
#     .\scripts\tunnel.ps1 -Tool cloudflare      # respaldo: Cloudflare Quick Tunnel
#     .\scripts\tunnel.ps1 -Tool ngrok -Port 8001
#
# Qué es cada uno:
#   ngrok       Necesita cuenta y authtoken (una sola vez: ngrok config add-authtoken ...).
#               La cuenta gratuita trae un dominio fijo (*.ngrok-free.dev): la URL se repite
#               entre arranques. Es el túnel principal.
#   cloudflare  Cloudflare Quick Tunnel (cloudflared). No necesita cuenta. La URL es
#               aleatoria (*.trycloudflare.com) y CAMBIA en cada arranque, así que es solo
#               el respaldo: si lo usas, hay que mandarle la URL nueva al instructor.
#
# Cada herramienta imprime la URL pública al arrancar. Déjalo corriendo en esta terminal;
# Ctrl+C cierra el túnel. La URL es pública: lo único que protege la API es el token.
#
# Con Cloudflare, el túnel tarda unos 10 a 20 segundos en registrarse después de imprimir la
# URL. Si en ese lapso la abres, sale "Error 1033": espera y vuelve a probar /health.
param(
    [ValidateSet("ngrok", "cloudflare")] [string]$Tool = "ngrok",
    [int]$Port = 8000
)

# 127.0.0.1 y no "localhost": run.ps1 escucha en IPv4 y "localhost" a veces resuelve a IPv6.
$target = "http://127.0.0.1:$Port"

switch ($Tool) {
    "ngrok" {
        if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
            throw "No encuentro ngrok. Instálalo desde la Microsoft Store (ver README)."
        }
        ngrok http $Port
    }
    "cloudflare" {
        # El instalador MSI de cloudflared lo deja aquí y no siempre lo agrega al PATH.
        $candidates = @(
            "C:\Program Files (x86)\cloudflared\cloudflared.exe",
            "C:\Program Files\cloudflared\cloudflared.exe"
        )
        $exe = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $exe) { $exe = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source }
        if (-not $exe) { throw "No encuentro cloudflared. Instálalo con: winget install --id Cloudflare.cloudflared" }
        # --no-autoupdate: que no se actualice solo en medio de la presentación.
        & $exe tunnel --no-autoupdate --url $target
    }
}
