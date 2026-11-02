# Tareafoncho — QA extractivo como servicio (equipo Fonchos)

Fine-tuning de `distilbert-base-uncased` para QA extractivo (SQuAD 2.0) y una API FastAPI
con token, expuesta por un túnel público. El reporte está en [REPORT.md](REPORT.md).

## Estructura

| Carpeta | Contenido |
|---|---|
| `src/` | Entrenamiento: datos, ventanas, modelo, métricas, post-procesamiento |
| `service/` | La API: `config.py` (entorno), `qa.py` (predictor), `app.py` (endpoints) |
| `scripts/` | Arranque: `run.ps1` (servicio) y `tunnel.ps1` (túnel). Medición: `smoke_test.py`, `robustness_check.py`, `bench_predict.py`, `eval_service_sample.py`. Gráficas del reporte: `plot_training.py` |
| `tests/` | Pruebas con `pytest` |
| `logs/` | Registros de entrenamiento, de la evaluación final y del servicio |
| `docs/` | Explicación de cada fase |

## Cómo correr el servicio desde cero

1. Instala las dependencias (versiones fijadas en `requirements.txt`):
   ```bash
   pip install -r requirements.txt
   ```
2. Consigue el modelo. **Los pesos no están en git**: bájalos del
   [release `weights-v1`](https://github.com/Joshua203-byte/Tareafoncho/releases/tag/weights-v1)
   (`fonchos-qa-weights-v1.zip`, 245 MB) y descomprímelos **en la raíz del repo**
   (`Expand-Archive fonchos-qa-weights-v1.zip -DestinationPath .`). Deja pesos y tokenizer en
   `models/full_ft_4ep/epoch_3`, la ruta por defecto: no hace falta `QA_TOKENIZER_DIR`.
3. Copia `.env.example` a `.env` y pon tu token en `QA_API_TOKEN`. `.env` está en
   `.gitignore`: el token nunca se versiona.
4. Arranca (PowerShell):
   ```powershell
   .\scripts\run.ps1
   ```
   o a mano:
   ```bash
   export QA_API_TOKEN=...                       # PowerShell: $env:QA_API_TOKEN="..."
   export QA_MODEL_DIR=models/full_ft_4ep/epoch_3
   export QA_TOKENIZER_DIR=models/full_ft_4ep/merged
   uvicorn service.app:app --host 127.0.0.1 --port 8000 --workers 1
   ```
5. Comprueba: `curl http://localhost:8000/health` debe dar `{"status":"ok"}`.

Variables: `QA_API_TOKEN` (obligatoria), `QA_MODEL_DIR`, `QA_TOKENIZER_DIR`, `QA_TAU`
(por defecto −2.0) y `QA_TEAM` (por defecto `Fonchos`). Detalle en `service/config.py`.

## Túnel público

La API corre en tu laptop y el túnel le da una URL pública HTTPS. Con el servicio corriendo
en el puerto 8000, en otra terminal:

```powershell
.\scripts\tunnel.ps1                    # principal: ngrok
.\scripts\tunnel.ps1 -Tool cloudflare   # respaldo: Cloudflare Quick Tunnel
```

| | ngrok (principal) | Cloudflare Quick Tunnel (respaldo) |
|---|---|---|
| Cuenta | Sí, gratuita. Una vez: `ngrok config add-authtoken <authtoken>` | No |
| URL | Dominio fijo `*.ngrok-free.dev`: se repite entre arranques | Aleatoria `*.trycloudflare.com`: **cambia en cada arranque** |
| Instalación | Microsoft Store (la versión de winget es vieja y la cuenta exige ≥ 3.28) | `winget install --id Cloudflare.cloudflared` |
| Comando que corre | `ngrok http 8000` | `cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000` |
| Detalle | El navegador ve una página de advertencia; el harness no | Tarda 10–20 s en registrarse: antes sale "Error 1033" |

Cada herramienta imprime la URL pública al arrancar. Mándasela al instructor antes de tu turno,
y de nuevo si cambia (siempre con Cloudflare). Los dos túneles pasan los mismos 5 `curl` de la
guía, el smoke test y la batería de robustez (`logs/service/*_ngrok.txt` y `*_cloudflare.txt`).

**Antes de tu turno:** laptop enchufada y sin suspensión (`powercfg /change standby-timeout-ac 0`),
servicio con `run.ps1`, túnel con `tunnel.ps1`, y `<url>/health` debe dar `{"status":"ok"}`.
El token del día de la presentación se pone en `.env` (`QA_API_TOKEN`) y se reinicia el servicio.

Qué es un túnel, cómo funcionan ngrok y Cloudflare, sus límites, qué medimos y qué hacer si falla
uno: [docs/fase6_tunel.md](docs/fase6_tunel.md).

## Pruebas y mediciones

```bash
python -m pytest -q                                                   # con QA_MODEL_DIR definido, corre todo
python scripts/smoke_test.py --base-url https://<url-publica> --n 50  # imita al harness (token en QA_API_TOKEN)
python scripts/robustness_check.py https://<url-publica>             # 42 casos raros: unicode, tamaños, cabeceras
python scripts/bench_predict.py --repeats 5                           # latencia del predictor en CPU
python scripts/eval_service_sample.py --n 1000                        # servicio vs. evaluación (dev_select)
python scripts/plot_training.py                                       # regenera logs/loss.png, dev_select.png y tau_sweep.png
```

Las gráficas salen de los registros que ya están en `logs/runs/` (no se entrena nada para generarlas).

Sin los pesos, las pruebas que usan el modelo real se saltan solas con el motivo visible.
