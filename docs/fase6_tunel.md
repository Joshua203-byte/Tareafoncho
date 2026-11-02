# Fase 6: túnel público (ngrok y Cloudflare Quick Tunnel)

Este documento explica qué es un túnel, por qué lo necesitamos, cómo funciona cada uno de los dos que dejamos listos, qué límites tienen, qué probamos y qué hacer el día de la presentación. Las decisiones que dependen de una medición citan el archivo de `logs/service/` que la respalda.

## 1. Para qué sirve el túnel

La API corre en un laptop. Ese laptop está detrás del router de la casa o de la universidad: no tiene una dirección pública a la que el instructor pueda llamar, y abrir puertos del router no es viable (no lo controlamos, y la red de clase suele bloquearlo). La guía (sección IX) lo resuelve con un **túnel inverso**: un programa en el laptop que abre una conexión hacia afuera y recibe por ella las peticiones que llegan a una URL pública HTTPS.

```text
harness del instructor ──HTTPS──▶ URL pública ──▶ red del proveedor del túnel
                                                        ▲
                       conexión que el laptop abrió     │  (saliente)
                                                        │
                       túnel (ngrok / cloudflared) ◀────┘
                              │
                              ▼
                  http://127.0.0.1:8000  ──▶  nuestra API (FastAPI)
```

Consecuencias que importan:

- **La conexión la abre el laptop, hacia afuera.** Por eso funciona detrás de un router sin configurar nada, y por eso no depende del Wi-Fi de la clase: el laptop solo necesita internet.
- **El proveedor del túnel ve el tráfico** (termina el HTTPS y lo reenvía). Para este proyecto eso es aceptable, porque los datos son textos de prueba, pero es una razón para no mandar por ahí nada sensible.
- **La URL es pública**: cualquiera en internet que la conozca puede llamarla. Lo único que protege la API es el token (paso de autenticación en `service/app.py`). Por eso la URL no se versiona y los logs la omiten.
- **El tiempo de ida y vuelta crece.** Cada petición pasa por el proveedor. Lo medimos (sección 5).
- **Si el laptop se apaga, se suspende o pierde internet, el túnel cae** y la API deja de ser alcanzable. La guía lo pide explícito: laptop enchufada, sin suspensión, servicio y túnel corriendo toda la sesión.

## 2. Las opciones que permite la guía

La guía (sección IX) acepta tres herramientas y pide instalar y probar una segunda como respaldo:

| Herramienta | Cuenta | URL pública | Notas de la guía |
|---|---|---|---|
| ngrok | Gratuita | `https://<nombre>.ngrok-free.app` o `.dev`, un dominio fijo por cuenta | Plan gratuito: 20 000 peticiones y 1 GB al mes. Los navegadores ven una página de advertencia; el harness no. |
| Cloudflare Quick Tunnel | Ninguna | `https://<aleatorio>.trycloudflare.com` | La URL cambia cada vez que se reinicia el túnel. Solo para pruebas, sin garantía de disponibilidad. |
| Tailscale Funnel | Gratuita | `https://<equipo>.<tailnet>.ts.net` | Pide MagicDNS y HTTPS en la tailnet. Una URL nueva puede tardar hasta 10 minutos. Está en beta. |

**Lo que elegimos:** ngrok como principal y Cloudflare como respaldo. Tailscale Funnel no lo probamos.

- **ngrok principal** porque el dominio es **fijo**: la URL que le mandamos al instructor no cambia entre reinicios.
- **Cloudflare respaldo** porque no necesita cuenta: si algo falla con ngrok (cuenta, cuota, caída del servicio), se levanta en un minuto sin depender de ningún acceso.

## 3. Cloudflare Quick Tunnel a fondo

### 3.1 Qué es

Un *Quick Tunnel* (TryCloudflare) es un túnel temporal que se abre con un solo comando, sin cuenta ni dominio, y recibe una URL `*.trycloudflare.com` aleatoria. Lo hace el programa `cloudflared`. Según la documentación de Cloudflare es "para pruebas y desarrollo".

```powershell
cloudflared tunnel --url http://127.0.0.1:8000
```

`cloudflared` abre conexiones salientes hacia la red de Cloudflare y reenvía lo que llega a esa URL hacia `127.0.0.1:8000`. En nuestro log se ve el registro de la conexión (`Registered tunnel connection ... location=mia01 protocol=quic`): se conectó a un punto de presencia de Cloudflare en Miami usando QUIC.

### 3.2 Cómo lo instalamos

```powershell
winget install --id Cloudflare.cloudflared
```

- Descarga el instalador MSI de los lanzamientos oficiales de `cloudflare/cloudflared` en GitHub y **verifica su hash** antes de instalar.
- Lo deja en `C:\Program Files (x86)\cloudflared\cloudflared.exe`. **No siempre lo agrega al PATH**, por eso `scripts/tunnel.ps1` busca esa ruta.
- Versión probada: `cloudflared 2026.10.7`. Comprobamos que el ejecutable tiene **firma Authenticode válida de "Cloudflare, Inc."**.

### 3.3 Cómo lo arrancamos

```powershell
.\scripts\tunnel.ps1 -Tool cloudflare
```

Equivale a `cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000`.

- **`--url http://127.0.0.1:8000`** y no `localhost`: nuestra API escucha en IPv4 (`run.ps1` usa `127.0.0.1`) y `localhost` a veces resuelve a IPv6.
- **`--no-autoupdate`**: que `cloudflared` no se actualice solo a mitad de la presentación.
- Imprime la URL `https://<palabras-aleatorias>.trycloudflare.com`. Hay que dejar esa terminal abierta; Ctrl+C cierra el túnel.

### 3.4 Qué pasa en los primeros segundos (Error 1033)

Después de imprimir la URL, el túnel tarda unos **10 a 20 segundos** en registrarse. Si se abre antes, Cloudflare responde una página HTML con **"Error 1033, Cloudflare Tunnel error"**: la URL ya existe pero todavía no hay un túnel conectado detrás. Lo vimos en la prueba. Es transitorio; basta con reintentar `/health` hasta que responda `{"status":"ok"}`. Un cliente que reciba esa página HTML en vez de JSON contaría el ítem como fallido, así que **no se manda la URL al instructor hasta que `/health` responda bien**.

En el log de `cloudflared` también aparece un error no fatal (`Failed to fetch features ... dnsquery: timeout`): es una consulta de configuración opcional que falló por DNS. El túnel funcionó igual.

### 3.5 Límites, según la documentación de Cloudflare

| Límite | Qué dice | Qué significa para nosotros |
|---|---|---|
| 200 peticiones en vuelo | Más de 200 simultáneas reciben `429` | El harness manda **una por una**: hay 1 en vuelo. Sin problema. |
| Sin Server-Sent Events | No soporta SSE | No usamos SSE. |
| Sin garantía de disponibilidad | "No uptime guarantee" | Por eso es **respaldo** y no principal. |
| No es para producción | Cloudflare recomienda un túnel con cuenta para tráfico real | Es una presentación de una sesión, no producción. |
| El hostname cambia | Cada túnel nuevo tiene otro | **Hay que mandarle al instructor la URL nueva cada vez** que se reinicie. |

### 3.6 Comparación directa con ngrok

| | ngrok (principal) | Cloudflare Quick Tunnel (respaldo) |
|---|---|---|
| Cuenta | Sí, gratuita; hay que guardar un authtoken una vez | No |
| URL | Fija entre arranques (`*.ngrok-free.dev`) | Nueva en cada arranque (`*.trycloudflare.com`) |
| Instalación | Microsoft Store (ver sección 6) | `winget install --id Cloudflare.cloudflared` |
| Tiempo hasta funcionar | Inmediato | 10–20 s de registro |
| Cuota | 20 000 peticiones y 1 GB al mes (plan gratuito) | Sin cuota documentada; límite de 200 en vuelo |
| Página para navegadores | Muestra una advertencia; el harness no se afecta | No muestra advertencia |
| Disponibilidad | Servicio con cuenta | Sin garantía |
| Pasa nuestras pruebas | Sí | Sí |

## 4. Cómo se usa el día de la prueba

1. Laptop **enchufada**, sin suspensión (`powercfg /change standby-timeout-ac 0`).
2. Terminal 1: `.\scripts\run.ps1` (servicio). Con el token del día en `.env`.
3. Terminal 2: `.\scripts\tunnel.ps1` (ngrok, principal).
4. Comprobar `https://<url>/health` → `{"status":"ok"}`.
5. Mandar la URL al instructor, **antes de tu turno**.
6. Si el túnel principal falla: cerrar la terminal 2, correr `.\scripts\tunnel.ps1 -Tool cloudflare`, **esperar a que `/health` responda**, y mandar la URL **nueva**.

**Qué hacer según el síntoma:**

| Síntoma | Causa probable | Acción |
|---|---|---|
| ngrok responde `ERR_NGROK_121` | Agente demasiado viejo para la cuenta | Actualizarlo desde la Microsoft Store |
| ngrok falla por cuenta o cuota | Servicio externo | Pasar a Cloudflare |
| Cloudflare muestra "Error 1033" | El túnel todavía se registra | Esperar 10–20 s y reintentar |
| `/health` da 503 `loading` | El modelo aún carga | Esperar unos segundos; si no termina, revisar la terminal del servicio |
| Todo da 401 | El token del servicio no es el que manda el harness | Poner el token correcto en `.env` y reiniciar el servicio |
| Ninguna URL responde | Laptop dormida o sin internet | Despertar el laptop, revisar la red y reiniciar el túnel |

## 5. Qué medimos (evidencia)

Todo se corrió con el modelo final (fine-tuning completo, época 3, τ = −2), servicio y cliente en el mismo laptop.

| Prueba | Local (sin túnel) | ngrok | Cloudflare |
|---|---|---|---|
| Los 5 `curl` de la guía | — | pasan | pasan |
| Smoke test, 50 peticiones secuenciales | 5.9 s, la más lenta 400 ms | 13.6 s, la más lenta 565 ms | 13.9 s, la más lenta 608 ms |
| Batería de robustez, 42 casos | — | 42/42 | 42/42 |

- **El túnel suma unos 150 ms por petición** (≈ 8 s de diferencia en 50 peticiones) con cualquiera de los dos. Contra el límite de 5 s por petición y el tope de 150 s hay un margen amplio: usamos cerca del 9 % del tiempo total.
- Además, el smoke test corrido desde **otra red** (laptop en el hotspot del teléfono) contra el túnel de ngrok dio 50 de 50 en 16.7 s, la más lenta 703 ms. Esa corrida la hizo el equipo y no quedó en un archivo de `logs/`.
- La prueba de Cloudflare **desde otra red** está pendiente.

Archivos: `logs/service/smoke_test_local.txt`, `smoke_test_ngrok.txt`, `smoke_test_cloudflare.txt`, `robustness_ngrok.txt` y `robustness_cloudflare.txt`. Cada uno trae el commit del código, las versiones y el equipo, y omite la URL.

## 6. Problemas que aparecieron y cómo se resolvieron

1. **ngrok `ERR_NGROK_121` (agente demasiado viejo).** winget instaló la 3.3.1 y la cuenta exige 3.28.0 o más. `ngrok update` fallaba por DNS. *Solución:* instalar desde la Microsoft Store (3.39.11), que se actualiza sola.
2. **El navegador marcó "Virus detected" en el ZIP de ngrok.** No pudimos confirmar si era un falso positivo (los túneles son herramientas que también usan atacantes), así que **no se forzó la descarga** y se usó el instalador de la Store, que es el canal que ngrok recomienda como el más seguro.
3. **El servicio se quedaba en 503 "loading" con `run.ps1`.** El `.env` tenía rutas al estilo de Git Bash (`/c/Users/...`) que Windows PowerShell no entiende. *Solución:* rutas de Windows (`C:/Users/...`). Un fallo al cargar el modelo deja el servicio en `loading` a propósito (no responde con un modelo roto) y registra el error en la terminal.
4. **Cloudflare "Error 1033" al principio.** El túnel aún no estaba registrado (sección 3.4).
5. **`README.md` estaba en UTF-16** desde el primer commit y mi escritura lo conservó; se pasó a UTF-8. Los `.ps1` se guardan con BOM UTF-8 para que Windows PowerShell 5.1 lea bien los acentos.

## 7. Limitaciones y pendientes

- **Cloudflare desde otra red:** todavía no se probó; las pruebas del respaldo salieron del mismo laptop.
- **Tailscale Funnel:** no se instaló ni probó (la guía exige uno de respaldo, no los tres).
- **Token de la presentación:** hoy el servicio corre con un token de desarrollo propio. El del día lo da el instructor y se pone en `QA_API_TOKEN` antes de empezar.
- **Dry run:** lo corre el instructor, con su harness, por nuestra URL (guía, sección XI). Nuestras pruebas lo ensayan, pero no lo reemplazan.
- **Disponibilidad del laptop:** el túnel depende de que esté encendido, enchufado y con internet. No hay forma de cubrir eso desde el código.
- **La URL de ngrok es fija, pero no secreta:** cualquiera que la conozca puede llamarla; solo el token protege la API.
