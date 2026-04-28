# Deployment Guide — Streamlit Community Cloud

The dashboard is designed to deploy in under five minutes on
[Streamlit Community Cloud](https://streamlit.io/cloud) (free tier).
All required artifacts are committed to the repository, so the hosted
app shows a fully populated dashboard the moment the build finishes.

## Prerequisites

- A GitHub account.
- A fork or clone of this repository pushed to your own GitHub.
- A free Streamlit Community Cloud account (sign in with GitHub).

## Steps

1. **Push the repo to GitHub** (or fork it).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Fill in the form:
   - **Repository:** `<your-username>/environmental-quality-ml-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
   - **Python version:** `3.11` (read from `.python-version`)
4. Click **Deploy**. The first build takes ~3–5 minutes.
5. Streamlit Cloud reads `requirements.txt` automatically. No `packages.txt` is needed.

## Configuration

- **`.streamlit/config.toml`** — already configured for headless operation, XSRF protection, and disabled usage stats. Do not edit unless you know what you're changing.
- **Secrets** — none are required today (Open-Meteo is keyless). If you add API keys in the future, copy `.streamlit/secrets.toml.example` into the **Secrets** panel of the Streamlit Cloud app settings.

## Post-deploy checklist

- [ ] Open the hosted URL and verify the **Executive Summary** tab shows the four headline metrics.
- [ ] Click through each tab — none should show a "metric file not found" warning.
- [ ] Open the **Live Weather** tab, search for "London", confirm the forecast loads.
- [ ] Add the hosted URL as a badge / link in the README.

## Updating the dashboard

Pushes to the configured branch trigger an automatic re-deploy on
Streamlit Cloud. To refresh the committed artifacts after a change:

```bash
make train
make report
git add artifacts/metrics artifacts/figures artifacts/tables artifacts/reports
git commit -m "Refresh dashboard artifacts"
git push
```

## Self-hosting (optional)

If you'd rather not use Streamlit Cloud, the app runs anywhere Python
3.11 is available. The simplest container recipe:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
```

Then `docker build -t aq-dashboard . && docker run -p 8501:8501 aq-dashboard`.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
