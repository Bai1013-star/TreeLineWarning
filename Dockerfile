FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY algorithms ./algorithms
COPY api ./api
COPY core ./core
RUN python -m pip install --no-cache-dir .

COPY web ./web
COPY models/species_growth.json ./models/species_growth.json
COPY models/temporal_attention.npz ./models/temporal_attention.npz

RUN addgroup --system --gid 10001 treeline \
    && adduser --system --uid 10001 --ingroup treeline --home /app treeline \
    && mkdir -p /app/data /app/models/trained \
    && chown -R treeline:treeline /app/data /app/models/trained

USER treeline

EXPOSE 8010

HEALTHCHECK --interval=30s --timeout=8s --start-period=30s --retries=3 \
    CMD python -c "import base64,os,urllib.request; u=os.getenv('TREELINE_WARNING_ACCESS_USERNAME','treeline'); p=os.environ['TREELINE_WARNING_ACCESS_PASSWORD']; token=base64.b64encode(f'{u}:{p}'.encode()).decode(); request=urllib.request.Request('http://127.0.0.1:8010/health',headers={'Authorization':f'Basic {token}'}); assert urllib.request.urlopen(request,timeout=5).status == 200"

CMD ["python", "-m", "uvicorn", "api.prediction_api:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips=*"]
