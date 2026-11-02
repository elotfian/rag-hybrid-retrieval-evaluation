FROM python:3.12-slim

WORKDIR /app

# CPU-only torch first, to keep the image smaller
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY manual_rag.py langchain_rrf.py eval.py ./
COPY tests ./tests

CMD ["python", "eval.py"]