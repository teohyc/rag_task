FROM python:3.10-slim

#directory
WORKDIR /app

#dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#backened logic
COPY agentic_backend.py .
COPY api.py . 

#expose the standard FastAPI port
EXPOSE 8000

#run the production ASGI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]