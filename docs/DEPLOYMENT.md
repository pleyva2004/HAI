# Deployment Guide - SAT Question Generator

This guide covers deploying the SAT Question Generator application, from local development to production Kubernetes clusters.

---

## Table of Contents

1. [Local Development](#local-development)
2. [Running the Application](#running-the-application)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Worker Processes & Concurrency](#worker-processes--concurrency)
6. [Scaling Strategies](#scaling-strategies)
7. [Production Best Practices](#production-best-practices)

---

## Local Development

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- Environment variables configured (see `backend/config.py`)

### Setup

```bash
cd backend
pip install -r requirements.txt
```

### Running Locally

**Option 1: Direct Python execution (development)**
```bash
python backend/main.py
```

**Option 2: Uvicorn CLI (recommended)**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables hot-reload for development.

---

## Running the Application

### Why `import uvicorn` is inside `if __name__ == "__main__"`

The current setup places `import uvicorn` inside the `if __name__ == "__main__"` block for:

1. **Lazy Loading**: Only imports uvicorn when running directly
2. **Optional Dependency**: Module can be imported without uvicorn installed
3. **Separation of Concerns**: App definition separate from execution

### Two Ways to Run

#### Method 1: Direct Python Execution
```bash
python backend/main.py
```
- Executes the `if __name__ == "__main__"` block
- Imports uvicorn and starts server
- Good for: Quick testing, development

#### Method 2: Uvicorn CLI (Production Standard)
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
- CLI imports the `app` object directly
- Doesn't execute `if __name__ == "__main__"` block
- Good for: Production, more control, better process management

### Production Differences

| Aspect | `python main.py` | `uvicorn main:app` |
|--------|------------------|-------------------|
| **Workers** | Single process only | Supports `--workers N` |
| **Configuration** | Hardcoded in script | CLI flags for all options |
| **Hot Reload** | No | `--reload` flag |
| **Process Management** | Basic | Production-ready |
| **Logging** | Basic | Structured access logs |
| **Graceful Shutdown** | Basic | Full signal handling |

**Recommendation**: Use `uvicorn main:app` for production deployments.

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed for PostgreSQL client)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run with multiple workers
# Note: In K8s, use fewer workers per pod since you'll have multiple pods
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Building and Running

```bash
# Build image
docker build -t sat-question-generator:latest .

# Run container
docker run -d \
  --name sat-generator \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e CLAUDE_API_KEY=your_key \
  sat-question-generator:latest
```

### Docker Compose (Development)

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/satdb
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
    depends_on:
      - db
    volumes:
      - ./backend:/app
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: pgvector/pgvector:pg16
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=satdb
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Kubernetes Deployment

### Architecture Overview

```
                    Internet
                       │
                       ▼
              ┌─────────────────┐
              │  Load Balancer  │
              │  (Kubernetes)   │
              └─────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │  Pod 1  │   │  Pod 2  │   │  Pod 3  │
   │(4 workers)│ │(4 workers)│ │(4 workers)│
   └─────────┘   └─────────┘   └─────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   PostgreSQL    │
              │   (Database)    │
              └─────────────────┘
```

### Deployment Configuration

**`k8s-deployment.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sat-question-generator
  labels:
    app: sat-generator
spec:
  replicas: 2  # Start with 2 pods
  selector:
    matchLabels:
      app: sat-generator
  template:
    metadata:
      labels:
        app: sat-generator
    spec:
      containers:
      - name: backend
        image: your-registry/sat-generator:v1.0.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: CLAUDE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: claude-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"      # 0.5 CPU cores
          limits:
            memory: "1Gi"
            cpu: "2000m"     # 2 CPU cores max
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
---
apiVersion: v1
kind: Service
metadata:
  name: sat-generator-service
spec:
  selector:
    app: sat-generator
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  type: LoadBalancer  # Use ClusterIP for internal-only access
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sat-generator-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sat-question-generator
  minReplicas: 2      # Minimum 2 pods
  maxReplicas: 10     # Maximum 10 pods
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # Scale when CPU > 70%
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80  # Scale when Memory > 80%
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 2
        periodSeconds: 15
      selectPolicy: Max
```

### Secrets Configuration

**`k8s-secrets.yaml`**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
type: Opaque
stringData:
  url: postgresql://user:password@postgres-service:5432/satdb
---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
stringData:
  claude-key: your-claude-api-key-here
```

### Deploying to Kubernetes

```bash
# Create secrets
kubectl apply -f k8s-secrets.yaml

# Deploy application
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=sat-generator
kubectl get hpa sat-generator-hpa

# View logs
kubectl logs -f deployment/sat-question-generator

# Scale manually (if needed)
kubectl scale deployment sat-question-generator --replicas=5
```

---

## Worker Processes & Concurrency

### How Workers Handle Multiple Users

Your application is **I/O-bound** (Claude API calls, database queries), making it ideal for multiple workers.

#### Without Workers (Single Process)
```
Request 1: Tutor A → 8 seconds
Request 2: Tutor B → waits 8s, then 8s = 16s total
Request 3: Tutor C → waits 16s, then 8s = 24s total
```

#### With 4 Workers
```
Request 1: Worker 1 → 8 seconds
Request 2: Worker 2 → 8 seconds (parallel)
Request 3: Worker 3 → 8 seconds (parallel)
Request 4: Worker 4 → 8 seconds (parallel)
Total: ~8 seconds for 4 requests
```

### Workflow Timing

Based on your LangGraph workflow:
- `extract_structure`: ~2.3s (Claude Vision)
- `classify_question`: ~0.8s (Claude)
- `retrieve_examples`: ~0.3s (DB + embedding)
- `generate_question`: ~3.1s (Claude)
- `validate_output`: ~0.1s
- **Total: ~6.6 seconds per request**

### Capacity Calculation

**Per Pod:**
- 4 workers per pod
- Each worker handles ~5-10 concurrent async requests
- **Per pod: ~20-40 concurrent users**

**With 3 Pods:**
- 3 pods × 4 workers = 12 workers
- **Total: ~60-120 concurrent users**

**With Autoscaling (2-10 pods):**
- Minimum: 2 pods = ~40-80 users
- Maximum: 10 pods = ~200-400 users

### Worker Configuration

```bash
# Light usage (1-5 concurrent users)
uvicorn backend.main:app --workers 2

# Medium usage (5-20 concurrent users)
uvicorn backend.main:app --workers 4

# Heavy usage (20-50 concurrent users)
uvicorn backend.main:app --workers 8

# Very heavy (50+ concurrent users)
# Use Gunicorn + Uvicorn workers
gunicorn backend.main:app \
  -w 8 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## Scaling Strategies

### Horizontal Scaling (Kubernetes)

**Automatic Scaling:**
- HPA monitors CPU/memory usage
- Scales pods up/down based on demand
- Handles traffic spikes automatically

**Manual Scaling:**
```bash
kubectl scale deployment sat-question-generator --replicas=5
```

### Vertical Scaling (Resource Limits)

Adjust pod resources in deployment:

```yaml
resources:
  requests:
    memory: "1Gi"    # Increase for more memory
    cpu: "1000m"      # Increase for more CPU
  limits:
    memory: "2Gi"
    cpu: "4000m"
```

### Database Connection Pooling

With multiple pods, configure connection pooling:

```python
# database/connection.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,        # Connections per pod
    max_overflow=20,     # Extra connections if needed
    pool_pre_ping=True   # Verify connections before use
)
```

**With 3 pods:**
- Each pod: 10 connections
- Total: 30 connections to PostgreSQL
- PostgreSQL can handle hundreds of connections

### Request Flow Example

**Single User:**
```
User → Load Balancer → Pod 2 → Worker 3 → FastAPI → LangGraph → Response
```

**Multiple Users:**
```
User A → Pod 1, Worker 1 → Processing...
User B → Pod 2, Worker 2 → Processing...
User C → Pod 3, Worker 1 → Processing...
User D → Pod 1, Worker 2 → Processing...
...all happening in parallel
```

---

## Production Best Practices

### 1. Health Checks

Add health check endpoints to `backend/main.py`:

```python
@app.get("/health")
async def health_check():
    """Liveness probe - is the app running?"""
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check():
    """Readiness probe - can the app handle requests?"""
    # Check database connection
    try:
        await init_database()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Not ready")
```

### 2. Environment Variables

Use Kubernetes secrets for sensitive data:
- Database credentials
- API keys (Claude, etc.)
- Never commit secrets to git

### 3. Logging

Configure structured logging:

```python
import logging
from pythonjsonlogger import jsonlogger

logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)
```

### 4. Rate Limiting

Consider rate limiting for Claude API calls:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/generate")
@limiter.limit("10/minute")
async def generate_question(request: Request, ...):
    ...
```

### 5. Monitoring

Set up monitoring for:
- Request latency
- Error rates
- Resource usage (CPU, memory)
- Database connection pool usage
- Claude API response times

### 6. Graceful Shutdown

Uvicorn handles graceful shutdown automatically:
- Waits for in-flight requests to complete
- Closes connections cleanly
- Kubernetes handles pod termination gracefully

### 7. Resource Limits

Always set resource limits to prevent:
- One pod consuming all cluster resources
- OOM (Out of Memory) kills
- CPU throttling issues

### 8. Rolling Updates

Kubernetes supports zero-downtime updates:

```bash
# Update image
kubectl set image deployment/sat-question-generator \
  backend=your-registry/sat-generator:v1.1.0

# Monitor rollout
kubectl rollout status deployment/sat-question-generator

# Rollback if needed
kubectl rollout undo deployment/sat-question-generator
```

### 9. CORS Configuration

Update CORS for production:

```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-production-domain.com",
        "https://www.your-production-domain.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 10. Database Migrations

Run migrations before deployment:

```bash
# In your deployment pipeline
kubectl run migration-job \
  --image=your-registry/sat-generator:v1.0.0 \
  --command -- python scripts/migrate.py
```

---

## Deployment Checklist

- [ ] Environment variables configured
- [ ] Secrets created in Kubernetes
- [ ] Health check endpoints implemented
- [ ] Resource limits set appropriately
- [ ] Database connection pooling configured
- [ ] CORS configured for production domains
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] HPA configured and tested
- [ ] Rolling update strategy tested
- [ ] Backup strategy for database
- [ ] Disaster recovery plan documented

---

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>

# Common issues:
# - Missing secrets
# - Database connection failures
# - Resource limits too low
```

### High Latency

```bash
# Check resource usage
kubectl top pods

# Check HPA status
kubectl describe hpa sat-generator-hpa

# Scale up if needed
kubectl scale deployment sat-question-generator --replicas=5
```

### Database Connection Issues

```bash
# Check connection pool settings
# Verify database is accessible from pods
kubectl run -it --rm debug --image=postgres:16 --restart=Never -- \
  psql -h postgres-service -U user -d satdb
```

---

## Summary

- **Local Development**: Use `uvicorn --reload` for hot-reload
- **Production**: Use `uvicorn main:app` with multiple workers
- **Docker**: Single container with 4 workers handles ~20-40 users
- **Kubernetes**: Multiple pods with HPA scales to 200-400+ users
- **Workers**: Essential for I/O-bound AI applications
- **Scaling**: Horizontal (pods) + Vertical (resources) + Connection pooling

Your application is well-suited for containerized deployment with automatic scaling, handling concurrent users efficiently through worker processes and Kubernetes orchestration.

