# Chapter 40: Deployment Patterns - Code Examples

This directory contains all code examples for Chapter 40.

## Files Overview

### Core API Examples

| File | Description |
|------|-------------|
| `basic_api.py` | Basic FastAPI wrapper for an AI agent |
| `streaming_api.py` | Streaming responses using Server-Sent Events |
| `background_worker.py` | Background task processing for long-running operations |
| `production_api.py` | Complete production-ready API with all features |

### Configuration & Infrastructure

| File | Description |
|------|-------------|
| `settings.py` | Environment configuration using Pydantic Settings |
| `health_checks.py` | Health check endpoints (liveness, readiness, detailed) |
| `rate_limiter.py` | Rate limiting middleware (in-memory and Redis) |

### Deployment Files

| File | Description |
|------|-------------|
| `Dockerfile` | Multi-stage Docker build for production |
| `docker-compose.yml` | Service orchestration for local development |
| `requirements.txt` | Python dependencies |

### Exercise

| File | Description |
|------|-------------|
| `exercise_solution.py` | Complete deployable example for cloud platforms |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:

```env
ANTHROPIC_API_KEY=your-api-key-here
ENVIRONMENT=development
DEBUG=true
```

### 3. Run the Basic API

```bash
python basic_api.py
```

Then visit:
- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

## Running with Docker

### Build and Run

```bash
# Build
docker build -t agent-api .

# Run
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your-key agent-api
```

### With Docker Compose

```bash
docker-compose up
```

## File Descriptions

### basic_api.py

The minimal setup needed to expose an AI agent as a REST API:
- Simple agent class with conversation management
- Health check endpoint
- Chat endpoint with error handling
- CORS middleware

### streaming_api.py

Demonstrates streaming responses for better user experience:
- Server-Sent Events (SSE) format
- Real-time text streaming
- Includes demo HTML page at `/demo`

### background_worker.py

For long-running tasks (>30 seconds):
- Task submission endpoint
- Status polling endpoint
- In-memory task store (use Redis in production)
- Includes demo page

### settings.py

Environment configuration management:
- Pydantic Settings for type-safe config
- Environment-specific settings (dev/staging/prod)
- Example `.env` files

### health_checks.py

Comprehensive health monitoring:
- `/health` - Detailed health with all checks
- `/health/live` - Liveness probe (is process running?)
- `/health/ready` - Readiness probe (can handle requests?)
- Memory and disk usage checks

### rate_limiter.py

Protect your API from abuse:
- In-memory rate limiter for single instances
- Redis rate limiter for distributed systems
- Tiered rate limiting for freemium models
- Rate limit headers in responses

### production_api.py

Everything combined for production:
- All health checks
- Rate limiting
- Streaming support
- Metrics collection
- Structured logging
- Proper error handling

## Deployment Checklist

Before deploying to production:

- [ ] Set `ENVIRONMENT=production`
- [ ] Disable debug mode (`DEBUG=false`)
- [ ] Configure proper CORS origins
- [ ] Set up Redis for rate limiting (if multiple instances)
- [ ] Configure logging aggregation
- [ ] Set up monitoring and alerts
- [ ] Use secrets management for API keys
- [ ] Enable HTTPS (via load balancer or reverse proxy)

## Cloud Deployment

See `exercise_solution.py` for a complete deployable example.

### Supported Platforms

- **Railway**: Connect GitHub repo, add env vars, deploy
- **Render**: Create web service, configure environment
- **Google Cloud Run**: Build container, deploy
- **AWS App Runner**: Push to ECR, create service
- **Kubernetes**: Use provided deployment manifests

## Requirements

- Python 3.10+
- FastAPI
- Anthropic SDK
- uvicorn
- pydantic-settings

Optional (for production features):
- redis
- psutil
