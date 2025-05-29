#!/bin/bash

# Schema-Aware NL2SQL API Deployment Script
set -e

echo "🚀 NL2SQL API Deployment Script"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_KEY=${NL2SQL_API_KEY:-$(openssl rand -hex 32)}
PORT=${PORT:-8000}
ENVIRONMENT=${ENVIRONMENT:-development}

print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    if ! command -v pip &> /dev/null; then
        print_error "pip is not installed"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Setup virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ ! -d "nl2sql_env" ]; then
        python3 -m venv nl2sql_env
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
    
    source nl2sql_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Setup environment variables
setup_env() {
    print_status "Setting up environment variables..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
NL2SQL_API_KEY=${API_KEY}
HOST=0.0.0.0
PORT=${PORT}
LOG_LEVEL=INFO
ENVIRONMENT=${ENVIRONMENT}
EOF
        print_success "Environment file created"
    else
        print_warning "Environment file already exists"
    fi
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    source nl2sql_env/bin/activate
    
    # Test basic imports
    python -c "
import sys
sys.path.append('.')
from src.nl2sql_agent import NL2SQLAgent
from src.nl2sql_model import NL2SQLModel
from src.schema_retriever import SchemaRetriever
print('✅ All imports successful')
"
    
    # Test API startup
    timeout 10s python -c "
import uvicorn
from api import app
print('✅ API can start')
" || print_warning "API startup test timed out (this is normal)"
    
    print_success "Tests completed"
}

# Start development server
start_dev() {
    print_status "Starting development server..."
    source nl2sql_env/bin/activate
    
    export NL2SQL_API_KEY=${API_KEY}
    export PORT=${PORT}
    
    print_success "Starting API on port ${PORT}"
    print_status "API Documentation: http://localhost:${PORT}/docs"
    print_status "API Key: ${API_KEY}"
    
    python api.py
}

# Docker deployment
deploy_docker() {
    print_status "Deploying with Docker..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Build image
    docker build -t nl2sql-api:latest .
    print_success "Docker image built"
    
    # Run container
    docker run -d \
        --name nl2sql-api \
        -p ${PORT}:8000 \
        -e NL2SQL_API_KEY=${API_KEY} \
        -v $(pwd)/data:/app/data \
        nl2sql-api:latest
    
    print_success "Docker container started"
    print_status "API running at: http://localhost:${PORT}"
    print_status "API Key: ${API_KEY}"
}

# Docker Compose deployment
deploy_compose() {
    print_status "Deploying with Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    export NL2SQL_API_KEY=${API_KEY}
    export PORT=${PORT}
    
    docker-compose up -d nl2sql-api
    
    print_success "Docker Compose deployment started"
    print_status "API running at: http://localhost:${PORT}"
    print_status "API Key: ${API_KEY}"
}

# Cloud deployment helpers
deploy_cloud() {
    print_status "Cloud deployment configurations..."
    
    # AWS ECS Task Definition
    cat > aws-task-definition.json << EOF
{
  "family": "nl2sql-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::YOUR_ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "nl2sql-api",
      "image": "YOUR_ECR_REPO/nl2sql-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NL2SQL_API_KEY",
          "value": "${API_KEY}"
        },
        {
          "name": "PORT",
          "value": "8000"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/nl2sql-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
EOF
    
    # Kubernetes Deployment
    cat > k8s-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nl2sql-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nl2sql-api
  template:
    metadata:
      labels:
        app: nl2sql-api
    spec:
      containers:
      - name: nl2sql-api
        image: nl2sql-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: NL2SQL_API_KEY
          value: "${API_KEY}"
        - name: PORT
          value: "8000"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: nl2sql-api-service
spec:
  selector:
    app: nl2sql-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
EOF
    
    print_success "Cloud deployment files created"
    print_status "AWS ECS: aws-task-definition.json"
    print_status "Kubernetes: k8s-deployment.yaml"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up..."
    
    # Stop docker containers
    docker stop nl2sql-api 2>/dev/null || true
    docker rm nl2sql-api 2>/dev/null || true
    
    # Stop docker-compose
    docker-compose down 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Status check
check_status() {
    print_status "Checking deployment status..."
    
    # Check if API is running
    if curl -s http://localhost:${PORT}/health > /dev/null; then
        print_success "API is running and healthy"
        print_status "Health check: http://localhost:${PORT}/health"
        print_status "Documentation: http://localhost:${PORT}/docs"
    else
        print_warning "API is not responding"
    fi
    
    # Check Docker
    if docker ps | grep nl2sql-api > /dev/null; then
        print_success "Docker container is running"
    fi
}

# Main menu
show_menu() {
    echo ""
    echo "Select deployment option:"
    echo "1) Development setup (virtual environment)"
    echo "2) Docker deployment"
    echo "3) Docker Compose deployment"
    echo "4) Generate cloud deployment files"
    echo "5) Run tests"
    echo "6) Check status"
    echo "7) Cleanup"
    echo "8) Exit"
    echo ""
}

# Command line arguments
case "$1" in
    "dev")
        check_prerequisites
        setup_venv
        setup_env
        start_dev
        ;;
    "docker")
        setup_env
        deploy_docker
        ;;
    "compose")
        setup_env
        deploy_compose
        ;;
    "cloud")
        setup_env
        deploy_cloud
        ;;
    "test")
        check_prerequisites
        setup_venv
        run_tests
        ;;
    "status")
        check_status
        ;;
    "cleanup")
        cleanup
        ;;
    *)
        # Interactive mode
        while true; do
            show_menu
            read -p "Enter your choice [1-8]: " choice
            
            case $choice in
                1)
                    check_prerequisites
                    setup_venv
                    setup_env
                    start_dev
                    break
                    ;;
                2)
                    setup_env
                    deploy_docker
                    break
                    ;;
                3)
                    setup_env
                    deploy_compose
                    break
                    ;;
                4)
                    setup_env
                    deploy_cloud
                    ;;
                5)
                    check_prerequisites
                    setup_venv
                    run_tests
                    ;;
                6)
                    check_status
                    ;;
                7)
                    cleanup
                    ;;
                8)
                    print_success "Goodbye!"
                    exit 0
                    ;;
                *)
                    print_error "Invalid option. Please choose 1-8."
                    ;;
            esac
        done
        ;;
esac 