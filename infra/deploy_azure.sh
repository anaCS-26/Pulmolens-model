#!/bin/bash
set -e

echo "=== Clean Slate Azure Deployment ==="

# Configuration
RESOURCE_GROUP="PulmoLense"
ACR_NAME="pulmolensacr"
CONTAINER_APP_NAME="pulmolens-container"
CONTAINER_ENV="pulmolens-env-fresh"
IMAGE_TAG="v5-ratio-fix"
LOCATION="canadacentral"

echo "Step 1: Building and pushing Docker image..."
az acr build \
    --registry $ACR_NAME \
    --image $CONTAINER_APP_NAME:$IMAGE_TAG \
    --file Dockerfile \
    .

echo "Step 2: Creating Container App Environment..."
az containerapp env create \
    --name $CONTAINER_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION

echo "Step 3: Creating Container App..."
az containerapp create \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $CONTAINER_ENV \
    --image "$ACR_NAME.azurecr.io/$CONTAINER_APP_NAME:$IMAGE_TAG" \
    --target-port 8000 \
    --ingress 'external' \
    --registry-server "$ACR_NAME.azurecr.io" \
    --cpu 1.0 \
    --memory 2.0Gi \
    --min-replicas 1 \
    --max-replicas 10

echo ""
echo "=== Deployment Complete ==="
echo "Container App URL: https://$CONTAINER_APP_NAME.jollycoast-40cf81b8.canadacentral.azurecontainerapps.io"
echo ""
