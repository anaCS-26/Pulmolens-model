#!/bin/bash
set -e

echo "=== Clean Slate Azure Deployment ==="

# Configuration
RESOURCE_GROUP="PulmoLense"
ACR_NAME="pulmolensacr"
CONTAINER_APP_NAME="pulmolens-container-dev"
CONTAINER_ENV="pulmolens-env-dev"
IMAGE_TAG="v6-conditional"
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
    
echo "Step 2.5: Fetching Secrets..."
STORAGE_CONN_STR=$(az storage account show-connection-string --name pulmolensstoragedev --resource-group $RESOURCE_GROUP --output tsv)
# Try to fetch Cosmos Key, might fail if not created yet
COSMOS_KEY=$(az cosmosdb keys list --name pulmolens-cosmos-dev --resource-group $RESOURCE_GROUP --query primaryMasterKey -o tsv || echo "placeholder")

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
    --max-replicas 10 \
    --env-vars \
        COSMOS_ENDPOINT=https://pulmolens-cosmos-dev.documents.azure.com:443/ \
        COSMOS_KEY=$COSMOS_KEY \
        STORAGE_CONN_STR=$STORAGE_CONN_STR

echo ""
echo "=== Deployment Complete ==="
echo "Container App URL: (Check Azure Portal or run 'az containerapp show ...')"
echo ""
