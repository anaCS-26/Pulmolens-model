// PulmoLens Enterprise Infrastructure - Azure Bicep Template
targetScope = 'subscription'

param location string = 'canadacentral'
param projectName string = 'pulmolens'
param environment string = 'prod'

var resourceGroupName = 'rg-${projectName}-${environment}'

resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: location
}

module storage 'storage.bicep' = {
  scope: rg
  name: 'storage-deployment'
  params: {
    location: location
    projectName: projectName
  }
}

module containerApp 'aca.bicep' = {
  scope: rg
  name: 'aca-deployment'
  params: {
    location: location
    projectName: projectName
    dockerImage: 'pulmolens-backend:latest'
  }
}
