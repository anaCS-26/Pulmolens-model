# PulmoLens Azure Resource Cleanup Script
# This script identifies and deletes orphaned Log Analytics workspaces to save billing and reduce portal clutter.

Write-Host "Searching for orphaned PulmoLens Log Analytics workspaces..." -ForegroundColor Cyan

# 1. List all PulmoLens workspaces
$workspaces = az monitor log-analytics workspace list --query "[?contains(to_string(name), 'pulmolens')].{Name:name, RG:resourceGroup, ID:id}" -o json | ConvertFrom-Json

if ($workspaces.Count -eq 0) {
    Write-Host "No orphaned PulmoLens workspaces found. Everything is clean!" -ForegroundColor Green
} else {
    Write-Host "Found $($workspaces.Count) potential orphaned workspaces:" -ForegroundColor Yellow
    $workspaces | Format-Table
    
    $confirm = Read-Host "Do you want to DELETE these workspaces? (y/n)"
    if ($confirm -eq 'y') {
        foreach ($ws in $workspaces) {
            Write-Host "Deleting $($ws.Name)..."
            az monitor log-analytics workspace delete --resource-group $ws.RG --workspace-name $ws.Name --no-wait
        }
        Write-Host "Cleanup triggered in Azure. Tasks are running in background." -ForegroundColor Green
    } else {
        Write-Host "Cleanup cancelled." -ForegroundColor Gray
    }
}

# 2. Tip: Also check for unused Public IPs or unattached Disks if your experimentation included VMs.
Write-Host "`nRecommendation: Set your Container App --min-replicas to 0 globally if you haven't yet." -ForegroundColor Cyan
