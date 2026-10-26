# Deploy the PlayerPulse API + dashboard to Azure App Service (Linux, free F1 tier).
# Run from the repo root after `az login`. Copies only what the API needs (no raw data, no model.joblib).
$name = "playerpulse-sanjay"; $rg = "playerpulse-rg"; $region = "indiasouthcentral"
$d = "..\PlayerPulse_azure"
Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $d, "$d\models" | Out-Null
Copy-Item api, requirements.txt $d -Recurse
Copy-Item models\* "$d\models" -Exclude model.joblib
Push-Location $d
az webapp up --name $name --resource-group $rg --runtime "PYTHON:3.12" --sku F1 --location $region
az webapp config set --name $name --resource-group $rg --startup-file "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"
Pop-Location
