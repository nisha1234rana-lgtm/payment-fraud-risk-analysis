$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RawDir = Join-Path $ProjectRoot "data\raw"

New-Item -ItemType Directory -Force -Path $RawDir | Out-Null

Write-Host "Downloading Kaggle dataset: ismetsemedov/transactions"
kaggle datasets download -d ismetsemedov/transactions -p $RawDir --unzip

$Expected = Join-Path $RawDir "synthetic_fraud_data.csv"

if (Test-Path $Expected) {
    Write-Host "Download complete:"
    Write-Host $Expected
} else {
    Write-Host "Download finished, but synthetic_fraud_data.csv was not found."
    Write-Host "Check the contents of data\raw."
}
