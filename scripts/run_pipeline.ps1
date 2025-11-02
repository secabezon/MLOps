# PowerShell helper to run DVC pipeline and optionally generate a DAG PNG
param(
    [switch]$GeneratePng
)

# Activate virtualenv if exists
if (Test-Path -Path .venv\Scripts\Activate.ps1) {
    Write-Host "Activating .venv"
    & .venv\Scripts\Activate.ps1
} else {
    Write-Host "No .venv activation script found; ensure your environment is active if needed."
}

Write-Host "Running: dvc repro"
dvc repro

if ($GeneratePng) {
    $dot = Get-Command dot -ErrorAction SilentlyContinue
    if ($dot) {
        Write-Host "Generating pipeline.png via dot"
        dvc dag --dot | dot -Tpng -o pipeline.png
        Write-Host "Wrote pipeline.png"
    } else {
        Write-Host "Graphviz 'dot' not found. Install graphviz to generate pipeline.png"
    }
}
