$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    throw 'Dependencies are not installed. Follow the setup steps in README.md.'
}
& '.\.venv\Scripts\python.exe' -m uvicorn app:app --host 127.0.0.1 --port 8000
