$ErrorActionPreference = 'Stop'
$destination = Join-Path $PSScriptRoot 'raw'
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$sourcesPath = Join-Path $destination 'sources.json'
$sources = Get-Content -LiteralPath $sourcesPath -Raw | ConvertFrom-Json
Invoke-WebRequest -Uri $sources.repodb_url -OutFile (Join-Path $destination 'repodb_original.RData')
Invoke-WebRequest -Uri $sources.do_url -OutFile (Join-Path $destination 'doid.obo')
Write-Output 'Downloaded pinned sources. The benchmark records SHA256 in its run manifest.'
