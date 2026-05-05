param(
    [string]$Python = ".\venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Desktop = Join-Path $Root "desktop"
$SidecarDir = Join-Path $Desktop "src-tauri\bin"
$DistDir = Join-Path $Desktop "dist"

New-Item -ItemType Directory -Force -Path $SidecarDir | Out-Null
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

& $Python -m pip install -r (Join-Path $Root "requirements.txt")
& $Python -m PyInstaller `
    --clean `
    --onefile `
    --name meeting-copilot-server `
    --add-data "frontend;frontend" `
    --add-data ".env.example;." `
    (Join-Path $Root "server.py")

$Exe = Join-Path $Root "dist\meeting-copilot-server.exe"
if (!(Test-Path $Exe)) {
    throw "PyInstaller did not create $Exe"
}

Copy-Item -Force $Exe (Join-Path $SidecarDir "meeting-copilot-server-x86_64-pc-windows-msvc.exe")

$Index = Join-Path $DistDir "index.html"
Set-Content -Encoding UTF8 -Path $Index -Value @"
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Meeting Co-Pilot</title>
    <meta http-equiv="refresh" content="0; url=http://127.0.0.1:8012">
  </head>
  <body>Opening Meeting Co-Pilot...</body>
</html>
"@

Push-Location $Desktop
try {
    npm install
    npm run build
}
finally {
    Pop-Location
}
