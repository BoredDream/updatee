param(
    [switch]$Install,
    [int]$K = 30
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
& (Join-Path $RepoRoot "run.ps1") -Question q3 -K $K -Install:$Install
