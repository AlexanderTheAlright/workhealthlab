<#
.SYNOPSIS
    Sociopath-it initialization utility - create projects and notebooks.

.DESCRIPTION
    Main wrapper script for Sociopath-it project and notebook templates.
    Quick and easy way to start new analysis projects.

.PARAMETER Action
    What to create: "project", "notebook", or "both"

.PARAMETER Path
    Where to create the files/directories

.PARAMETER Type
    For notebooks: general, regression, textual, descriptive, causal, longitudinal
    For projects: N/A (always creates full structure)

.PARAMETER ProjectName
    Optional project name (for projects only)

.EXAMPLE
    .\workhealthlab-init.ps1 -Action project -Path "C:\Research\MyStudy"

.EXAMPLE
    .\workhealthlab-init.ps1 -Action notebook -Path "analysis.ipynb" -Type regression

.EXAMPLE
    .\workhealthlab-init.ps1 -Action both -Path "C:\Research\NewProject" -Type general

.NOTES
    Author: Sociopath-it
    Quick start utility for the Sociopath-it package
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("project", "notebook", "both")]
    [string]$Action,

    [Parameter(Mandatory=$true)]
    [string]$Path,

    [Parameter(Mandatory=$false)]
    [ValidateSet("general", "regression", "textual", "descriptive", "causal", "longitudinal")]
    [string]$Type = "general",

    [Parameter(Mandatory=$false)]
    [string]$ProjectName = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host @"

    ███████╗ ██████╗  ██████╗██╗ ██████╗ ██████╗  █████╗ ████████╗██╗  ██╗    ██╗████████╗
    ██╔════╝██╔═══██╗██╔════╝██║██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝██║  ██║    ██║╚══██╔══╝
    ███████╗██║   ██║██║     ██║██║   ██║██████╔╝███████║   ██║   ███████║    ██║   ██║
    ╚════██║██║   ██║██║     ██║██║   ██║██╔═══╝ ██╔══██║   ██║   ██╔══██║    ██║   ██║
    ███████║╚██████╔╝╚██████╗██║╚██████╔╝██║     ██║  ██║   ██║   ██║  ██║    ██║   ██║
    ╚══════╝ ╚═════╝  ╚═════╝╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝   ╚═╝

    Initialization Utility - Because your data won't analyze itself
"@ -ForegroundColor Cyan

Write-Host ""

switch ($Action) {
    "project" {
        $createProjectScript = Join-Path $scriptDir "create-project.ps1"

        if (Test-Path $createProjectScript) {
            if ($ProjectName) {
                & $createProjectScript -ProjectPath $Path -ProjectName $ProjectName
            } else {
                & $createProjectScript -ProjectPath $Path
            }
        } else {
            Write-Host "[!] create-project.ps1 not found in $scriptDir" -ForegroundColor Red
            exit 1
        }
    }

    "notebook" {
        $createNotebookScript = Join-Path $scriptDir "create-notebook.ps1"

        if (Test-Path $createNotebookScript) {
            & $createNotebookScript -NotebookPath $Path -NotebookType $Type
        } else {
            Write-Host "[!] create-notebook.ps1 not found in $scriptDir" -ForegroundColor Red
            exit 1
        }
    }

    "both" {
        Write-Host "Creating project with starter notebook..." -ForegroundColor Yellow
        Write-Host ""

        $createProjectScript = Join-Path $scriptDir "create-project.ps1"
        $createNotebookScript = Join-Path $scriptDir "create-notebook.ps1"

        if (!(Test-Path $createProjectScript) -or !(Test-Path $createNotebookScript)) {
            Write-Host "[!] Required scripts not found in $scriptDir" -ForegroundColor Red
            exit 1
        }

        # Create project
        if ($ProjectName) {
            & $createProjectScript -ProjectPath $Path -ProjectName $ProjectName
        } else {
            & $createProjectScript -ProjectPath $Path
        }

        Write-Host ""

        # Create notebook in analyses folder
        $notebookPath = Join-Path $Path "analyses\$Type`_analysis.ipynb"
        & $createNotebookScript -NotebookPath $notebookPath -NotebookType $Type
    }
}

Write-Host ""
Write-Host "Done! Now go forth and analyze." -ForegroundColor Green
Write-Host ""
