<#
.SYNOPSIS
    Create a standardized Sociopath-it project structure.

.DESCRIPTION
    Generates a complete project directory structure for sociological data analysis.
    Includes data, notes, analyses, and output folders with README templates.

.PARAMETER ProjectPath
    The root directory where the project will be created.

.PARAMETER ProjectName
    Optional name for the project (used in README). Defaults to directory name.

.PARAMETER IncludeNotebooks
    Switch to include template notebooks in the project.

.EXAMPLE
    .\create-project.ps1 -ProjectPath "C:\Research\MyStudy"

.EXAMPLE
    .\create-project.ps1 -ProjectPath "C:\Research\MyStudy" -ProjectName "Income Inequality Study" -IncludeNotebooks

.NOTES
    Author: Sociopath-it
    Part of the Sociopath-it package utilities
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectPath,

    [Parameter(Mandatory=$false)]
    [string]$ProjectName = "",

    [Parameter(Mandatory=$false)]
    [switch]$IncludeNotebooks
)

# Resolve full path
$ProjectPath = [System.IO.Path]::GetFullPath($ProjectPath)

# Use directory name if no project name provided
if ([string]::IsNullOrEmpty($ProjectName)) {
    $ProjectName = Split-Path -Leaf $ProjectPath
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Sociopath-it Project Generator" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Creating project structure at: $ProjectPath" -ForegroundColor Yellow
Write-Host "Project name: $ProjectName`n" -ForegroundColor Yellow

# Create main directory
if (!(Test-Path $ProjectPath)) {
    New-Item -ItemType Directory -Path $ProjectPath | Out-Null
    Write-Host "[✓] Created root directory" -ForegroundColor Green
} else {
    Write-Host "[!] Root directory already exists" -ForegroundColor Yellow
}

# Create subdirectories
$directories = @(
    "data\raw",
    "data\processed",
    "data\metadata",
    "notes\literature",
    "notes\codebooks",
    "notes\meetings",
    "analyses\descriptive",
    "analyses\models",
    "analyses\figures",
    "output\tables",
    "output\figures",
    "output\reports"
)

foreach ($dir in $directories) {
    $fullPath = Join-Path $ProjectPath $dir
    if (!(Test-Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath | Out-Null
        Write-Host "[✓] Created: $dir" -ForegroundColor Green
    }
}

# Create main README
$mainReadme = @"
# $ProjectName

Research project using Sociopath-it for data analysis and visualization.

## Project Structure

``````
$ProjectName/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned and transformed data
│   └── metadata/         # Codebooks, variable descriptions
├── notes/
│   ├── literature/       # Literature reviews, theory notes
│   ├── codebooks/        # Variable coding schemes
│   └── meetings/         # Meeting notes, project updates
├── analyses/
│   ├── descriptive/      # Exploratory analysis notebooks
│   ├── models/           # Statistical models and results
│   └── figures/          # Saved visualizations
└── output/
    ├── tables/           # Publication-ready tables
    ├── figures/          # Final figures for papers
    └── reports/          # Generated reports and manuscripts
``````

## Getting Started

1. Place raw data files in ``data/raw/``
2. Create analysis notebooks in ``analyses/``
3. Run data preparation and cleaning
4. Generate figures and tables
5. Export results to ``output/``

## Dependencies

``````python
pip install git+https://github.com/AlexanderTheAlright/workhealthlab.git
``````

## Notes

- Never modify files in ``data/raw/`` - keep original data immutable
- Document all data transformations in notebooks
- Use consistent naming conventions for files
- Commit notebooks with outputs cleared (except final versions)

---

**Status**: In Progress
**Created**: $(Get-Date -Format "yyyy-MM-dd")
**Last Updated**: $(Get-Date -Format "yyyy-MM-dd")
"@

Set-Content -Path (Join-Path $ProjectPath "README.md") -Value $mainReadme
Write-Host "[✓] Created main README.md" -ForegroundColor Green

# Create data README
$dataReadme = @"
# Data Directory

## Structure

- ``raw/`` - Original, immutable data files. Never modify these directly.
- ``processed/`` - Cleaned, transformed, and analysis-ready datasets.
- ``metadata/`` - Codebooks, variable descriptions, survey instruments.

## Data Management

### Raw Data
- Keep all original data files here
- Document data sources in this README
- Use descriptive filenames with dates (e.g., ``survey_wave1_2023-01-15.csv``)

### Processed Data
- Store cleaned datasets ready for analysis
- Include processing date in filename
- Document transformations in analysis notebooks

### Metadata
- Store codebooks and data dictionaries
- Include variable labels and value codes
- Document missing data conventions

## Data Sources

| File | Source | Date | Description |
|------|--------|------|-------------|
| | | | |

## Notes

- All data files are .gitignored by default for privacy
- Ensure compliance with ethics protocols
- Document any data restrictions or usage limitations
"@

Set-Content -Path (Join-Path $ProjectPath "data\README.md") -Value $dataReadme
Write-Host "[✓] Created data/README.md" -ForegroundColor Green

# Create .gitignore
$gitignore = @"
# Data files (privacy and size)
data/raw/*
data/processed/*
*.csv
*.dta
*.sav
*.xlsx
*.xls

# Except metadata
!data/metadata/*

# Jupyter Notebook checkpoints
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Virtual environments
venv/
env/
ENV/

# Output files (keep in local only)
output/figures/*.png
output/figures/*.pdf
output/tables/*.csv
output/tables/*.xlsx

# OS files
.DS_Store
Thumbs.db
desktop.ini

# Temporary files
*.tmp
*.bak
*~
"@

Set-Content -Path (Join-Path $ProjectPath ".gitignore") -Value $gitignore
Write-Host "[✓] Created .gitignore" -ForegroundColor Green

# Create environment.yml for conda
$envYml = @"
name: $($ProjectName.Replace(' ', '-').ToLower())
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scipy
  - statsmodels
  - scikit-learn
  - matplotlib
  - seaborn
  - plotly
  - jupyter
  - notebook
  - pip
  - pip:
    - git+https://github.com/AlexanderTheAlright/workhealthlab.git
"@

Set-Content -Path (Join-Path $ProjectPath "environment.yml") -Value $envYml
Write-Host "[✓] Created environment.yml" -ForegroundColor Green

# Create requirements.txt for pip
$requirements = @"
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
jupyter>=1.0.0
notebook>=6.5.0
git+https://github.com/AlexanderTheAlright/workhealthlab.git
"@

Set-Content -Path (Join-Path $ProjectPath "requirements.txt") -Value $requirements
Write-Host "[✓] Created requirements.txt" -ForegroundColor Green

# Create template notebooks if requested
if ($IncludeNotebooks) {
    Write-Host "`nCreating template notebooks..." -ForegroundColor Cyan

    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $notebookScript = Join-Path $scriptDir "create-notebook.ps1"

    if (Test-Path $notebookScript) {
        # Create one of each notebook type
        $notebookTypes = @("general", "regression", "textual")
        foreach ($type in $notebookTypes) {
            $notebookPath = Join-Path $ProjectPath "analyses\$type`_analysis.ipynb"
            & $notebookScript -NotebookPath $notebookPath -NotebookType $type
        }
    } else {
        Write-Host "[!] create-notebook.ps1 not found, skipping notebooks" -ForegroundColor Yellow
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Project created successfully!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. cd $ProjectPath"
Write-Host "2. conda env create -f environment.yml"
Write-Host "3. conda activate $($ProjectName.Replace(' ', '-').ToLower())"
Write-Host "4. jupyter notebook`n"

Write-Host "Happy analyzing!`n" -ForegroundColor Cyan
