# Sociopath-it Quick Start

One-page reference for getting started fast.

## Installation
```bash
pip install git+https://github.com/AlexanderTheAlright/workhealthlab.git
```

## Create New Project (One Command)
```powershell
cd C:\Research
.\workhealthlab-init.ps1 -Action both -Path "MyStudy" -Type general
cd MyStudy
conda env create -f environment.yml
conda activate mystudy
jupyter notebook
```

## Notebook Types Cheat Sheet

| Type | Use When | Key Features |
|------|----------|--------------|
| `general` | Standard analysis | Load → Clean → Analyze → Visualize |
| `regression` | OLS/logit/Poisson | Model comparison, diagnostics, margins |
| `textual` | Text data | Topic models, sentiment, TF-IDF |
| `descriptive` | Initial exploration | EDA, distributions, correlations |
| `causal` | Treatment effects | Propensity scores, IPW, DiD |
| `longitudinal` | Panel data | Fixed effects, within/between variation |

## Common Commands

### Create Project Only
```powershell
.\create-project.ps1 -ProjectPath "C:\Research\NewStudy"
```

### Create Notebook Only
```powershell
.\create-notebook.ps1 -NotebookPath "analysis.ipynb" -NotebookType regression
```

### Multiple Notebooks
```powershell
.\create-notebook.ps1 -NotebookPath "analyses\01_eda.ipynb" -Type descriptive
.\create-notebook.ps1 -NotebookPath "analyses\02_models.ipynb" -Type regression
.\create-notebook.ps1 -NotebookPath "analyses\03_causal.ipynb" -Type causal
```

## Typical Workflow

1. **Create project**: `workhealthlab-init.ps1 -Action project -Path MyStudy`
2. **Add data**: Place files in `data/raw/`
3. **Start exploring**: Create descriptive notebook
4. **Run analysis**: Create regression/causal notebook
5. **Export results**: Tables go to `output/tables/`, figures to `output/figures/`

## Key Imports

### Data Utilities
```python
from workhealthlab.data.loading import load_all_surveys
from workhealthlab.data.preparation import prepare_for_analysis
```

### Analysis
```python
from workhealthlab.analyses.regress import ols, logit
from workhealthlab.analyses.descriptive import crosstab, group_summary
from workhealthlab.analyses.causal import propensity_score, ipw
```

### Visualization
```python
from workhealthlab.visuals.coef import coef
from workhealthlab.visuals.margins import margins
from workhealthlab.visuals.heatmap import heatmap
```

## Project Structure
```
MyStudy/
├── data/raw/          # Your original data (gitignored)
├── data/processed/    # Cleaned data (gitignored)
├── analyses/          # Your notebooks
├── output/tables/     # Exported tables
└── output/figures/    # Exported plots
```

## Style Modes
- `viridis` - Default perceptually uniform gradient
- `sentiment` - Red (negative) to green (positive)
- `fiery` - Warm gradient for emphasis
- `plainjane` - Simple greys and blues

## Tips
- Never modify `data/raw/` - keep original data immutable
- Clear notebook outputs before committing (except finals)
- Use descriptive filenames with dates
- Document every transformation
- Export final results to `output/`

## Need Help?
- Full documentation: `workhealthlab/utils/README.md`
- Test notebooks: `tests/` directory
- Package README: Root `README.md`

---

Now stop reading and start analyzing.
