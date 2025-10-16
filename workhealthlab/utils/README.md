# Sociopath-it Utilities

PowerShell scripts for creating analysis projects and template notebooks. Because setting up project structures manually is for people with too much time.

## Scripts

### `workhealthlab-init.ps1` - Main Utility
Quick start wrapper for creating projects and notebooks.

**Usage:**
```powershell
# Create a project structure
.\workhealthlab-init.ps1 -Action project -Path "C:\Research\MyStudy"

# Create a notebook
.\workhealthlab-init.ps1 -Action notebook -Path "analysis.ipynb" -Type regression

# Create both (project + starter notebook)
.\workhealthlab-init.ps1 -Action both -Path "C:\Research\NewProject" -Type general
```

**Parameters:**
- `-Action`: What to create (`project`, `notebook`, or `both`)
- `-Path`: Where to create files/directories
- `-Type`: Notebook template type (see below)
- `-ProjectName`: Optional project name (for READMEs)

---

### `create-project.ps1` - Project Structure Generator
Creates a complete research project directory structure.

**What it creates:**
```
MyStudy/
├── data/
│   ├── raw/              # Original data (gitignored)
│   ├── processed/        # Cleaned data (gitignored)
│   └── metadata/         # Codebooks and documentation
├── notes/
│   ├── literature/       # Literature reviews
│   ├── codebooks/        # Variable coding schemes
│   └── meetings/         # Meeting notes
├── analyses/
│   ├── descriptive/      # EDA notebooks
│   ├── models/           # Statistical models
│   └── figures/          # Saved visualizations
├── output/
│   ├── tables/           # Publication tables
│   ├── figures/          # Final figures
│   └── reports/          # Manuscripts
├── README.md             # Project documentation
├── .gitignore            # Privacy-first git config
├── environment.yml       # Conda environment
└── requirements.txt      # Pip dependencies
```

**Usage:**
```powershell
# Basic project
.\create-project.ps1 -ProjectPath "C:\Research\IncomeInequality"

# With custom name
.\create-project.ps1 -ProjectPath "C:\Research\Study2024" -ProjectName "Wage Gap Study"

# Include starter notebooks
.\create-project.ps1 -ProjectPath "C:\Research\NewStudy" -IncludeNotebooks
```

**Features:**
- Complete directory structure with READMEs
- Pre-configured .gitignore (protects data privacy)
- Conda environment.yml with all dependencies
- requirements.txt for pip users
- Project-specific documentation templates

---

### `create-notebook.ps1` - Notebook Template Generator
Creates structured Jupyter notebooks for different analysis types.

**Notebook Types:**

#### 1. `general` - General Purpose Analysis
Standard workflow: load → prep → analyze → visualize → export

**Sections:**
- Setup & Configuration
- Data Loading
- Data Preparation
- Exploratory Analysis
- Statistical Analysis
- Visualization
- Export Results

**Use for:** Most analyses, quick explorations, standard workflows

---

#### 2. `regression` - Regression Analysis
Complete regression workflow with diagnostics.

**Sections:**
- Setup & Data Loading
- Exploratory Data Analysis
- Model Specification
- Model Estimation (multiple models)
- Model Diagnostics (residuals, VIF)
- Visualization & Interpretation
- Export Results

**Includes:**
- OLS/logit/Poisson models
- Model comparison tables
- Residual diagnostics
- Coefficient plots
- Marginal effects

**Use for:** OLS, logistic, count models, model comparisons

---

#### 3. `textual` - Text Analysis & NLP
NLP and text mining workflows.

**Sections:**
- Setup & Data Loading
- Text Preprocessing
- Descriptive Text Analysis
- Topic Modeling
- Sentiment Analysis
- Similarity & Clustering
- Visualization & Export

**Includes:**
- Text cleaning and tokenization
- TF-IDF matrices
- LDA topic models
- Sentiment analysis (lexicon & transformer)
- N-gram analysis
- Word clouds

**Use for:** Survey open-ends, document analysis, social media text

---

#### 4. `descriptive` - Exploratory Data Analysis
Comprehensive EDA template.

**Sections:**
- Setup & Data Loading
- Data Quality Check
- Univariate Analysis
- Bivariate Analysis
- Multivariate Patterns
- Visualization Dashboard
- Summary & Next Steps

**Includes:**
- Descriptive statistics
- Distribution plots
- Crosstabs and group comparisons
- Correlation matrices
- Pair plots and heatmaps

**Use for:** Initial data exploration, data quality checks, pattern discovery

---

#### 5. `causal` - Causal Inference
Propensity score and causal analysis workflows.

**Sections:**
- Setup & Data Loading
- Treatment Assignment & Balance
- Propensity Score Estimation
- Matching or Weighting
- Treatment Effect Estimation
- Sensitivity Analysis
- Results & Interpretation

**Includes:**
- Propensity score estimation
- IPW (inverse probability weighting)
- Balance diagnostics
- ATE estimation
- Difference-in-differences

**Use for:** Treatment effects, policy evaluation, quasi-experiments

---

#### 6. `longitudinal` - Panel Data Analysis
Repeated measures and panel data workflows.

**Sections:**
- Setup & Data Loading
- Panel Structure Verification
- Descriptive Panel Statistics
- Within-Person Change
- Fixed Effects Models
- Dynamic Models
- Results & Interpretation

**Includes:**
- Panel structure checks
- Within/between variation decomposition
- Fixed effects regression
- First-difference models
- Individual trajectories

**Use for:** Longitudinal surveys, panel studies, repeated measures

---

**Usage:**
```powershell
# Create a regression notebook
.\create-notebook.ps1 -NotebookPath "regression_models.ipynb" -NotebookType regression

# Create in specific directory
.\create-notebook.ps1 -NotebookPath "C:\Research\analyses\text_analysis.ipynb" -NotebookType textual

# Default type (general)
.\create-notebook.ps1 -NotebookPath "analysis.ipynb"
```

---

## Quick Start Examples

### Example 1: New Research Project
```powershell
# Navigate to where you want the project
cd C:\Research

# Create full project with regression notebook
.\workhealthlab-init.ps1 -Action both -Path "WageGapStudy" -Type regression

# Start working
cd WageGapStudy
conda env create -f environment.yml
conda activate wagegapstudy
jupyter notebook
```

### Example 2: Add Notebook to Existing Project
```powershell
# Already have a project, add text analysis notebook
cd C:\Research\ExistingProject
.\create-notebook.ps1 -NotebookPath "analyses\text_analysis.ipynb" -NotebookType textual
```

### Example 3: Multiple Notebooks for Different Stages
```powershell
cd C:\Research\MyStudy

# EDA notebook
.\create-notebook.ps1 -NotebookPath "analyses\01_exploratory.ipynb" -Type descriptive

# Main analysis
.\create-notebook.ps1 -NotebookPath "analyses\02_regression.ipynb" -Type regression

# Causal analysis
.\create-notebook.ps1 -NotebookPath "analyses\03_causal.ipynb" -Type causal
```

---

## Tips & Best Practices

### Project Organization
- Keep raw data in `data/raw/` and never modify it directly
- Store processed data in `data/processed/`
- Use descriptive filenames with dates: `survey_wave1_2023-01-15.csv`
- Document all transformations in notebooks

### Notebook Workflow
1. Start with `descriptive` template for initial EDA
2. Move to `regression` or `causal` for main analysis
3. Use `textual` for any open-ended text data
4. Use `longitudinal` if you have panel structure

### Version Control
- The generated .gitignore protects data privacy by default
- Commit notebooks with outputs cleared (except final versions)
- Use meaningful commit messages
- Don't commit large data files

### Environment Management
```powershell
# Create conda environment from project
conda env create -f environment.yml

# Or use pip
pip install -r requirements.txt

# Update environment file after adding packages
conda env export > environment.yml
```

---

## Customization

All templates use Sociopath-it functions but you can customize them:

1. **Edit templates directly** - Modify the switch cases in `create-notebook.ps1`
2. **Add new notebook types** - Add new cases to the switch statement
3. **Modify project structure** - Edit directory lists in `create-project.ps1`
4. **Change default imports** - Update the setup cells in each template

---

## Troubleshooting

**Script won't run:**
```powershell
# May need to allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Notebook won't open:**
- Ensure .ipynb extension is added
- Check that directory exists
- Use full paths if having issues

**Missing dependencies:**
```powershell
# Install from environment file
conda env create -f environment.yml

# Or install Sociopath-it manually
pip install git+https://github.com/AlexanderTheAlright/workhealthlab.git
```

---

## Philosophy

These scripts embody the Sociopath-it philosophy:
- **Structured but flexible** - Templates give structure, you add substance
- **Privacy-first** - Data files gitignored by default
- **Reproducible** - Environment files ensure consistent setups
- **Practical** - No over-engineering, just what you need to start analyzing

Now go forth and generate some beautiful, reproducible research.
