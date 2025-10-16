<#
.SYNOPSIS
    Create a template Jupyter notebook for Sociopath-it analyses.

.DESCRIPTION
    Generates structured Jupyter notebooks with sections for data loading,
    preparation, analysis, and visualization. Choose from several templates
    optimized for different analysis types.

.PARAMETER NotebookPath
    Full path where the notebook will be created (include .ipynb extension).

.PARAMETER NotebookType
    Type of notebook template to create:
    - general: Standard workflow (load, prep, analyze, visualize)
    - regression: Regression analysis template
    - textual: Text analysis and NLP template
    - descriptive: Exploratory data analysis template
    - causal: Causal inference template
    - longitudinal: Panel/longitudinal data template

.EXAMPLE
    .\create-notebook.ps1 -NotebookPath "C:\Research\analysis.ipynb" -NotebookType "general"

.EXAMPLE
    .\create-notebook.ps1 -NotebookPath ".\regression_models.ipynb" -NotebookType "regression"

.NOTES
    Author: Sociopath-it
    Part of the Sociopath-it package utilities
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$NotebookPath,

    [Parameter(Mandatory=$false)]
    [ValidateSet("general", "regression", "textual", "descriptive", "causal", "longitudinal")]
    [string]$NotebookType = "general"
)

# Resolve full path
$NotebookPath = [System.IO.Path]::GetFullPath($NotebookPath)

# Ensure .ipynb extension
if ($NotebookPath -notmatch '\.ipynb$') {
    $NotebookPath = $NotebookPath + ".ipynb"
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Sociopath-it Notebook Generator" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Creating $NotebookType notebook at:" -ForegroundColor Yellow
Write-Host "$NotebookPath`n" -ForegroundColor Yellow

# Check if file exists
if (Test-Path $NotebookPath) {
    Write-Host "[!] Notebook already exists. Overwrite? (y/n): " -NoNewline -ForegroundColor Yellow
    $response = Read-Host
    if ($response -ne "y") {
        Write-Host "Cancelled." -ForegroundColor Red
        exit
    }
}

# Create directory if it doesn't exist
$directory = Split-Path -Parent $NotebookPath
if (!(Test-Path $directory)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
}

# Helper function to create notebook JSON structure
function New-NotebookCell {
    param(
        [string]$CellType,
        [string[]]$Source
    )

    return @{
        cell_type = $CellType
        metadata = @{}
        source = $Source
    }
}

# Define templates based on notebook type
$cells = @()

switch ($NotebookType) {
    "general" {
        # Title
        $cells += New-NotebookCell "markdown" @(
            "# Sociopath-it Analysis Notebook`n",
            "`n",
            "A general-purpose template for sociological data analysis.`n",
            "`n",
            "## Sections`n",
            "1. Setup & Configuration`n",
            "2. Data Loading`n",
            "3. Data Preparation`n",
            "4. Exploratory Analysis`n",
            "5. Statistical Analysis`n",
            "6. Visualization`n",
            "7. Export Results"
        )

        # Setup
        $cells += New-NotebookCell "markdown" @("## 1. Setup & Configuration")
        $cells += New-NotebookCell "code" @(
            "import numpy as np`n",
            "import pandas as pd`n",
            "import matplotlib.pyplot as plt`n",
            "import seaborn as sns`n",
            "`n",
            "# Sociopath-it imports`n",
            "from workhealthlab.data.loading import load_all_surveys`n",
            "from workhealthlab.data.preparation import prepare_for_analysis`n",
            "from workhealthlab.analyses.descriptive import crosstab, group_summary`n",
            "from workhealthlab.analyses.regress import ols, logit`n",
            "from workhealthlab.visuals.bar import bar`n",
            "from workhealthlab.visuals.coef import coef`n",
            "`n",
            "# Display settings`n",
            "pd.set_option('display.max_columns', None)`n",
            "pd.set_option('display.width', None)`n",
            "`n",
            "print('✓ Packages loaded successfully')"
        )

        # Data Loading
        $cells += New-NotebookCell "markdown" @("## 2. Data Loading")
        $cells += New-NotebookCell "code" @(
            "# Load your data`n",
            "# Option 1: Single file`n",
            "# df = pd.read_csv('data/raw/survey_data.csv')`n",
            "`n",
            "# Option 2: Multiple surveys`n",
            "# surveys = load_all_surveys('data/raw/', target_vars=['age', 'income', 'education'])`n",
            "`n",
            "# For now, create sample data`n",
            "np.random.seed(42)`n",
            "n = 500`n",
            "df = pd.DataFrame({`n",
            "    'id': range(1, n+1),`n",
            "    'age': np.random.randint(18, 75, n),`n",
            "    'income': np.random.randint(20000, 150000, n),`n",
            "    'education': np.random.choice(['High School', 'Some College', 'Bachelor', 'Graduate'], n),`n",
            "    'satisfaction': np.random.randint(1, 11, n)`n",
            "})`n",
            "`n",
            "print(f'Loaded {len(df)} observations')`n",
            "df.head()"
        )

        # Data Preparation
        $cells += New-NotebookCell "markdown" @("## 3. Data Preparation")
        $cells += New-NotebookCell "code" @(
            "# Check data structure`n",
            "print('Data shape:', df.shape)`n",
            "print('\\nColumn types:')`n",
            "print(df.dtypes)`n",
            "print('\\nMissing values:')`n",
            "print(df.isnull().sum())"
        )

        $cells += New-NotebookCell "code" @(
            "# Data cleaning and preparation`n",
            "# df_clean = prepare_for_analysis(`n",
            "#     df,`n",
            "#     satisfaction_vars=['satisfaction'],`n",
            "#     min_valid_pct=0.6`n",
            "# )`n",
            "`n",
            "df_clean = df.copy()`n",
            "print(f'Clean data: {len(df_clean)} observations')"
        )

        # Exploratory Analysis
        $cells += New-NotebookCell "markdown" @("## 4. Exploratory Analysis")
        $cells += New-NotebookCell "code" @(
            "# Descriptive statistics`n",
            "df_clean.describe()"
        )

        $cells += New-NotebookCell "code" @(
            "# Crosstabs and group comparisons`n",
            "ct = crosstab(df_clean, 'education', 'satisfaction')`n",
            "print(ct)"
        )

        # Statistical Analysis
        $cells += New-NotebookCell "markdown" @("## 5. Statistical Analysis")
        $cells += New-NotebookCell "code" @(
            "# Regression model`n",
            "model = ols(`n",
            "    df=df_clean,`n",
            "    outcome='satisfaction',`n",
            "    predictors=['age', 'income']`n",
            ")`n",
            "`n",
            "print(model.summary())"
        )

        # Visualization
        $cells += New-NotebookCell "markdown" @("## 6. Visualization")
        $cells += New-NotebookCell "code" @(
            "# Coefficient plot`n",
            "coef(`n",
            "    df=model.get_estimates(),`n",
            "    title='Predictors of Satisfaction',`n",
            "    style_mode='viridis'`n",
            ")"
        )

        $cells += New-NotebookCell "code" @(
            "# Distribution plots`n",
            "from workhealthlab.visuals.hist import histogram`n",
            "`n",
            "histogram(`n",
            "    df=df_clean,`n",
            "    x='satisfaction',`n",
            "    bins=10,`n",
            "    title='Distribution of Satisfaction Scores',`n",
            "    style_mode='viridis'`n",
            ")"
        )

        # Export
        $cells += New-NotebookCell "markdown" @("## 7. Export Results")
        $cells += New-NotebookCell "code" @(
            "# Export tables`n",
            "# model.get_estimates().to_csv('output/tables/regression_results.csv', index=False)`n",
            "`n",
            "# Export figures`n",
            "# plt.savefig('output/figures/satisfaction_distribution.png', dpi=300, bbox_inches='tight')`n",
            "`n",
            "print('✓ Results exported')"
        )
    }

    "regression" {
        # Title
        $cells += New-NotebookCell "markdown" @(
            "# Regression Analysis`n",
            "`n",
            "Template for statistical modeling and regression analysis.`n",
            "`n",
            "## Workflow`n",
            "1. Setup & Data Loading`n",
            "2. Exploratory Data Analysis`n",
            "3. Model Specification`n",
            "4. Model Estimation`n",
            "5. Model Diagnostics`n",
            "6. Visualization & Interpretation`n",
            "7. Export Results"
        )

        # Setup
        $cells += New-NotebookCell "markdown" @("## 1. Setup & Data Loading")
        $cells += New-NotebookCell "code" @(
            "import numpy as np`n",
            "import pandas as pd`n",
            "import matplotlib.pyplot as plt`n",
            "`n",
            "from workhealthlab.analyses.regress import ols, logit, poisson, compare_models`n",
            "from workhealthlab.analyses.pubtable import regression_table`n",
            "from workhealthlab.visuals.coef import coef, coef_interactive`n",
            "from workhealthlab.visuals.margins import margins, margins_comparison`n",
            "from workhealthlab.visuals.residuals import residuals`n",
            "`n",
            "# Load data`n",
            "# df = pd.read_csv('data/processed/analysis_data.csv')`n",
            "`n",
            "# Sample data for demonstration`n",
            "np.random.seed(42)`n",
            "n = 1000`n",
            "df = pd.DataFrame({`n",
            "    'outcome': np.random.normal(50, 10, n),`n",
            "    'predictor1': np.random.normal(0, 1, n),`n",
            "    'predictor2': np.random.normal(0, 1, n),`n",
            "    'control': np.random.choice(['A', 'B', 'C'], n)`n",
            "})`n",
            "`n",
            "df['outcome'] = 30 + 5*df['predictor1'] - 3*df['predictor2'] + np.random.normal(0, 5, n)`n",
            "`n",
            "print(f'Data loaded: {df.shape[0]} observations')"
        )

        # EDA
        $cells += New-NotebookCell "markdown" @("## 2. Exploratory Data Analysis")
        $cells += New-NotebookCell "code" @(
            "# Descriptive statistics`n",
            "print('Descriptive Statistics:')`n",
            "print(df.describe())`n",
            "`n",
            "# Correlation matrix`n",
            "print('\\nCorrelations:')`n",
            "print(df[['outcome', 'predictor1', 'predictor2']].corr())"
        )

        # Model Specification
        $cells += New-NotebookCell "markdown" @("## 3. Model Specification")
        $cells += New-NotebookCell "code" @(
            "# Define model variables`n",
            "outcome_var = 'outcome'`n",
            "predictors = ['predictor1', 'predictor2']`n",
            "controls = ['control']`n",
            "`n",
            "print(f'Outcome: {outcome_var}')`n",
            "print(f'Predictors: {predictors}')`n",
            "print(f'Controls: {controls}')"
        )

        # Model Estimation
        $cells += New-NotebookCell "markdown" @("## 4. Model Estimation")
        $cells += New-NotebookCell "code" @(
            "# Model 1: Bivariate`n",
            "model1 = ols(df, outcome_var, ['predictor1'])`n",
            "print('Model 1: Bivariate')`n",
            "print(model1.summary())`n",
            "print('\\n' + '='*60 + '\\n')"
        )

        $cells += New-NotebookCell "code" @(
            "# Model 2: Multivariate`n",
            "model2 = ols(df, outcome_var, predictors)`n",
            "print('Model 2: Multivariate')`n",
            "print(model2.summary())`n",
            "print('\\n' + '='*60 + '\\n')"
        )

        $cells += New-NotebookCell "code" @(
            "# Model 3: Full model with controls`n",
            "model3 = ols(df, outcome_var, predictors + controls)`n",
            "print('Model 3: Full Model')`n",
            "print(model3.summary())`n",
            "print('\\n' + '='*60 + '\\n')"
        )

        $cells += New-NotebookCell "code" @(
            "# Model comparison`n",
            "comparison = compare_models([model1, model2, model3])`n",
            "print('Model Comparison:')`n",
            "print(comparison)"
        )

        # Diagnostics
        $cells += New-NotebookCell "markdown" @("## 5. Model Diagnostics")
        $cells += New-NotebookCell "code" @(
            "# Residual diagnostics`n",
            "residuals(`n",
            "    model=model3,`n",
            "    plot_type='all',`n",
            "    title='Model 3 Diagnostics',`n",
            "    style_mode='viridis'`n",
            ")"
        )

        $cells += New-NotebookCell "code" @(
            "# Check for multicollinearity`n",
            "print('Variance Inflation Factors:')`n",
            "vif = model3.vif()`n",
            "print(vif)"
        )

        # Visualization
        $cells += New-NotebookCell "markdown" @("## 6. Visualization & Interpretation")
        $cells += New-NotebookCell "code" @(
            "# Coefficient plot`n",
            "coef(`n",
            "    df=model3.get_estimates(),`n",
            "    title='Regression Coefficients',`n",
            "    subtitle='Full model with 95% confidence intervals',`n",
            "    style_mode='viridis'`n",
            ")"
        )

        $cells += New-NotebookCell "code" @(
            "# Marginal effects`n",
            "margins(`n",
            "    model=model3,`n",
            "    variable='predictor1',`n",
            "    title='Marginal Effect of Predictor 1',`n",
            "    style_mode='viridis'`n",
            ")"
        )

        # Export
        $cells += New-NotebookCell "markdown" @("## 7. Export Results")
        $cells += New-NotebookCell "code" @(
            "# Create publication table`n",
            "pub_table = regression_table([model1, model2, model3], model_names=['Model 1', 'Model 2', 'Model 3'])`n",
            "print(pub_table)`n",
            "`n",
            "# Export`n",
            "# pub_table.to_csv('output/tables/regression_table.csv', index=True)`n",
            "# model3.get_estimates().to_csv('output/tables/model3_estimates.csv', index=False)`n",
            "`n",
            "print('\\n✓ Analysis complete')"
        )
    }

    "textual" {
        # Title
        $cells += New-NotebookCell "markdown" @(
            "# Text Analysis & NLP`n",
            "`n",
            "Template for textual analysis and natural language processing.`n",
            "`n",
            "## Workflow`n",
            "1. Setup & Data Loading`n",
            "2. Text Preprocessing`n",
            "3. Descriptive Text Analysis`n",
            "4. Topic Modeling`n",
            "5. Sentiment Analysis`n",
            "6. Similarity & Clustering`n",
            "7. Visualization & Export"
        )

        # Setup
        $cells += New-NotebookCell "markdown" @("## 1. Setup & Data Loading")
        $cells += New-NotebookCell "code" @(
            "import numpy as np`n",
            "import pandas as pd`n",
            "import matplotlib.pyplot as plt`n",
            "`n",
            "from workhealthlab.analyses.text_analysis import (`n",
            "    clean_text, tokenize, complexity_scores,`n",
            "    create_tfidf_matrix, TopicModel,`n",
            "    extract_ngrams, ngram_frequency,`n",
            "    SentimentAnalyzer, jaccard_similarity`n",
            ")`n",
            "from workhealthlab.visuals.wordcloud import wordcloud`n",
            "from workhealthlab.visuals.cooccur import cooccur`n",
            "`n",
            "# Load text data`n",
            "# df = pd.read_csv('data/processed/text_data.csv')`n",
            "`n",
            "# Sample data for demonstration`n",
            "sample_texts = [`n",
            "    'Sociological research examines patterns of social behavior and cultural norms.',`n",
            "    'Inequality affects access to education, healthcare, and economic opportunities.',`n",
            "    'Cultural capital plays a crucial role in reproducing social stratification.',`n",
            "    'Social networks influence individual outcomes and community cohesion.',`n",
            "    'Power structures shape institutional practices and policy decisions.'`n",
            "]`n",
            "`n",
            "df = pd.DataFrame({'text': sample_texts, 'id': range(len(sample_texts))})`n",
            "`n",
            "print(f'Loaded {len(df)} documents')"
        )

        # Preprocessing
        $cells += New-NotebookCell "markdown" @("## 2. Text Preprocessing")
        $cells += New-NotebookCell "code" @(
            "# Clean and tokenize texts`n",
            "df['clean_text'] = df['text'].apply(clean_text)`n",
            "df['tokens'] = df['clean_text'].apply(tokenize)`n",
            "`n",
            "print('Sample cleaned text:')`n",
            "print(df['clean_text'].iloc[0])`n",
            "print('\\nTokens:')`n",
            "print(df['tokens'].iloc[0])"
        )

        # Descriptive Analysis
        $cells += New-NotebookCell "markdown" @("## 3. Descriptive Text Analysis")
        $cells += New-NotebookCell "code" @(
            "# Text complexity metrics`n",
            "for idx, row in df.iterrows():`n",
            "    scores = complexity_scores(row['text'])`n",
            "    print(f'Document {idx}:')`n",
            "    print(f\"  Flesch Reading Ease: {scores['flesch_reading_ease']:.2f}\")`n",
            "    print(f\"  Avg Sentence Length: {scores['avg_sentence_length']:.2f}\")`n",
            "    print()"
        )

        $cells += New-NotebookCell "code" @(
            "# Most common n-grams`n",
            "bigrams = ngram_frequency(df['text'].tolist(), n=2, top_k=10)`n",
            "print('Top 10 Bigrams:')`n",
            "print(bigrams)"
        )

        $cells += New-NotebookCell "code" @(
            "# TF-IDF matrix`n",
            "tfidf_matrix, feature_names, vectorizer = create_tfidf_matrix(df['text'].tolist())`n",
            "print(f'TF-IDF matrix shape: {tfidf_matrix.shape}')`n",
            "print(f'Top features: {feature_names[:10]}')"
        )

        # Topic Modeling
        $cells += New-NotebookCell "markdown" @("## 4. Topic Modeling")
        $cells += New-NotebookCell "code" @(
            "# LDA topic model`n",
            "topic_model = TopicModel(n_topics=2, method='lda', random_state=42)`n",
            "topic_model.fit(df['text'].tolist())`n",
            "`n",
            "print('Topics:')`n",
            "topics_df = topic_model.get_topics(n_words=5)`n",
            "print(topics_df)`n",
            "`n",
            "print('\\nDocument-Topic Distribution:')`n",
            "doc_topics = topic_model.get_document_topics()`n",
            "print(doc_topics)"
        )

        # Sentiment Analysis
        $cells += New-NotebookCell "markdown" @("## 5. Sentiment Analysis")
        $cells += New-NotebookCell "code" @(
            "# Sentiment analysis`n",
            "analyzer = SentimentAnalyzer(method='lexicon')`n",
            "`n",
            "df['sentiment_label'] = df['text'].apply(lambda x: analyzer.analyze(x)['label'])`n",
            "df['sentiment_score'] = df['text'].apply(lambda x: analyzer.analyze(x)['score'])`n",
            "`n",
            "print('Sentiment Analysis Results:')`n",
            "print(df[['text', 'sentiment_label', 'sentiment_score']])"
        )

        # Similarity
        $cells += New-NotebookCell "markdown" @("## 6. Similarity & Clustering")
        $cells += New-NotebookCell "code" @(
            "# Text similarity matrix`n",
            "similarity_matrix = np.zeros((len(df), len(df)))`n",
            "`n",
            "for i in range(len(df)):`n",
            "    for j in range(len(df)):`n",
            "        similarity_matrix[i, j] = jaccard_similarity(df['text'].iloc[i], df['text'].iloc[j])`n",
            "`n",
            "print('Similarity Matrix:')`n",
            "print(pd.DataFrame(similarity_matrix, index=df['id'], columns=df['id']))"
        )

        # Visualization
        $cells += New-NotebookCell "markdown" @("## 7. Visualization & Export")
        $cells += New-NotebookCell "code" @(
            "# Word cloud from TF-IDF`n",
            "# Get word frequencies for visualization`n",
            "word_freq = {}`n",
            "for i, word in enumerate(feature_names):`n",
            "    word_freq[word] = float(tfidf_matrix[:, i].sum())`n",
            "`n",
            "# Sort and get top words`n",
            "top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50])`n",
            "`n",
            "wordcloud(`n",
            "    freq_dict=top_words,`n",
            "    title='Most Important Terms (TF-IDF)',`n",
            "    style_mode='viridis',`n",
            "    max_words=50`n",
            ")"
        )

        $cells += New-NotebookCell "code" @(
            "# Export results`n",
            "# df.to_csv('output/tables/text_analysis_results.csv', index=False)`n",
            "# topics_df.to_csv('output/tables/topics.csv', index=True)`n",
            "`n",
            "print('\\n✓ Text analysis complete')"
        )
    }

    "descriptive" {
        # Title
        $cells += New-NotebookCell "markdown" @(
            "# Exploratory Data Analysis`n",
            "`n",
            "Template for descriptive statistics and exploratory analysis.`n",
            "`n",
            "## Workflow`n",
            "1. Setup & Data Loading`n",
            "2. Data Quality Check`n",
            "3. Univariate Analysis`n",
            "4. Bivariate Analysis`n",
            "5. Multivariate Patterns`n",
            "6. Visualization Dashboard`n",
            "7. Summary & Next Steps"
        )

        # Setup
        $cells += New-NotebookCell "markdown" @("## 1. Setup & Data Loading")
        $cells += New-NotebookCell "code" @(
            "import numpy as np`n",
            "import pandas as pd`n",
            "import matplotlib.pyplot as plt`n",
            "import seaborn as sns`n",
            "`n",
            "from workhealthlab.analyses.descriptive import crosstab, group_summary, correlation_matrix`n",
            "from workhealthlab.analyses.pubtable import proportion_table, descriptive_table`n",
            "from workhealthlab.visuals.hist import histogram`n",
            "from workhealthlab.visuals.boxplot import boxplot`n",
            "from workhealthlab.visuals.heatmap import heatmap`n",
            "from workhealthlab.visuals.pair import pair`n",
            "`n",
            "# Load data`n",
            "np.random.seed(42)`n",
            "n = 500`n",
            "df = pd.DataFrame({`n",
            "    'age': np.random.randint(18, 75, n),`n",
            "    'income': np.random.randint(20000, 150000, n),`n",
            "    'education': np.random.choice(['HS', 'College', 'Graduate'], n),`n",
            "    'region': np.random.choice(['Urban', 'Suburban', 'Rural'], n),`n",
            "    'satisfaction': np.random.randint(1, 11, n)`n",
            "})`n",
            "`n",
            "print(f'Loaded {len(df)} observations with {df.shape[1]} variables')"
        )

        # Data Quality
        $cells += New-NotebookCell "markdown" @("## 2. Data Quality Check")
        $cells += New-NotebookCell "code" @(
            "# Data structure`n",
            "print('Dataset shape:', df.shape)`n",
            "print('\\nData types:')`n",
            "print(df.dtypes)`n",
            "print('\\nMissing values:')`n",
            "print(df.isnull().sum())`n",
            "print('\\nFirst few rows:')`n",
            "df.head()"
        )

        # Univariate
        $cells += New-NotebookCell "markdown" @("## 3. Univariate Analysis")
        $cells += New-NotebookCell "code" @(
            "# Numeric variables`n",
            "print('Numeric Variable Summary:')`n",
            "print(df[['age', 'income', 'satisfaction']].describe())"
        )

        $cells += New-NotebookCell "code" @(
            "# Categorical variables`n",
            "print('Education Distribution:')`n",
            "print(df['education'].value_counts(normalize=True))`n",
            "print('\\nRegion Distribution:')`n",
            "print(df['region'].value_counts(normalize=True))"
        )

        $cells += New-NotebookCell "code" @(
            "# Distribution plots`n",
            "histogram(df, x='satisfaction', bins=10, title='Satisfaction Distribution', style_mode='viridis')"
        )

        # Bivariate
        $cells += New-NotebookCell "markdown" @("## 4. Bivariate Analysis")
        $cells += New-NotebookCell "code" @(
            "# Crosstab analysis`n",
            "ct = crosstab(df, 'education', 'satisfaction')`n",
            "print(ct)"
        )

        $cells += New-NotebookCell "code" @(
            "# Group comparisons`n",
            "summary = group_summary(df, group_var='education', numeric_vars=['age', 'income', 'satisfaction'])`n",
            "print(summary)"
        )

        $cells += New-NotebookCell "code" @(
            "# Boxplot comparison`n",
            "boxplot(df, x='education', y='satisfaction', title='Satisfaction by Education', style_mode='viridis')"
        )

        # Multivariate
        $cells += New-NotebookCell "markdown" @("## 5. Multivariate Patterns")
        $cells += New-NotebookCell "code" @(
            "# Correlation matrix`n",
            "corr = correlation_matrix(df, vars=['age', 'income', 'satisfaction'])`n",
            "print(corr)"
        )

        $cells += New-NotebookCell "code" @(
            "# Correlation heatmap`n",
            "heatmap(df[['age', 'income', 'satisfaction']], title='Correlation Matrix', style_mode='viridis')"
        )

        # Visualization
        $cells += New-NotebookCell "markdown" @("## 6. Visualization Dashboard")
        $cells += New-NotebookCell "code" @(
            "# Pair plot`n",
            "pair(df, vars=['age', 'income', 'satisfaction'], hue='education', style_mode='viridis')"
        )

        # Summary
        $cells += New-NotebookCell "markdown" @("## 7. Summary & Next Steps")
        $cells += New-NotebookCell "code" @(
            "print('Key Findings:')`n",
            "print('1. Sample size:', len(df))`n",
            "print('2. Average satisfaction:', df['satisfaction'].mean())`n",
            "print('3. Income range:', df['income'].min(), '-', df['income'].max())`n",
            "print('\\nNext steps: Proceed to regression analysis or causal inference')"
        )
    }

    "causal" {
        # Title
        $cells += New-NotebookCell "markdown" @(
            "# Causal Inference Analysis`n",
            "`n",
            "Template for causal analysis using propensity scores and difference-in-differences.`n",
            "`n",
            "## Workflow`n",
            "1. Setup & Data Loading`n",
            "2. Treatment Assignment & Balance`n",
            "3. Propensity Score Estimation`n",
            "4. Matching or Weighting`n",
            "5. Treatment Effect Estimation`n",
            "6. Sensitivity Analysis`n",
            "7. Results & Interpretation"
        )

        # Setup
        $cells += New-NotebookCell "markdown" @("## 1. Setup & Data Loading")
        $cells += New-NotebookCell "code" @(
            "import numpy as np`n",
            "import pandas as pd`n",
            "import matplotlib.pyplot as plt`n",
            "`n",
            "from workhealthlab.analyses.causal import propensity_score, ipw, diff_in_diff`n",
            "from workhealthlab.visuals.coef import coef`n",
            "from workhealthlab.visuals.density import kde`n",
            "`n",
            "# Load data`n",
            "np.random.seed(42)`n",
            "n = 1000`n",
            "df = pd.DataFrame({`n",
            "    'age': np.random.randint(18, 75, n),`n",
            "    'income': np.random.randint(20000, 150000, n),`n",
            "    'education': np.random.choice([0, 1], n),  # 0=Low, 1=High`n",
            "    'treatment': np.random.choice([0, 1], n),`n",
            "    'outcome': np.random.normal(50, 10, n)`n",
            "})`n",
            "`n",
            "# Create treatment effect`n",
            "df.loc[df['treatment'] == 1, 'outcome'] += 5`n",
            "`n",
            "print(f'Loaded {len(df)} observations')`n",
            "print(f\"Treatment group: {df['treatment'].sum()} ({df['treatment'].mean():.1%})\")"
        )

        # Balance Check
        $cells += New-NotebookCell "markdown" @("## 2. Treatment Assignment & Balance")
        $cells += New-NotebookCell "code" @(
            "# Check covariate balance between treatment groups`n",
            "print('Covariate Balance:')`n",
            "for var in ['age', 'income', 'education']:`n",
            "    mean_t = df[df['treatment']==1][var].mean()`n",
            "    mean_c = df[df['treatment']==0][var].mean()`n",
            "    diff = mean_t - mean_c`n",
            "    print(f'{var}: Treatment={mean_t:.2f}, Control={mean_c:.2f}, Diff={diff:.2f}')"
        )

        # Propensity Scores
        $cells += New-NotebookCell "markdown" @("## 3. Propensity Score Estimation")
        $cells += New-NotebookCell "code" @(
            "# Estimate propensity scores`n",
            "ps_result = propensity_score(`n",
            "    df=df,`n",
            "    treatment='treatment',`n",
            "    covariates=['age', 'income', 'education']`n",
            ")`n",
            "`n",
            "df['propensity_score'] = ps_result['propensity_scores']`n",
            "print('Propensity score summary:')`n",
            "print(df.groupby('treatment')['propensity_score'].describe())"
        )

        $cells += New-NotebookCell "code" @(
            "# Visualize propensity score overlap`n",
            "kde(`n",
            "    df=df,`n",
            "    x='propensity_score',`n",
            "    group='treatment',`n",
            "    title='Propensity Score Distribution by Treatment',`n",
            "    style_mode='sentiment'`n",
            ")"
        )

        # Weighting
        $cells += New-NotebookCell "markdown" @("## 4. Matching or Weighting")
        $cells += New-NotebookCell "code" @(
            "# Inverse probability weighting`n",
            "weighted_result = ipw(`n",
            "    df=df,`n",
            "    treatment='treatment',`n",
            "    outcome='outcome',`n",
            "    covariates=['age', 'income', 'education']`n",
            ")`n",
            "`n",
            "print(f\"ATE (IPW): {weighted_result['ate']:.3f}\")`n",
            "print(f\"Standard Error: {weighted_result['se']:.3f}\")`n",
            "print(f\"95% CI: [{weighted_result['ci_lower']:.3f}, {weighted_result['ci_upper']:.3f}]\")"
        )

        # Treatment Effect
        $cells += New-NotebookCell "markdown" @("## 5. Treatment Effect Estimation")
        $cells += New-NotebookCell "code" @(
            "# Naive comparison (biased)`n",
            "naive_effect = df[df['treatment']==1]['outcome'].mean() - df[df['treatment']==0]['outcome'].mean()`n",
            "print(f'Naive treatment effect: {naive_effect:.3f}')`n",
            "`n",
            "# IPW estimate (adjusted)`n",
            "print(f'IPW treatment effect: {weighted_result[\"ate\"]:.3f}')`n",
            "`n",
            "print(f'\\nDifference: {abs(naive_effect - weighted_result[\"ate\"]):.3f}')"
        )

        # Sensitivity
        $cells += New-NotebookCell "markdown" @("## 6. Sensitivity Analysis")
        $cells += New-NotebookCell "code" @(
            "# Check balance after weighting`n",
            "df['ipw_weight'] = weighted_result['weights']`n",
            "`n",
            "print('Weighted Covariate Balance:')`n",
            "for var in ['age', 'income', 'education']:`n",
            "    mean_t_w = np.average(df[df['treatment']==1][var], weights=df[df['treatment']==1]['ipw_weight'])`n",
            "    mean_c_w = np.average(df[df['treatment']==0][var], weights=df[df['treatment']==0]['ipw_weight'])`n",
            "    diff = mean_t_w - mean_c_w`n",
            "    print(f'{var}: Weighted Diff={diff:.2f}')"
        )

        # Results
        $cells += New-NotebookCell "markdown" @("## 7. Results & Interpretation")
        $cells += New-NotebookCell "code" @(
            "print('\\n' + '='*60)`n",
            "print('CAUSAL ANALYSIS SUMMARY')`n",
            "print('='*60)`n",
            "print(f'\\nAverage Treatment Effect: {weighted_result[\"ate\"]:.3f}')`n",
            "print(f'95% Confidence Interval: [{weighted_result[\"ci_lower\"]:.3f}, {weighted_result[\"ci_upper\"]:.3f}]')`n",
            "print(f'\\nInterpretation: The treatment increases the outcome by {weighted_result[\"ate\"]:.2f} units on average.')`n",
            "print('\\n✓ Causal analysis complete')"
        )
    }

    "longitudinal" {
        # Title
        $cells += New-NotebookCell "markdown" @(
            "# Longitudinal/Panel Data Analysis`n",
            "`n",
            "Template for analyzing repeated measures and panel data.`n",
            "`n",
            "## Workflow`n",
            "1. Setup & Data Loading`n",
            "2. Panel Structure Verification`n",
            "3. Descriptive Panel Statistics`n",
            "4. Within-Person Change`n",
            "5. Fixed Effects Models`n",
            "6. Dynamic Models`n",
            "7. Results & Interpretation"
        )

        # Setup
        $cells += New-NotebookCell "markdown" @("## 1. Setup & Data Loading")
        $cells += New-NotebookCell "code" @(
            "import numpy as np`n",
            "import pandas as pd`n",
            "import matplotlib.pyplot as plt`n",
            "`n",
            "from workhealthlab.data.longitudinal import detect_longitudinal, align_longitudinal_data, sort_by_wave`n",
            "from workhealthlab.analyses.panel import fixed_effects, random_effects, first_difference`n",
            "from workhealthlab.visuals.trend import trend`n",
            "from workhealthlab.visuals.coef import coef`n",
            "`n",
            "# Load panel data`n",
            "# df_long = pd.read_csv('data/processed/panel_data.csv')`n",
            "`n",
            "# Create sample panel data`n",
            "np.random.seed(42)`n",
            "n_individuals = 100`n",
            "n_waves = 5`n",
            "`n",
            "df_long = pd.DataFrame({`n",
            "    'id': np.repeat(range(1, n_individuals+1), n_waves),`n",
            "    'wave': np.tile(range(1, n_waves+1), n_individuals),`n",
            "    'year': np.tile(range(2019, 2024), n_individuals),`n",
            "    'age': np.repeat(np.random.randint(25, 65, n_individuals), n_waves) + np.tile(range(n_waves), n_individuals),`n",
            "    'income': np.repeat(np.random.randint(30000, 100000, n_individuals), n_waves) + `n",
            "              np.random.normal(0, 5000, n_individuals*n_waves),`n",
            "    'satisfaction': np.repeat(np.random.randint(1, 11, n_individuals), n_waves) + `n",
            "                    np.random.normal(0, 1, n_individuals*n_waves)`n",
            "})`n",
            "`n",
            "print(f'Panel data: {n_individuals} individuals over {n_waves} waves')`n",
            "print(f'Total observations: {len(df_long)}')"
        )

        # Panel Structure
        $cells += New-NotebookCell "markdown" @("## 2. Panel Structure Verification")
        $cells += New-NotebookCell "code" @(
            "# Check panel structure`n",
            "print('Panel Structure:')`n",
            "print(f'Unique individuals: {df_long[\"id\"].nunique()}')`n",
            "print(f'Unique waves: {df_long[\"wave\"].nunique()}')`n",
            "print(f'\\nObservations per individual:')`n",
            "print(df_long.groupby('id').size().describe())`n",
            "print(f'\\nBalanced panel: {df_long.groupby(\"id\").size().nunique() == 1}')"
        )

        # Descriptive
        $cells += New-NotebookCell "markdown" @("## 3. Descriptive Panel Statistics")
        $cells += New-NotebookCell "code" @(
            "# Overall, between, and within variation`n",
            "print('Satisfaction Statistics:')`n",
            "print(f'Overall mean: {df_long[\"satisfaction\"].mean():.2f}')`n",
            "print(f'Overall SD: {df_long[\"satisfaction\"].std():.2f}')`n",
            "`n",
            "# Between-person variation`n",
            "between_mean = df_long.groupby('id')['satisfaction'].mean()`n",
            "print(f'\\nBetween-person SD: {between_mean.std():.2f}')`n",
            "`n",
            "# Within-person variation`n",
            "df_long['satisfaction_demean'] = df_long.groupby('id')['satisfaction'].transform(lambda x: x - x.mean())`n",
            "print(f'Within-person SD: {df_long[\"satisfaction_demean\"].std():.2f}')"
        )

        $cells += New-NotebookCell "code" @(
            "# Trend over waves`n",
            "trend(`n",
            "    df=df_long.groupby('wave')['satisfaction'].mean().reset_index(),`n",
            "    x='wave',`n",
            "    y='satisfaction',`n",
            "    kind='line',`n",
            "    title='Average Satisfaction Across Waves',`n",
            "    style_mode='viridis'`n",
            ")"
        )

        # Within-Person Change
        $cells += New-NotebookCell "markdown" @("## 4. Within-Person Change")
        $cells += New-NotebookCell "code" @(
            "# Individual trajectories (sample)`n",
            "sample_ids = df_long['id'].unique()[:5]`n",
            "df_sample = df_long[df_long['id'].isin(sample_ids)]`n",
            "`n",
            "trend(`n",
            "    df=df_sample,`n",
            "    x='wave',`n",
            "    y='satisfaction',`n",
            "    group='id',`n",
            "    kind='line',`n",
            "    title='Individual Trajectories (Sample)',`n",
            "    style_mode='plainjane'`n",
            ")"
        )

        # Fixed Effects
        $cells += New-NotebookCell "markdown" @("## 5. Fixed Effects Models")
        $cells += New-NotebookCell "code" @(
            "# Fixed effects regression`n",
            "fe_model = fixed_effects(`n",
            "    df=df_long,`n",
            "    outcome='satisfaction',`n",
            "    predictors=['age', 'income'],`n",
            "    entity_id='id',`n",
            "    time_id='wave'`n",
            ")`n",
            "`n",
            "print('Fixed Effects Model:')`n",
            "print(fe_model.summary())"
        )

        $cells += New-NotebookCell "code" @(
            "# Coefficient plot`n",
            "coef(`n",
            "    df=fe_model.get_estimates(),`n",
            "    title='Fixed Effects Coefficients',`n",
            "    subtitle='Within-person estimates',`n",
            "    style_mode='viridis'`n",
            ")"
        )

        # Dynamic Models
        $cells += New-NotebookCell "markdown" @("## 6. Dynamic Models")
        $cells += New-NotebookCell "code" @(
            "# Create lagged variables`n",
            "df_long = df_long.sort_values(['id', 'wave'])`n",
            "df_long['satisfaction_lag'] = df_long.groupby('id')['satisfaction'].shift(1)`n",
            "`n",
            "# First-difference model`n",
            "fd_model = first_difference(`n",
            "    df=df_long.dropna(),`n",
            "    outcome='satisfaction',`n",
            "    predictors=['age', 'income'],`n",
            "    entity_id='id'`n",
            ")`n",
            "`n",
            "print('First-Difference Model:')`n",
            "print(fd_model.summary())"
        )

        # Results
        $cells += New-NotebookCell "markdown" @("## 7. Results & Interpretation")
        $cells += New-NotebookCell "code" @(
            "print('\\n' + '='*60)`n",
            "print('PANEL ANALYSIS SUMMARY')`n",
            "print('='*60)`n",
            "print(f'\\nPanel structure: {n_individuals} individuals, {n_waves} waves')`n",
            "print(f'Total observations: {len(df_long)}')`n",
            "print(f'\\nFixed Effects R²: {fe_model.get_stats()[\"R_squared\"]:.3f}')`n",
            "print('\\nKey findings:')`n",
            "print('- Within-person changes analyzed')`n",
            "print('- Time-invariant confounders controlled')`n",
            "print('\\n✓ Longitudinal analysis complete')"
        )
    }

    default {
        Write-Host "[!] Invalid notebook type: $NotebookType" -ForegroundColor Red
        exit 1
    }
}

# Create notebook structure
$notebook = @{
    cells = $cells
    metadata = @{
        kernelspec = @{
            display_name = "Python 3"
            language = "python"
            name = "python3"
        }
        language_info = @{
            name = "python"
            version = "3.11.0"
        }
    }
    nbformat = 4
    nbformat_minor = 5
}

# Convert to JSON and save
$json = $notebook | ConvertTo-Json -Depth 10
Set-Content -Path $NotebookPath -Value $json

Write-Host "[✓] Created $NotebookType notebook" -ForegroundColor Green
Write-Host "[✓] Location: $NotebookPath`n" -ForegroundColor Green

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. jupyter notebook"
Write-Host "2. Open $([System.IO.Path]::GetFileName($NotebookPath))"
Write-Host "3. Start analyzing!`n"
