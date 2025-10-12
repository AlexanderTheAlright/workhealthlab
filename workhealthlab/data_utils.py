# workhealthlab/
"""
WorkHealthLab Data Utilities
----------------------------
Tools for discovering, loading, and describing QWELS-based survey data (.dta files).

Designed for use in Jupyter and relative to the QWELS folder structure.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Union

warnings.filterwarnings("ignore")

# =============================================================================
# PATH RESOLUTION
# =============================================================================

def resolve_qwels_root() -> Path:
    """Locate the QWELS directory relative to current working dir."""
    cwd = Path.cwd()
    for parent in cwd.parents:
        if (parent / "ANALYSIS" / "DATASETS").exists():
            return parent
    raise FileNotFoundError("❌ Could not locate QWELS directory (expecting 'ANALYSIS/DATASETS').")

def get_data_dir() -> Path:
    """Return the path to QWELS/ANALYSIS/DATASETS."""
    qwels = resolve_qwels_root()
    return qwels / "ANALYSIS" / "DATASETS"


# =============================================================================
# DATA DISCOVERY
# =============================================================================

def discover_dta_files(data_dir: Optional[Path] = None, exclude: Optional[List[str]] = None) -> List[Path]:
    """Return a list of all .dta files, excluding those in the exclude list."""
    data_dir = data_dir or get_data_dir()
    exclude = exclude or []
    files = [f for f in data_dir.glob("*.dta") if f.stem not in exclude]
    return sorted(files)


def summarize_datasets(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Return a searchable summary table of available datasets."""
    data_dir = data_dir or get_data_dir()
    files = discover_dta_files(data_dir)
    summary = []
    for f in files:
        size_mb = round(f.stat().st_size / (1024 ** 2), 2)
        summary.append({"dataset": f.stem, "size_MB": size_mb, "path": str(f)})
    df = pd.DataFrame(summary).sort_values("dataset")

    # Jupyter display
    try:
        import qgrid
        return qgrid.show_grid(df, show_toolbar=True)
    except Exception:
        try:
            from IPython.display import display
            from ipywidgets import interact
            @interact(filter="")
            def _show(filter=""):
                subset = df[df["dataset"].str.contains(filter, case=False, na=False)]
                display(subset)
        except Exception:
            pass
        return df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_dta(path: Union[str, Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Robust .dta loader preserving labels when possible.
    Falls back gracefully between pyreadstat and pandas.
    """
    path = Path(path)
    try:
        import pyreadstat
        df, meta = pyreadstat.read_dta(path, apply_value_formats=True)
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        print(f"✅ Loaded with pyreadstat: {path.name}")
        return df
    except ModuleNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ pyreadstat failed: {e}")

    # Fallback
    try:
        df = pd.read_stata(path, columns=columns, convert_categoricals=True)
        print(f"✅ Loaded with pandas.read_stata: {path.name}")
        return df
    except Exception as e:
        print(f"❌ Could not load {path.name}: {e}")
        raise


# =============================================================================
# VARIABLE SEARCH
# =============================================================================

def find_variable_across_datasets(var_name: str, data_dir: Optional[Path] = None, exclude: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Search all .dta files for presence of a variable.
    Returns a summary DataFrame with matches.
    """
    data_dir = data_dir or get_data_dir()
    matches = []
    for f in discover_dta_files(data_dir, exclude):
        try:
            cols = pd.io.stata.StataReader(f).variables()
            if any(var_name.lower() == c.lower() for c in cols):
                matches.append({"dataset": f.stem, "path": str(f)})
        except Exception:
            continue
    return pd.DataFrame(matches)


# =============================================================================
# QUICK DISCOVERY + LOAD PIPELINE
# =============================================================================

def discover_data(
    data_dir: Optional[Path],
    construct_names: List[str],
    predictors: List[str],
    exclude_list: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Scan directory for .dta files containing any of the construct or predictor vars.
    """
    data_dir = data_dir or get_data_dir()
    exclude_list = exclude_list or []
    relevant_surveys = {}

    print(f"🔎 Scanning {data_dir} for *.dta files...")
    files = discover_dta_files(data_dir, exclude_list)
    if not files:
        print("⚠️ No .dta files found.")
        return {}

    for f in files:
        try:
            df = pd.read_stata(f, convert_categoricals=True)
            df.columns = [c.lower() for c in df.columns]
            if any(c in df.columns for c in construct_names) and any(p in df.columns for p in predictors):
                relevant_surveys[f.stem] = df
                print(f"  ✓ Loaded: {f.stem} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ❌ {f.stem} failed: {e}")

    print(f"\n✅ Loaded {len(relevant_surveys)} relevant surveys.")
    return relevant_surveys
