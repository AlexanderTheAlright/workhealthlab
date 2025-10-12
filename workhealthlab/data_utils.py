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
    """Locate the QWELS directory relative to the current working directory."""

    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
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

def discover_dta_files(
    data_dir: Optional[Path] = None,
    exclude: Optional[List[str]] = None,
) -> List[Path]:
    """Return a sorted list of ``.dta`` files, excluding filenames in ``exclude``."""

    data_dir = Path(data_dir) if data_dir else get_data_dir()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    exclude = set(exclude or [])
    files = [f for f in data_dir.glob("*.dta") if f.stem not in exclude]
    return sorted(files)


def summarize_datasets(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Return a summary table of available datasets (always as a DataFrame)."""

    data_dir = Path(data_dir) if data_dir else get_data_dir()
    files = discover_dta_files(data_dir)
    summary = []
    for f in files:
        size_mb = round(f.stat().st_size / (1024 ** 2), 2)
        summary.append({"dataset": f.stem, "size_MB": size_mb, "path": str(f)})

    df = pd.DataFrame(summary).sort_values("dataset")

    try:  # Optional pretty display in notebook environments
        from IPython.display import display  # type: ignore

        display(df)
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

def find_variable_across_datasets(
    var_name: str,
    data_dir: Optional[Path] = None,
    exclude: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Search ``.dta`` files for the presence of ``var_name`` (case-insensitive)."""

    data_dir = Path(data_dir) if data_dir else get_data_dir()
    var_name = var_name.lower()
    matches = []
    for f in discover_dta_files(data_dir, exclude):
        try:
            reader = pd.io.stata.StataReader(str(f))
            cols = [c.lower() for c in reader.varlist]
            reader.close()
            if var_name in cols:
                matches.append({"dataset": f.stem, "path": str(f)})
        except Exception:
            continue
    return pd.DataFrame(matches)



# =============================================================================
# QUICK DISCOVERY + LOAD PIPELINE
# =============================================================================

def discover_data(
    data_dir: Optional[Path] = None,
    construct_names: Optional[List[str]] = None,
    predictors: Optional[List[str]] = None,
    exclude_list: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Scan directory for .dta files containing any of the construct or predictor vars.
    """
    if construct_names is None or predictors is None:
        raise ValueError("construct_names and predictors must be provided.")

    data_dir = Path(data_dir) if data_dir else get_data_dir()
    exclude_list = exclude_list or []
    relevant_surveys = {}

    print(f"🔎 Scanning {data_dir} for *.dta files...")
    files = discover_dta_files(data_dir, exclude_list)
    if not files:
        print("⚠️ No .dta files found.")
        return {}

    constructs = {c.lower() for c in construct_names}
    predictor_set = {p.lower() for p in predictors}

    for f in files:
        try:
            df = pd.read_stata(f, convert_categoricals=True)
            df.columns = [c.lower() for c in df.columns]
            if constructs.intersection(df.columns) and predictor_set.intersection(df.columns):
                relevant_surveys[f.stem] = df
                print(f"  ✓ Loaded: {f.stem} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ❌ {f.stem} failed: {e}")

    print(f"\n✅ Loaded {len(relevant_surveys)} relevant surveys.")
    return relevant_surveys
