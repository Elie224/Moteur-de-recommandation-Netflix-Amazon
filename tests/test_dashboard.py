"""Tests for the dashboard module loaders (uses production artifacts)."""
import json, os, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

os.environ.setdefault("SURPRISE_DATA_FOLDER", str(Path.cwd() / ".surprise_data"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ART = ROOT / "data" / "processed" / "artifacts"


def test_artifacts_exist():
    assert (ART / "cosine.pkl").exists()
    assert (ART / "svd.pkl").exists()
    assert (ART / "lightfm.pt").exists()
    assert (ART / "movies.csv").exists()
    assert (ART / "meta.json").exists()


def test_dashboard_load_models():
    from app.dashboard import load_models
    cos, svd, lfm_state, movies, meta = load_models()
    assert cos.item_similarity_.shape == (3706, 3706)
    assert svd.algo is not None
    assert len(lfm_state["model_state"]) > 0
    assert len(movies) > 3000
    assert "n_users" in meta


def test_dashboard_build_lightfm():
    from app.dashboard import load_models, build_lightfm
    _, _, lfm_state, _, meta = load_models()
    lfm, movies_df = build_lightfm(meta, lfm_state)
    assert lfm.model is not None
    assert len(movies_df) > 3000


def test_dashboard_cold_start_recommends():
    from app.dashboard import load_models, build_lightfm
    _, _, lfm_state, _, meta = load_models()
    lfm, _ = build_lightfm(meta, lfm_state)
    recs = lfm.recommend_for_new_user(
        [("gender", "F"), ("age", "25"), ("occupation", "10")],
        top_k=5,
    )
    assert len(recs) == 5
    for iid, score in recs:
        assert isinstance(iid, int)
        assert 1.0 <= score <= 5.0
