"""Tests for the dashboard module loaders (uses production artifacts)."""
import json, os, sys
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
    """All standardized artifacts must be present."""
    required = [
        "item_item_cosine.pkl", "surprise_svd.pkl", "hybrid_lightfm_state.pt",
        "movies.csv", "metadata.json", "manifest.json",
        "user_index.json", "item_index.json",
        "user_features.npy", "item_features.npy",
        "user_feat_vocab.json", "item_feat_vocab.json",
    ]
    for name in required:
        assert (ART / name).exists(), f"missing {name}"


def test_dashboard_load_models():
    from app.dashboard import load_models
    result = load_models()
    cos, svd, lfm_state, movies, meta = result[:5]
    assert cos.item_similarity_.shape[0] > 0
    assert svd.algo is not None
    assert len(lfm_state["model_state"]) > 0
    assert len(movies) > 3000
    assert "n_users" in meta


def test_dashboard_build_lightfm():
    from app.dashboard import load_models, build_lightfm
    result = load_models()
    lfm_state = result[2]
    user_features = result[6]
    item_features = result[7]
    user_index = result[8]
    item_index = result[9]
    user_vocab = result[10]
    item_vocab = result[11]
    lfm = build_lightfm(lfm_state, user_features, item_features, user_index, item_index, user_vocab, item_vocab)
    assert lfm.model is not None


def test_dashboard_cold_start_recommends():
    from app.dashboard import load_models, build_lightfm
    result = load_models()
    lfm = build_lightfm(result[2], result[6], result[7], result[8], result[9], result[10], result[11])
    recs = lfm.recommend_for_new_user(
        [("gender", "F"), ("age", "25"), ("occupation", "10")],
        top_k=5,
    )
    assert len(recs) == 5
    for iid, score in recs:
        assert isinstance(iid, int)
        assert 1.0 <= score <= 5.0
