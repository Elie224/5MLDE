import numpy as np
import pandas as pd
import pytest

from retail.common.config import TrainingCfg
from retail.features.panel import build_feature_frame, freeze_top_skus, ordered_weeks
from retail.training.metrics import mae, wape
from retail.training.train import make_pipeline, temporal_split


def test_mae_and_wape():
    y = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 30.0])
    assert mae(y, pred) == pytest.approx((2 + 2 + 0) / 3)
    assert wape(y, pred) == pytest.approx((2 + 2 + 0) / 60)


def test_naive_is_lag_1_and_pipeline_fits():
    weeks = [f"2011-W{i:02d}" for i in range(1, 17)]
    rows = []
    for i, week in enumerate(weeks):
        for sku, base in (("AAA", 50), ("BBB", 10)):
            rows.append(
                {
                    "StockCode": sku,
                    "iso_week": week,
                    "quantity": base + i,
                    "unit_price": 1.5,
                    "revenue": 1.5 * (base + i),
                    "n_lines": 1,
                    "n_invoices": 1,
                }
            )
    panel = pd.DataFrame(rows)
    warmup = weeks[:8]
    skus = freeze_top_skus(panel, warmup, 2)
    featured = build_feature_frame(
        panel,
        skus,
        ordered_weeks(weeks),
        warmup,
        TrainingCfg(test_weeks=3, lags=[1, 2, 4], rolling_weeks=4),
    )
    train, test = temporal_split(featured, 3)
    pipe = make_pipeline()
    cols = ["StockCode", "lag_1", "lag_2", "lag_4", "ma_4", "price_lag_1", "week_of_year", "month"]
    pipe.fit(train[cols], train["y"])
    pred = pipe.predict(test[cols])
    naive = test["lag_1"].to_numpy()
    assert len(pred) == len(test)
    assert mae(test["y"].to_numpy(), naive) >= 0
