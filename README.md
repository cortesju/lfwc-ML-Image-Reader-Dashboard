# LFWC Post-Disaster AI Triage Dashboard

End-to-end workflow for classifying field photos collected after the 2025 Eaton/Altadena wildfire and converting them into a ranked decision-support product for post-disaster infrastructure triage. Built in collaboration with Las Flores Water Company.

## Problem

After a major disaster, utilities and local agencies must review hundreds of field photos collected under difficult conditions. Manual review is slow and hard to prioritize. This project automates classification and produces a ranked queue so crews can address the highest-priority sites first.

## Pipeline

```
Survey123 field collection
    → ArcGIS Pro data cleaning & photo download
    → Manual labeling of ~20 survey points
    → Two PyTorch image classifiers (meter condition + site condition)
    → Confidence scoring → PriorityScore + NeedsReview flag
    → Predictions joined back to ArcGIS Pro feature class
    → Streamlit dashboard
```

1. **Data collection** — Field crews collected GPS points and photos via Survey123, producing ~60 survey points and ~120 images.
2. **Labeling** — ~20 points labeled manually (water meters: damaged/destroyed; site locations: cleared/ruins).
3. **Model training** — Two binary image classifiers trained with PyTorch/torchvision, one per photo type.
4. **Inference** — Models applied to 21 unlabeled survey points; each prediction includes a confidence score.
5. **Prioritization** — Confidence values combined into a `PriorityScore`; low-confidence predictions flagged as `NeedsReview = 1`.
6. **Dashboard** — Interactive Streamlit app displaying predictions, priority queues, confidence levels, and supporting images on a Folium map.

## Tech stack

- Python, PyTorch, torchvision
- ArcGIS Pro / ArcPy (spatial cleaning, feature class integration)
- Streamlit, Folium, pandas
- Survey123 (field data collection)

## Files

| File | Description |
|---|---|
| `app.py` | Streamlit dashboard — map, queues, image review, artifacts |
| `requirements.txt` | Python dependencies for the dashboard |
| `dashboard_data.csv` | All 21 evaluated survey points with predictions and scores |
| `predictions_with_priority.csv` | Full inference output with PriorityScore and NeedsReview flag |
| `demo_top10_priority.csv` | Top 10 highest-priority records for quick review |
| `demo_needs_review.csv` | Records flagged for human review (low confidence) |
| `holdout_predictions.csv` | Holdout set predictions |
| `holdout_needs_review_queue.csv` | Holdout NeedsReview queue |

> The training images and trained model weights are not included in this repo (field photo data from a private client engagement). The dashboard runs on the exported CSV outputs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live demo

[Streamlit app](https://lfwc-ml-image-reader-dashboard-yowhr3vdwzlcdrs5kqsnmk.streamlit.app/) — [StoryMap](https://storymaps.arcgis.com/stories/85bff3144ac048faae7432be7e494d5c)

## Status & future work

Current demo classifies 21 survey points. The full dataset is ~1,000 survey points and ~4,000 images. Future work will expand labels (burned, buried, partially obstructed) and scale inference accordingly.
