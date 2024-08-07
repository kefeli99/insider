import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

current_dir = Path(__file__).resolve().parent.parent
MODEL_DIR = current_dir / "model_artifacts"

app = FastAPI(
    title="LightFM Recommendation API",
    description="Inference API for LightFM recommendation model to provide personalized recommendations based on user events",
    version="0.0.1",
    contact={
        "name": "Osman Dogukan Kefeli",
        "email": "dogukankefeli@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Load the trained model and necessary data
model = joblib.load(MODEL_DIR / "lightfm_model.joblib")
dataset = joblib.load(MODEL_DIR / "lightfm_dataset.joblib")
item_features_matrix = joblib.load(MODEL_DIR / "item_features_matrix.joblib")
item_features = joblib.load(MODEL_DIR / "item_features.joblib")
user_features = joblib.load(MODEL_DIR / "user_features.joblib")
user_features_matrix = joblib.load(MODEL_DIR / "user_features_matrix.joblib")

# Create reverse mappings
user_id_map, _, item_id_map, _ = dataset.mapping()
reverse_item_map = {v: k for k, v in item_id_map.items()}


class UserEvent(BaseModel):
    date: str
    userId: str
    sessionId: str
    pageType: str
    itemId: str
    category: str
    productPrice: float | None = Field(None, alias="productPrice")
    oldProductPrice: float | None = Field(None, alias="oldProductPrice")

    model_config = ConfigDict(populate_by_name=True)


class RecommendationRequest(BaseModel):
    events: list[UserEvent]
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "events": [
                    {
                        "date": "2019-08-05 19:30:37",
                        "userId": "00172f1d9a71e9a8de0aa34288a6b19b",
                        "sessionId": "e8167c23f8ac2f9be979c32380e0fc2b7e94941e917d30d3376cc2005b163998",
                        "pageType": "productDetail",
                        "itemId": "83472aea4051c00d031b01ff42ef73fc",
                        "category": '["kadın çanta","omuz askılı çanta"]',
                        "productPrice": 622.0,
                        "oldProductPrice": 1220.0,
                    }
                ]
            }
        },
    )


def get_base_recommendations(user_id: str, n: int = 10) -> list[Any]:
    if user_id not in user_id_map:
        return []

    user_idx = user_id_map[user_id]
    n_items = item_features_matrix.shape[0]

    scores = model.predict(
        user_idx, np.arange(n_items), item_features=item_features_matrix
    )
    top_items = np.argsort(-scores)

    return [
        reverse_item_map[item] for item in top_items[:n] if item in reverse_item_map
    ]


def parse_category(category_str: str) -> list[Any]:
    try:
        categories = json.loads(category_str)
        if isinstance(categories, list):
            return categories
        else:
            return [categories]
    except json.JSONDecodeError:
        return [category_str] if category_str != "[]" else []


def rerank_recommendations(base_recs, recent_events, n=5):
    event_categories = []
    for event in recent_events:
        event_categories.extend(parse_category(event.category))

    # Create a simple category-based score for each recommendation
    reranked_recs = []
    for item_id in base_recs:
        # Select the row corresponding to the current item_id
        item_row = item_features[item_features["item_id"] == item_id]

        # Check which categories the item belongs to
        item_categories = [
            col
            for col in item_row.columns
            if col.startswith("cat_") and item_row[col].iloc[0] == 1
        ]

        # Calculate the category score based on recent events
        category_score = sum(
            any(cat in event_cat for cat in item_categories)
            for event_cat in event_categories
        )

        reranked_recs.append((item_id, category_score))

    # Sort by category score, then by original order
    reranked_recs.sort(key=lambda x: (-x[1], base_recs.index(x[0])))

    return [item_id for item_id, _ in reranked_recs[:n]]


@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    if not request.events:
        raise HTTPException(status_code=400, detail="No events provided")

    user_id = request.events[0].userId  # Assume all events are for the same user
    base_recs = get_base_recommendations(user_id)
    if not base_recs:
        raise HTTPException(status_code=404, detail="User not found")

    final_recs = rerank_recommendations(base_recs, request.events)
    return {"recommendations": final_recs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
