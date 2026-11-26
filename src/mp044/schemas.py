"""Stage 3: input validation for the prediction API."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SENTINEL = -999  # Stage 1 finding: typed in instead of a blank in fodder_price_rs


class MilkRecord(BaseModel):
    """One village centre on one day. Unknown fields (e.g. the leaking column) are rejected."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "record_id": "R50001",
                "cows": 55,
                "buffaloes": 146,
                "fodder_price_rs": 18.0,
                "temp_c": 28.9,
                "vet_visits": 2,
                "feed_kg": 2076,
                "breed_mix": "mixed",
                "village_cluster": "Cluster A",
            }
        },
    )

    record_id: Optional[str] = None
    cows: int = Field(ge=0)
    buffaloes: int = Field(ge=0)
    # null (or -999) means "not known"; the model pipeline fills it with the median
    fodder_price_rs: Optional[float] = Field(ge=0)
    temp_c: float = Field(ge=-10, le=60)
    vet_visits: int = Field(ge=0)
    feed_kg: float = Field(ge=0)
    breed_mix: Literal["desi", "crossbred", "mixed"]
    village_cluster: Literal["Cluster A", "Cluster B", "Cluster C"]

    @field_validator("fodder_price_rs", mode="before")
    @classmethod
    def sentinel_is_missing(cls, value):
        return None if value == SENTINEL else value

    @field_validator("breed_mix", mode="before")
    @classmethod
    def tidy_breed(cls, value):
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("village_cluster", mode="before")
    @classmethod
    def tidy_cluster(cls, value):
        return value.strip() if isinstance(value, str) else value


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[MilkRecord] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    record_id: Optional[str] = None
    prediction: float


class BatchResponse(BaseModel):
    predictions: list[Prediction]
