from dataclasses import dataclass
from datetime import datetime

@dataclass
class Prediction:
    timestamp: datetime
    prediction: bool
    prediction_rate: float
    page_values: int
    exit_rates: float
    bounce_rates: float
    weekend: bool
    administrative: int
    informational: int
    product_related: int
    administrative_duration: int
    informational_duration: int
    product_related_duration: int
    month: int
    new_visitor: bool
