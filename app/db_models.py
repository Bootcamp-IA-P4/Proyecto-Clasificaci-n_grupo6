from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import Column, Integer, Float, Boolean, String, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now)
    prediction = Column(Boolean, nullable=False)
    prediction_rate = Column(Float, nullable=False)
    page_values = Column(Float)
    exit_rates = Column(Float)
    bounce_rates = Column(Float)
    weekend = Column(Boolean)
    administrative = Column(Integer)
    informational = Column(Integer)
    product_related = Column(Integer)
    administrative_duration = Column(Float)
    informational_duration = Column(Float)
    product_related_duration = Column(Float)
    month = Column(String)
    visitor_type = Column(String)
    
    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "prediction": "Compra" if self.prediction else "No Compra",
            "probability": f"{self.prediction_rate:.2%}",
            "page_values": self.page_values,
            "exit_rates": self.exit_rates,
            "bounce_rates": self.bounce_rates,
            "weekend": "Sí" if self.weekend else "No",
            "administrative": self.administrative,
            "informational": self.informational,
            "product_related": self.product_related,
            "administrative_duration": self.administrative_duration,
            "informational_duration": self.informational_duration,
            "product_related_duration": self.product_related_duration,
            "month": self.month,
            "visitor_type": self.visitor_type
        }