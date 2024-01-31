from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, Float, ForeignKey
from database import BasePredictions as Base

class PredictionResultDB(Base):
    __tablename__ = 'prediction_results'

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey('prediction_requests.id'))
    prediction: Mapped[int] = mapped_column()
    probability: Mapped[float] = mapped_column()
