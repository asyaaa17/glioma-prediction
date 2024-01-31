from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Integer, String, Float
from database import BasePredictions as Base


class PredictionRequestDB(Base):
    __tablename__ = 'prediction_requests'

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[str] = mapped_column()
    gender: Mapped[str] = mapped_column()
    age_at_diagnosis: Mapped[float] = mapped_column()
    primary_diagnosis: Mapped[str] = mapped_column()
    race: Mapped[str] = mapped_column()
    idh1: Mapped[int] = mapped_column()
    tp53: Mapped[int] = mapped_column()
    atrx: Mapped[int] = mapped_column()
    pten: Mapped[int] = mapped_column()
    egfr: Mapped[int] = mapped_column()
    cic: Mapped[int] = mapped_column()
    muc16: Mapped[int] = mapped_column()
    pik3ca: Mapped[int] = mapped_column()
    nf1: Mapped[int] = mapped_column()
    pik3r1: Mapped[int] = mapped_column()
    fubp1: Mapped[int] = mapped_column()
    rb1: Mapped[int] = mapped_column()
    notch1: Mapped[int] = mapped_column()
    bcor: Mapped[int] = mapped_column()
    csmd3: Mapped[int] = mapped_column()
    smarca4: Mapped[int] = mapped_column()
    grin2a: Mapped[int] = mapped_column()
    idh2: Mapped[int] = mapped_column()
    fat4: Mapped[int] = mapped_column()
    pdgfra: Mapped[int] = mapped_column()