from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


engine_predictions = create_engine('sqlite:///predictions.db')
SessionLocalPredictions = sessionmaker(autocommit=False, autoflush=False, bind=engine_predictions)

BasePredictions = declarative_base()


engine_users = create_engine('sqlite:///users.db')
SessionLocalUsers = sessionmaker(autocommit=False, autoflush=False, bind=engine_users)

BaseUsers = declarative_base()


BaseUsers.metadata.create_all(engine_users)
print("Таблицы созданы.")    

BasePredictions.metadata.create_all(engine_predictions)
