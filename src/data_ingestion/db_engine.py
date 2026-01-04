from sqlalchemy import create_engine

def get_engine():
    """
    Local PostgreSQL connection (Unix socket).
    Database: walmart_drift_db
    """
    return create_engine("postgresql+psycopg2:///walmart_drift_db", echo=False)

