from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class TransactionInput(BaseModel):
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
