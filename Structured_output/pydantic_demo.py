from pydantic import BaseModel


# schema
class User(BaseModel):
    name: str
    age: int


user_data = {
    "name": "John",
    "age": 25
}

user = User(**user_data)

print(user)
