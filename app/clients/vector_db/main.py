from app.clients.vector_db.lancedb_client import LancedbClient
from csv import DictReader

if __name__ == "__main__":
    client = LancedbClient("./data.db")
    with open("./data.csv", "r") as file:
        data = DictReader(file)
        data = list(data)
    client.insert_documents(data)
    print(client.search("healthcare"))