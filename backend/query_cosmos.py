import os
from azure.cosmos import CosmosClient
import json

ENDPOINT = os.getenv("COSMOS_ENDPOINT")
KEY = os.getenv("COSMOS_KEY")

client = CosmosClient(ENDPOINT, KEY)
database = client.get_database_client("pulmolens-db")
container = database.get_container_client("feedback")

query = "SELECT * FROM c ORDER BY c._ts DESC OFFSET 0 LIMIT 1"
items = list(container.query_items(query=query, enable_cross_partition_query=True))

print(json.dumps(items, indent=2))
