import config
import weaviate
from weaviate.classes.config import Property,DataType,Tokenization
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter 



def connect():
    if config.WEAVIATE_URL:
        from weaviate.classes.init import Auth

        if not config.WEAVIATE_API_KEY:
            raise RuntimeError("URL is set but API Key is missing. Please check your .env file")

        return weaviate.connect_to_weaviate_cloud(
            cluster_url = config.WEAVIATE_URL,
            auth_credentials = config.WEAVIATE_API_KEY
        ) 

    return weaviate.connect_to_local()


def get_collections(client):

       if not client.collections.exists(config.COLLECTION):

            client.collections.create(
                 config.COLLECTION,
                    properties = [
                         Property(name = "text", data_type = DataType.TEXT),
                         Property(name = "source", data_type = DataType.TEXT, tokenization = Tokenization.FIELD),
                         Property(name = "page", data_type = DataType.INT),
                         Property(name = "chunk", data_type = DataType.INT)


                    ]
            )
       return client.collections.get(config.COLLECTION)


def count_all(col) -> int:
     return col.aggregate.over_all(total_count = True).total_count or 0


def count_for(col, source: str) -> int:
     res = col.aggregate.over_all(
          filters = Filter.by_property('source').equal(source), total_count = True
     )
     return res.total_count or 0


def remove_file(col, source:str) -> None:

     col.data.delete_many(where = Filter.by_property('source').equal(source))


def insert_chunks(col, rows: list[dict]):

     objects = [
          DataObject(
               properties = {k: r[k] for k in ("text","source","page","chunk")},
               vector = r['vector'],
          ) for r in rows
     ]
     
     res = col.data.insert_many(objects)

     if res.has_errors:
          raise RuntimeError(f"Weaviate insert failed: {res.errors} " )

def hybrid_search(col, query : str, vector: list[float], limit: int, alpha: float, source_like: str | None = None ) -> list[dict]:

     filters = Filter.by_property("source").like(source_like) if source_like else None

     res = col.query.hybrid(
          query = query, vector = vector, alpha = alpha, limit = limit, filters = filters
     )

     return [o.properties for o in res.objects]     




