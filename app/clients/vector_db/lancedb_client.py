from typing import Optional, Dict, Any, List
import lancedb
import logging

from app.clients.vector_db.schema import Words, model


class LancedbClient:
    def __init__(self, url: str, table: Optional[str] = None) -> None:
        self._db = lancedb.connect(url)

        self._table_name = table or "funding_events"
        self._table = self._get_or_create_table(self._table_name)

    def insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        self._table.add(documents)
        logging.info(f"Inserted {len(documents)} documents into table {self._table_name}")
        logging.info(f"First document: {self._table.to_pandas().head(2)}")
        return

    def search(self, query: str) -> Words:
        results = self._table.search(query).limit(1).to_pydantic(Words)[0]
        return results.event_text

    def _get_or_create_table(self, table_name: str) -> lancedb.table.Table:
        if table_name in self._db.table_names():
            return self._db.open_table(table_name)
        return self._db.create_table(table_name, schema=Words)
