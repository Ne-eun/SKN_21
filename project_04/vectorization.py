from langchain_core.documents import Document
import glob
import json
from pydantic import BaseModel
from tqdm import tqdm
from stores import get_client, COLLECTION_NAME
from qdrant_client.models import PayloadSchemaType
from stores import init_vector_store


class LoadedData(BaseModel):
    title: str
    contents: list[str]
    url: str


class MetaData(BaseModel):
    file_name: str
    url: str
    title: str
    seq: int


class Content(BaseModel):
    content: str
    metadata: MetaData


def make_document_from_data():
    docs = []

    for file_path in tqdm(glob.glob("./datas/*.json"), desc="데이터 로드"):
        file_name = file_path.split("/")[-1].replace(".json", "")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data = LoadedData.model_validate(data)

        for i, content in tqdm(enumerate(data.contents), desc="데이터 변환"):
            docs.append(
                Document(
                    page_content=f"{data.title}\n{content}",
                    metadata=MetaData(
                        file_name=file_name, url=data.url, title=data.title, seq=i
                    ).model_dump(),
                )
            )

    return docs


def init_payload():

    client = get_client()

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="title",
        field_schema=PayloadSchemaType.TEXT,
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="seq",
        field_schema=PayloadSchemaType.INTEGER,
    )


def save_document(docs):

    init_vector_store(docs)
    init_payload()


if __name__ == "__main__":
    docs = make_document_from_data()
    save_document(docs)
