from FlagEmbedding import FlagModel
import faiss
from fastapi import FastAPI, Depends, Body
from pydantic import BaseModel
import uvicorn
from py_eureka_client import eureka_client
from py_eureka_client import netint_utils
import os
from typing import List
import numpy as np

FAISS_DATA_PATH = os.getenv("FAISS_DATA_PATH", "/app/data/faiss_data.index")

def get_data_faiss():
    faiss_location_file = FAISS_DATA_PATH
    if os.path.exists(faiss_location_file):
        data_index = faiss.read_index(faiss_location_file)
    else:
        data_index = faiss.index_factory(
            768, "IDMap,Flat", faiss.METRIC_INNER_PRODUCT)
    try:
        yield data_index
    finally:
        faiss.write_index(data_index, faiss_location_file)


def get_flag_model():
    flag_model = FlagModel(
        'BAAI//bge-small-zh-v1.5', query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", use_fp16=False)
    return flag_model


app = FastAPI()
server_port = 8000


class InsertDataModel(BaseModel):
    data: str
    id: int
    # 生成example

    class Config:
        json_schema_extra = {
            "example":
                {"data": "你好呀，今天是周几？", "id": 1022}
        }


@app.on_event("startup")
async def startup_event():
    # 注册到registry
    ip = netint_utils.get_first_non_loopback_ip("10.96.0.0/12")
    if str(ip).strip() == '':
        print('未获取到IP，不进行注册')
    else:
        print("get ip is " + ip)
        cluster_name = os.getenv("SPRING_PROFILES_ACTIVE", "test")
        print("cluster name is " + cluster_name)
        await eureka_client.init_async(
            eureka_server=f"http://registry.support-{cluster_name}.svc.cluster.local:8761",
            app_name="faiss-server",
            instance_host=ip,
            instance_ip=ip,
            instance_port=server_port,
        )


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/insert")
def insert_data(insertData: List[InsertDataModel] = Body(default=[],
                                                         example=[{"data": "你好呀，今天是周几？", "id": 1022}, {"data": "你好", "id": 1023}, {"data": "明天是17号", "id": 1024}, {"data": "后天是19号", "id": 1025}, {"data": "后天是19号", "id": 1026}]),
                data_index=Depends(get_data_faiss), flag_model=Depends(get_flag_model)):
    if insertData == []:
        return {"message": "No data to insert", "status": 400, "data": data_index.ntotal}
    data = [item.data for item in insertData]
    ids = np.array([item.id for item in insertData]).astype(np.int32)
    data_index.remove_ids(ids)
    encodedData = flag_model.encode(data)
    data_index.add_with_ids(encodedData, ids)
    return {"message": "Insert data success", "status": 200, "data": data_index.ntotal}


@app.post("/delete")
def delete_data(deleteData: List[int] = Body(default=[],
                                             example=[1022, 1023, 1024, 1025, 1026]),
                data_index=Depends(get_data_faiss)):
    if deleteData == []:
        return {"message": "No data to delete", "status": 400, "data": data_index.ntotal}
    ids = np.array(deleteData).astype(np.int32)
    data_index.remove_ids(ids)
    return {"message": "Delete data success", "status": 200, "data": data_index.ntotal}


@app.get("/data_amount")
def data_amount(data_index=Depends(get_data_faiss)):
    return {"message": "Get data amount success", "status": 200, "data": data_index.ntotal}


@app.get("/search")
def search_data(query: str, top_k: int = 5, use_query : bool = True, data_index=Depends(get_data_faiss), flag_model=Depends(get_flag_model)):
    if use_query:
        encodedQuery = flag_model.encode_queries([query])
    else:
        encodedQuery = flag_model.encode([query])
    distances, ids = data_index.search(encodedQuery, top_k)
    # 转换distances和ids两个数组为对象数组
    distances = distances.tolist()[0]
    ids = ids.tolist()[0]
    ids = [item for item in ids if item != -1]
    result = []
    for i in range(len(ids)):
        result.append({"score": distances[i]*100, "id": ids[i]})
    return {"message": "Search data success", "status": 200, "data": result}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=server_port)
