# -*- coding: utf-8 -*-
import re
import json
import uuid
import os
import numpy as np

import torch

import umap.umap_ as umap
from sklearn.mixture import GaussianMixture
from typing import Optional
from collections import defaultdict
import pynndescent.distances
import numba
import tempfile
import base64
import pymupdf4llm
import pymupdf
import glob

from fastapi import FastAPI, Path, Body, BackgroundTasks, HTTPException, UploadFile
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from singleton import SingletonABCMeta
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
from FlagEmbedding import FlagReranker
from llama_index.readers.file import PyMuPDFReader


@numba.njit(fastmath=True)
def correct_alternative_cosine(ds):
    result = np.empty_like(ds)
    for i in range(ds.shape[0]):
        result[i] = 1.0 - np.power(2.0, ds[i])
    return result


pynndescent.distances.correct_alternative_cosine = correct_alternative_cosine

app = FastAPI()


class Content(BaseModel):
    content: str


class Contents(BaseModel):
    contents: List[str]


class Rerank(BaseModel):
    contexts: List[str]
    question: str


class _SegmentationModel(metaclass=SingletonABCMeta):
    __pipe = pipeline(task='token-classification',
                      model="igorsterner/xlmr-multilingual-sentence-segmentation",
                      stride=5,
                      framework='pt')

    def split_sentences(self, document: str, min_chunk_size: int = 6):
        """
        문서를 문장 단위로 분할합니다.

        :param document: 문장 분할 대상 문서
        :param min_chunk_size: 청크 최소 사이즈
        :return: 청크 목록
        """
        segments = self.__pipe(document)
        if len(segments) == 0:
            return [document]

        chunks = []
        stripped_space_chunks = set()

        def process_chunk(chunk: str):
            stripped_space_chunk = chunk.replace(' ', '')
            if stripped_space_chunk not in stripped_space_chunks and len(stripped_space_chunk) >= min_chunk_size:
                chunks.append(chunk)
                stripped_space_chunks.add(stripped_space_chunk)

        previous_segment_end_pos = 0
        for segment in segments:
            current_segment_end_pos = segment['end']
            chunk = re.sub(r'[\r\n\s]+', ' ', document[previous_segment_end_pos: current_segment_end_pos]).strip()
            previous_segment_end_pos = current_segment_end_pos
            process_chunk(chunk)

        last_segment_end_pos = segments[-1]['end']
        if last_segment_end_pos < len(document):
            partial_document = document[last_segment_end_pos:].strip()
            for chunk in self.split_sentences(partial_document, min_chunk_size):
                process_chunk(chunk)

        return chunks


# embedding model
model_ko = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
model_en = SentenceTransformer('all-MiniLM-L6-v2')
model_ko_sroberta = SentenceTransformer('jhgan/ko-sroberta-multitask')
model_ko_roberta = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')

# re-ranker model
tokenizer_ko_reranker = AutoTokenizer.from_pretrained("Dongjin-kr/ko-reranker")
model_ko_reranker = AutoModelForSequenceClassification.from_pretrained("Dongjin-kr/ko-reranker")
bge_m3_reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
bge_gemma_reranker = FlagReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True)

# sentence segmentation model
_segmentation_model = _SegmentationModel()

#summarizer
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

RANDOM_SEED = 42


@app.post("/embedding/{model_name}")
async def get_embedding(model_name: str = Path(...), content: Content = Body(...)):
    sentences = content.content

    if not sentences or sentences.lower() == "null":
        raise HTTPException(status_code=400, detail="Invalid content: sentences cannot be null or 'null'.")

    embeddings = await get_embeddings(model_name, sentences)

    return {"embedding": embeddings.tolist()}


async def get_embeddings(model_name, sentences):
    if model_name == "ko":
        model = model_ko
    elif model_name == "ko_roberta":
        model = model_ko_roberta
    elif model_name == "ko_sroberta":
        model = model_ko_sroberta
    else:
        model = model_en
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=64)
    return embeddings


@app.post("/re-rank/{model_name}")
async def re_rank(model_name: str = Path(...), rerank: Rerank = Body(...)):

    contexts = rerank.contexts
    question = rerank.question

    return {"result": re_ranking(contexts, question, model_name)}


@app.post("/chunk")
async def get_chunk(background_tasks: BackgroundTasks, content: Content = Body(...)):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(process_file, task_id, content.content)

    return {"task_id": task_id}


@app.get("/task/result/{task_id}")
async def get_chunk(task_id: str):
    with open(task_id+".json", 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


@app.post("/cluster-documents/{model_name}")
async def get_raptor(background_tasks: BackgroundTasks, contents: Contents = Body(...), model_name: str = Path(...)):
    task_id = str(uuid.uuid4())
    background_tasks.add_task(cluster_document, task_id, model_name, contents.contents)

    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    status_file = f"{task_id}.status"
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = f.read()
        return {"task_id": task_id, "status": status}
    else:
        return {"task_id": task_id, "status": "Not found"}


@app.delete("/status/{task_id}")
async def delete_status(task_id: str):
    status_file = f"{task_id}.status"
    result_file = f"{task_id}.json"
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = f.read()
        if status == "Completed":
            os.remove(status_file)
            os.remove(result_file)
            return {"task_id": task_id, "status": "Deleted"}
        else:
            return {"task_id": task_id, "status": status + "can't delete"}
    else:
        return {"task_id": task_id, "status": "Not found"}


@app.post("/pdf/extract")
async def extract_pdf_data(file: UploadFile):
    content_type = file.headers['content-type'] if 'content-type' in file.headers else None
    if content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail='Invalid content type: only pdf files are allowed.')

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(file.file.read())
        file_path = temp_file.name

    try:
        result = []
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                pymupdf4llm.to_markdown(
                    file_path,
                    write_images=True,
                    image_path=temp_dir,
                    image_size_limit=0.2
                )

                page_size_by_page = {}
                with pymupdf.open(file_path, filetype='pdf') as pdf_image_reader:
                    for i, page in enumerate(pdf_image_reader):
                        page_size_by_page[i] = page.mediabox.width * page.mediabox.height

                pdf_text_reader = PyMuPDFReader().load_data(file_path)
                for i, page in enumerate(pdf_text_reader):
                    images = []
                    for img_file in glob.glob(f'''{temp_dir}/{file_path.split(os.path.sep)[-1]}-{i}-*'''):
                        with pymupdf.open(img_file) as img:
                            rect = img[0].rect
                            img_area = rect.width * rect.height
                        if img_area < page_size_by_page[i] * 0.8:
                            with open(img_file, 'rb') as f:
                                images.append(base64.b64encode(f.read()).decode('utf-8'))

                    result.append({'page': i + 1, 'text': page.text, 'images': images})
        finally:
            if os.path.exists(temp_dir):
                os.unlink(temp_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error occurred while extracting pdf data: {str(e)}')
    finally:
        os.unlink(file_path)

    return {"result": result}


@app.post("/summarize")
async def get_embedding(content: Content = Body(...)):

    sentence = content.content

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(sentence)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=3000
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=400,
        min_length=100,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return {"result": summary}


def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED) -> int:
    """
    가우시안 혼합 모델(Gaussian Mixture Model)을 사용하여 베이지안 정보 기준(BIC)을 통해 최적의 클러스터 수를 결정합니다.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED):
    """
    확률 임계값을 기반으로 가우시안 혼합 모델(GMM)을 사용하여 임베딩을 클러스터링합니다.
    """
    n_clusters = get_optimal_clusters(embeddings, random_state=random_state)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0].tolist() for prob in probs]
    return labels, n_clusters


def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    """
    UMAP을 사용하여 임베딩의 전역 차원 축소를 수행합니다.
    """
    dataset_size = len(embeddings)

    if dataset_size <= 4:
        print(f"Dataset size is too small ({dataset_size}) larger than 4. Skipping UMAP.")
        return embeddings

    if n_neighbors is None:
        n_neighbors = min(int((len(embeddings) - 1) ** 0.5), dataset_size - 1)

    try:
        return umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=dim,
            metric=metric,
            min_dist=0.1,
            spread=1.0
        ).fit_transform(embeddings)
    except Exception as e:
        print(f"UMAP failed: {str(e)}. Returning original embeddings.")
        return embeddings


async def cluster_document(task_id, model_name, contents):
    status_file = f"{task_id}.status"
    current_step = "Starting"
    try:
        with open(status_file, 'w') as f:
            f.write(current_step)
        print(task_id + "embedding_start")
        if not contents:
            raise ValueError("Contents list is empty")

        current_step = "Embedding Start"
        with open(status_file, 'w') as f:
            f.write(current_step)
        embeddings = await get_embeddings(model_name, contents)
        if embeddings.size == 0:
            raise ValueError("Generated embeddings are empty")

        print(task_id + "embedding_done")
        current_step = "Embedding Done"
        with open(status_file, 'w') as f:
            f.write(current_step)

        print(task_id + "clustering_start")

        current_step = "UMAP Dimensionality Reduction Start"
        with open(status_file, 'w') as f:
            f.write(current_step)
        embeddings = embeddings.astype(np.float64)
        print(f"Embeddings dtype before UMAP: {embeddings.dtype}, shape: {embeddings.shape}")
        reduced_embeddings = global_cluster_embeddings(embeddings, dim=10)

        if reduced_embeddings.size == 0:
            raise ValueError("Reduced embeddings are empty after UMAP")

        current_step = "UMAP Dimensionality Reduction Done"
        with open(status_file, 'w') as f:
            f.write(current_step)

        current_step = "GMM Clustering Start"
        with open(status_file, 'w') as f:
            f.write(current_step)
        reduced_embeddings = reduced_embeddings.astype(np.float64)
        labels, n_clusters = GMM_cluster(reduced_embeddings, threshold)

        print(task_id + "clustering_done")
        current_step = "GMM Clustering Done"
        with open(status_file, 'w') as f:
            f.write(current_step)

        current_step = "Grouping Clusters Start"
        with open(status_file, 'w') as f:
            f.write(current_step)
        cluster_dict = defaultdict(list)
        for content, label_list in zip(contents, labels):
            for label in label_list:
                cluster_dict[int(label)].append(content)

        clusters = list(cluster_dict.values())
        current_step = "Grouping Clusters Done"
        with open(status_file, 'w') as f:
            f.write(current_step)

        current_step = "Saving Results Start"
        with open(status_file, 'w') as f:
            f.write(current_step)
        with open(task_id + ".json", 'w', encoding='utf-8') as json_file:
            json.dump({"clusters": clusters, "clusters_size": int(n_clusters)}, json_file, ensure_ascii=False, indent=4,
                      default=convert_to_serializable)

        current_step = "Completed"
        with open(status_file, 'w') as f:
            f.write(current_step)
    except Exception as e:
        with open(status_file, 'w') as f:
            f.write(f"Failed during {current_step}: {str(e)}")


def convert_to_serializable(obj):
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError("Object of type %s is not JSON serializable" % type(obj).__name__)


def process_file(task_id, content):
    status_file = f"{task_id}.status"
    try:
        with open(status_file, 'w') as f:
            f.write("Processing")

        print(task_id + "chunking_start")
        small_chunks = _get_small_chunks(content)
        print(task_id + "chunking_done")

        with open(task_id+".json", 'w', encoding='utf-8') as json_file:
            json.dump({"chunks": small_chunks}, json_file, ensure_ascii=False, indent=4)

        with open(status_file, 'w') as f:
            f.write("Completed")

    except Exception as e:
        with open(status_file, 'w') as f:
            f.write(f"Failed: {str(e)}")


def _get_small_chunks(doc: str) -> List[str]:
    """
    문서 내 문장을 청크로 분할합니다.
    :param doc: 청킹 대상 문서
    :return: 청크 목록
    """
    chunks = _segmentation_model.split_sentences(doc)

    return chunks


def re_ranking(contexts, question, model="ko_reranker"):
    pairs = []
    for context in contexts:
        pair = [question, context]
        pairs.append(pair)
    result = []
    if model == "ko_reranker":
        model_ko_reranker.eval()
        with torch.no_grad():
            inputs = tokenizer_ko_reranker(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = model_ko_reranker(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(scores.numpy())

            scores = [float(score) for score in scores]
    elif model == "bge_m3_reranker":
        scores = bge_m3_reranker.compute_score(pairs, normalize=True)
    elif model == "bge_gemma_reranker":
        scores = bge_gemma_reranker.compute_score(pairs, normalize=True)
    else:
        return "Not supported model"
    for i, context in enumerate(contexts):
        result.append({"context": context, "score": scores[i]})
    result = sorted(result, key=lambda x: x["score"], reverse=True)

    return result


def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()
