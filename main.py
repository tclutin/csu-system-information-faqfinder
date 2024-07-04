from contextlib import asynccontextmanager
from http import HTTPStatus

import httpx
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from config.config import settings
from model.query import QueryModel

model_name = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

auth_token = None
faqs = []
question_vectors = None
question_cache = {}

'''
embeddings_cache = {}
def embed_text(texts):
    embeddings = []
    for text in texts:
        if text not in embeddings_cache:
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings_cache[text] = outputs.pooler_output.numpy()

        embeddings.append(embeddings_cache[text])

    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    return embeddings

def update_question_vectors(faqs):
    questions = [item['question'] for item in faqs]
    return embed_text(questions)

async def initialize_service():
    global faqs, question_vectors
    await login_and_fetch_token()
    faqs = await fetch_and_transform_faqs(settings.api_url)
    question_vectors = update_question_vectors(faqs)


def perform_faq_search(query, threshold=0.848):
    query_vector = embed_text([query])[0].reshape(1, -1)
    similarities = cosine_similarity(question_vectors, query_vector).flatten()
    results = [
        {"question": faqs[idx]['question'], "answer": faqs[idx]['answer'], "similarity": float(similarities[idx])}
        for idx in range(len(similarities)) if similarities[idx] > threshold
    ]
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    if results:
        return results[0]
    else:
        return {"question": "Не найдено", "answer": "К сожалению, ответ на ваш запрос не найден.", "similarity": 0.0}

anoter
def vectorize_questions_bad_woman_moment(faqs, model, tokenizer):
    vectors = []
    for item in faqs:
        combined_text = f"{item['question']} {item['answer']}"
        if combined_text in question_cache:
            vectors.append(question_cache[combined_text])
        else:
            embeddings = vectorize_question(combined_text, model, tokenizer)
            vectors.append(embeddings)
            question_cache[combined_text] = embeddings
    return vectors

'''


async def login_and_fetch_token():
    global auth_token
    login_data = {
        "username": settings.api_login,
        "password": settings.api_password,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(settings.api_auth_url, json=login_data)
        response.raise_for_status()
        auth_token = response.json().get("access_token")


async def fetch_and_transform_faqs(api_url):
    global auth_token
    headers = {"Authorization": f"Bearer {auth_token}"}
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, headers=headers)

        if response.status_code in [HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN]:
            await login_and_fetch_token()
            headers = {"Authorization": f"Bearer {auth_token}"}
            response = await client.get(api_url, headers=headers)

        response.raise_for_status()
        faqs_data = response.json()
    transformed_faqs = [{"question": faq["question"], "answer": faq["answer"]} for faq in faqs_data]
    return transformed_faqs


def vectorize_question(question, model, tokenizer):
    tokens = tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings


def vectorize_questions(pack, model, tokenizer):
    question_vectors = []
    for item in pack:
        if item['question'] in question_cache:
            question_vectors.append(question_cache[item['question']])
        else:
            embeddings = vectorize_question(item['question'], model, tokenizer)
            question_vectors.append(embeddings)
            question_cache[item['question']] = embeddings
    return question_vectors


def update_question_vectors(faqs):
    return vectorize_questions(faqs, model, tokenizer)


async def initialize_service():
    global faqs, question_vectors
    await login_and_fetch_token()
    faqs = await fetch_and_transform_faqs(settings.api_url)
    question_vectors = update_question_vectors(faqs)


async def update_faqs_and_vectors():
    global faqs, question_vectors
    try:
        new_faqs = await fetch_and_transform_faqs(settings.api_url)
    except Exception as e:
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))

    if new_faqs != faqs:
        faqs = new_faqs
        question_vectors = update_question_vectors(faqs)
        print("FAQs and question vectors updated successfully.")


def find_question_answers_oldversion(input_question, question_vectors, pack, top_k, min_similarity):
    if input_question in question_cache:
        input_output = question_cache[input_question]
    else:
        input_output = vectorize_question(input_question, model, tokenizer)
        question_cache[input_question] = input_output

    similarities = cosine_similarity([input_output], question_vectors)[0]

    filtered_indices = np.where(similarities > min_similarity)[0]
    top_indices = filtered_indices[np.argsort(similarities[filtered_indices])[::-1][:top_k]]

    results = []
    for idx in top_indices:
        results.append({
            "question": pack[idx]['question'],
            "answer": pack[idx]['answer'],
            "similarity": float(similarities[idx])
        })

    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_service()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/query", response_model=None, status_code=HTTPStatus.OK)
def search_faq(query: QueryModel):
    try:
        results = find_question_answers_oldversion(query.message, question_vectors, faqs, top_k=query.top_k,
                                                   min_similarity=query.min_similarity)
        return results
    except HTTPException as e:
        return {"error": str(e)}


@app.post("/update", response_model=None, status_code=HTTPStatus.OK)
async def update_faq():
    try:
        await update_faqs_and_vectors()
        return {"message": "updated"}
    except HTTPException as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.http_host,
        port=settings.http_port,
        reload=True
    )
