"""Small HTTP boundary between Streamlit and the local FastAPI application."""

import os

import httpx

API_URL = os.environ.get('RAG_LAB_API_URL', 'http://127.0.0.1:8000').rstrip('/')


def request(method: str, path: str, payload=None, params=None):
    response = httpx.request(method, API_URL + path, json=payload, params=params, timeout=180, trust_env=False)
    if response.is_error:
        try:
            detail = response.json().get('detail', response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(str(detail))
    return response.json()
