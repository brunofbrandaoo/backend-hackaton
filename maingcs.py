# main.py
import os
import io
import json
import base64
import logging
from typing import List, Optional, Literal

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from supabase import create_client, Client
from google import genai

# ==== LOG ====
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("educa-ai")

# ==== ENV (Supabase) ====
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "materiais")

# ==== ENV (Gemini) ====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ==== ENV (GCS) ====
GCS_DEFAULT_BUCKET = os.getenv("GCS_DEFAULT_BUCKET", "materiais-hackaton")
GCS_SA_JSON = os.getenv("GCS_SERVICE_ACCOUNT_JSON")  # conteúdo JSON da SA (opcional)
# alternativo: usar GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

# ==== VERIFICAÇÕES BÁSICAS ====
if not GEMINI_API_KEY:
    log.warning("⚠️ GEMINI_API_KEY não configurada — chamadas ao Gemini irão falhar.")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    log.info("ℹ️ SUPABASE_URL/SUPABASE_ANON_KEY não configurados (rotas Supabase podem falhar se usadas).")
if not (GCS_SA_JSON or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")):
    log.info("ℹ️ Credenciais do GCS não configuradas via env (GCS_SERVICE_ACCOUNT_JSON ou GOOGLE_APPLICATION_CREDENTIALS).")

# ==== CONEXÕES ====
def supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Variáveis do Supabase ausentes.")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

def gemini_client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY ausente.")
    return genai.Client(api_key=GEMINI_API_KEY)

# ==== GCS Client / Download ====
from google.cloud import storage
from google.oauth2 import service_account

def gcs_client() -> storage.Client:
    """
    Cria cliente GCS.
    - Se GCS_SERVICE_ACCOUNT_JSON estiver setado (conteúdo JSON), usa-o.
    - Senão, tenta GOOGLE_APPLICATION_CREDENTIALS (path) ou ADC padrão.
    """
    if GCS_SA_JSON and os.path.isfile(GCS_SA_JSON):
        # É um caminho para arquivo
        creds = service_account.Credentials.from_service_account_file(GCS_SA_JSON)
        with open(GCS_SA_JSON, 'r') as f:
            project_id = json.load(f).get("project_id")
        return storage.Client(credentials=creds, project=project_id)
    elif GCS_SA_JSON and GCS_SA_JSON.startswith('{'):
        # É conteúdo JSON direto
        info = json.loads(GCS_SA_JSON)
        creds = service_account.Credentials.from_service_account_info(info)
        return storage.Client(credentials=creds, project=info.get("project_id"))
    
    # Fallback: usa GOOGLE_APPLICATION_CREDENTIALS ou ADC
    return storage.Client()

def gcs_baixar_pdf_bytes(bucket_name: str, blob_path: str) -> bytes:
    cli = gcs_client()
    bucket = cli.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        raise RuntimeError(f"Objeto não encontrado no GCS: gs://{bucket_name}/{blob_path}")
    return blob.download_as_bytes()

# ==== PDF -> IMAGENS ====
def pdf_para_imagens(
    pdf_bytes: bytes,
    dpi: int = 200,
    max_paginas: Optional[int] = None,
    como_png: bool = True,
    qualidade_jpeg: int = 85,
) -> List[bytes]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    imgs: List[bytes] = []
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            if max_paginas is not None and i >= max_paginas:
                break
            pix = page.get_pixmap(matrix=mat)
            imgs.append(pix.tobytes("png") if como_png else pix.tobytes("jpeg", quality=qualidade_jpeg))
    finally:
        doc.close()
    return imgs

def baixar_pdf_bucket(caminho: str, bucket: str) -> bytes:
    sb = supabase_client()
    try:
        return sb.storage.from_(bucket).download(caminho)
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar do Storage (Supabase): {e}")

def inline_img(b: bytes, mime: str) -> dict:
    return {"inline_data": {"mime_type": mime, "data": base64.b64encode(b).decode("utf-8")}}

# ==== PROMPTS ====
def prompt_ocr() -> str:
    return (
        "Você receberá páginas de um PDF como imagens. Faça OCR de alta fidelidade.\n"
        "- Extraia o texto exatamente como está, preservando títulos, listas e formatação essencial.\n"
        "- NÃO resuma, NÃO interprete, NÃO traduza.\n"
        "- Retorne apenas o texto puro, sem comentários."
    )

def prompt_gerar_questoes(serie: str, qtd: int, ineditas: bool) -> str:
    """
    - 1º, 2º ou 3º ano do ensino médio -> estilo ENEM
    - Demais séries -> padrão didático da série
    """
    serie_lower = serie.lower().strip()
    # aceita "medio" sem acento também
    eh_medio = ("médio" in serie_lower) or ("medio" in serie_lower)
    if any(x in serie_lower for x in ["1", "2", "3"]) and eh_medio:
        base = (
            f"Você é um professor de {serie}. "
            "Você receberá páginas de material didático como imagens. "
            f"Primeiro, faça OCR e compreenda o conteúdo. "
            f"Em seguida, elabore {qtd} questões de múltipla escolha no estilo do ENEM, "
            "com quatro alternativas (A–D) e contextualização interdisciplinar.\n\n"
            "Regras para estilo ENEM:\n"
            "- Contextualize com situações-problema do cotidiano e temas interdisciplinares.\n"
            "- O enunciado deve exigir interpretação e análise, não apenas memorização.\n"
            "- As alternativas devem ser plausíveis e testar raciocínio.\n"
            "- Linguagem formal, clara, em português padrão. Uma resposta correta por questão.\n\n"
            "Formato obrigatório de saída:\n"
            "1) Pergunta\nA) ...\nB) ...\nC) ...\nD) ...\nCorreta: <A|B|C|D>\n"
            "Sem explicações adicionais."
        )
        if ineditas:
            base += "\nAs questões devem ser INÉDITAS; não copie trechos literais do material."
        return base

    base = (
        f"Você é um professor de {serie}. "
        "Você receberá páginas de material didático como imagens. "
        "Primeiro, faça OCR e compreenda o conteúdo. "
        f"Depois, crie {qtd} questões de múltipla escolha (4 alternativas) adequadas à série, "
        "respeitando o nível cognitivo e linguístico do aluno.\n\n"
        "Formato obrigatório:\n"
        "1) Pergunta\nA) ...\nB) ...\nC) ...\nD) ...\nCorreta: <A|B|C|D>\n"
        "Sem explicações adicionais."
    )
    if ineditas:
        base += "\nAs questões devem ser INÉDITAS; não copie trechos literais do material."
    return base

def prompt_ingestao_questoes() -> str:
    return (
        "Você receberá páginas de um PDF contendo questões. Faça OCR e extraia as questões exatamente como estão.\n"
        "Inclua enunciado, alternativas e gabarito se disponível.\n"
        "Saída no formato JSONL, uma linha por questão, no seguinte schema:\n"
        "{"
        "\"numero\": <int>, "
        "\"enunciado\": \"texto\", "
        "\"alternativas\": [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"], "
        "\"correta\": \"A|B|C|D\" ou null"
        "}\n"
        "Sem comentários adicionais."
    )

# ==== GEMINI PROCESS ====
def gemini_processar_imagens(
    imagens: List[bytes],
    modelo: str,
    prompt: str,
    mime: Literal["image/png", "image/jpeg"] = "image/png",
    tamanho_lote: int = 4,
) -> str:
    if not imagens:
        raise ValueError("Nenhuma imagem fornecida.")
    cli = gemini_client()
    saidas: List[str] = []
    for i in range(0, len(imagens), tamanho_lote):
        lote = imagens[i:i+tamanho_lote]
        conteudos = [prompt] + [inline_img(b, mime) for b in lote]
        try:
            resp = cli.models.generate_content(model=modelo, contents=conteudos)
            texto = getattr(resp, "text", None) or str(resp)
            saidas.append(texto.strip())
        except Exception as e:
            saidas.append(f"[ERRO no lote {i//tamanho_lote + 1}: {e}]")
    return "\n\n".join(saidas)

# ==== SALVAR (Supabase DB) ====
def salvar_texto(tabela: str, conteudo: str, meta: Optional[dict] = None) -> dict:
    sb = supabase_client()
    payload = {"conteudo": conteudo, "meta": meta or {}}
    try:
        data = sb.table(tabela).insert(payload).execute()
        return {"sucesso": True, "data": data.data}
    except Exception as e:
        raise RuntimeError(f"Falha ao salvar no Supabase: {e}")

# ==== SCHEMAS ====
class BasePDFParams(BaseModel):
    dpi: int = 200
    max_paginas: Optional[int] = 20
    como_png: bool = True
    qualidade_jpeg: int = 85
    modelo: str = DEFAULT_MODEL
    tamanho_lote: int = 4

class OCRBucketReq(BasePDFParams):
    caminho_no_bucket: str
    bucket: Optional[str] = None

class GenBucketReq(BasePDFParams):
    caminho_no_bucket: str
    bucket: Optional[str] = None
    serie: str
    qtd_questoes: int = 10
    ineditas: bool = True

class IngestBucketReq(BasePDFParams):
    caminho_no_bucket: str
    bucket: Optional[str] = None

# ---- GCS ----
class GCSBaseReq(BasePDFParams):
    bucket: Optional[str] = None
    blob_path: str

class GCSGenReq(GCSBaseReq):
    serie: str
    qtd_questoes: int = 10
    ineditas: bool = True

class SaveReq(BaseModel):
    tabela: str = "conteudos"
    conteudo: str
    meta: Optional[dict] = None

class RespTexto(BaseModel):
    texto: str
    paginas_processadas: int

class RespSalvar(BaseModel):
    sucesso: bool
    data: Optional[dict] = None

# ==== FASTAPI APP ====
app = FastAPI(
    title="EducaAI API — OCR, Geração e Ingestão de Questões (Supabase + GCS)",
    version="4.0.0",
)

@app.get("/healthcheck")
def healthcheck():
    return {
        "status": "ok",
        "supabase_configurado": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
        "gcs_bucket_default": GCS_DEFAULT_BUCKET or None,
        "gemini_configurado": bool(GEMINI_API_KEY),
        "modelo_padrao": DEFAULT_MODEL,
    }

# ---------- OCR (Supabase) ----------
@app.post("/document-processing/ocr", response_model=RespTexto)
def processar_ocr(req: OCRBucketReq):
    try:
        pdf = baixar_pdf_bucket(req.caminho_no_bucket, req.bucket or SUPABASE_BUCKET)
        imgs = pdf_para_imagens(pdf, req.dpi, req.max_paginas, req.como_png, req.qualidade_jpeg)
        if not imgs:
            raise HTTPException(status_code=422, detail="Nenhuma página renderizada.")
        mime = "image/png" if req.como_png else "image/jpeg"
        texto = gemini_processar_imagens(imgs, req.modelo, prompt_ocr(), mime, req.tamanho_lote)
        return RespTexto(texto=texto, paginas_processadas=len(imgs))
    except Exception as e:
        log.exception("Erro em /document-processing/ocr")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- GERAÇÃO (Supabase) ----------
@app.post("/question-generation", response_model=RespTexto)
def gerar_questoes(req: GenBucketReq):
    try:
        pdf = baixar_pdf_bucket(req.caminho_no_bucket, req.bucket or SUPABASE_BUCKET)
        imgs = pdf_para_imagens(pdf, req.dpi, req.max_paginas, req.como_png, req.qualidade_jpeg)
        if not imgs:
            raise HTTPException(status_code=422, detail="Nenhuma página renderizada.")
        mime = "image/png" if req.como_png else "image/jpeg"
        prompt = prompt_gerar_questoes(req.serie, req.qtd_questoes, req.ineditas)
        texto = gemini_processar_imagens(imgs, req.modelo, prompt, mime, req.tamanho_lote)
        return RespTexto(texto=texto, paginas_processadas=len(imgs))
    except Exception as e:
        log.exception("Erro em /question-generation")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- INGESTÃO (Supabase) ----------
@app.post("/question-ingestion", response_model=RespTexto)
def ingestao_questoes(req: IngestBucketReq):
    try:
        pdf = baixar_pdf_bucket(req.caminho_no_bucket, req.bucket or SUPABASE_BUCKET)
        imgs = pdf_para_imagens(pdf, req.dpi, req.max_paginas, req.como_png, req.qualidade_jpeg)
        if not imgs:
            raise HTTPException(status_code=422, detail="Nenhuma página renderizada.")
        mime = "image/png" if req.como_png else "image/jpeg"
        texto = gemini_processar_imagens(imgs, req.modelo, prompt_ingestao_questoes(), mime, req.tamanho_lote)
        return RespTexto(texto=texto, paginas_processadas=len(imgs))
    except Exception as e:
        log.exception("Erro em /question-ingestion")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- OCR (GCS) ----------
@app.post("/document-processing/ocr/gcs", response_model=RespTexto)
def processar_ocr_gcs(req: GCSBaseReq):
    try:
        bucket = req.bucket or GCS_DEFAULT_BUCKET
        if not bucket:
            raise HTTPException(status_code=400, detail="Bucket do GCS não informado.")
        pdf_bytes = gcs_baixar_pdf_bytes(bucket, req.blob_path)

        imgs = pdf_para_imagens(pdf_bytes, req.dpi, req.max_paginas, req.como_png, req.qualidade_jpeg)
        if not imgs:
            raise HTTPException(status_code=422, detail="Nenhuma página renderizada.")
        mime = "image/png" if req.como_png else "image/jpeg"
        texto = gemini_processar_imagens(imgs, req.modelo, prompt_ocr(), mime, req.tamanho_lote)
        return RespTexto(texto=texto, paginas_processadas=len(imgs))
    except Exception as e:
        log.exception("Erro em /document-processing/ocr/gcs")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- GERAÇÃO (GCS) ----------
@app.post("/question-generation/gcs", response_model=RespTexto)
def gerar_questoes_gcs(req: GCSGenReq):
    try:
        bucket = req.bucket or GCS_DEFAULT_BUCKET
        if not bucket:
            raise HTTPException(status_code=400, detail="Bucket do GCS não informado.")
        pdf_bytes = gcs_baixar_pdf_bytes(bucket, req.blob_path)

        imgs = pdf_para_imagens(pdf_bytes, req.dpi, req.max_paginas, req.como_png, req.qualidade_jpeg)
        if not imgs:
            raise HTTPException(status_code=422, detail="Nenhuma página renderizada.")
        mime = "image/png" if req.como_png else "image/jpeg"
        prompt = prompt_gerar_questoes(req.serie, req.qtd_questoes, req.ineditas)
        texto = gemini_processar_imagens(imgs, req.modelo, prompt, mime, req.tamanho_lote)
        return RespTexto(texto=texto, paginas_processadas=len(imgs))
    except Exception as e:
        log.exception("Erro em /question-generation/gcs")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- INGESTÃO (GCS) ----------
@app.post("/question-ingestion/gcs", response_model=RespTexto)
def ingestao_questoes_gcs(req: GCSBaseReq):
    try:
        bucket = req.bucket or GCS_DEFAULT_BUCKET
        if not bucket:
            raise HTTPException(status_code=400, detail="Bucket do GCS não informado.")
        pdf_bytes = gcs_baixar_pdf_bytes(bucket, req.blob_path)

        imgs = pdf_para_imagens(pdf_bytes, req.dpi, req.max_paginas, req.como_png, req.qualidade_jpeg)
        if not imgs:
            raise HTTPException(status_code=422, detail="Nenhuma página renderizada.")
        mime = "image/png" if req.como_png else "image/jpeg"
        texto = gemini_processar_imagens(imgs, req.modelo, prompt_ingestao_questoes(), mime, req.tamanho_lote)
        return RespTexto(texto=texto, paginas_processadas=len(imgs))
    except Exception as e:
        log.exception("Erro em /question-ingestion/gcs")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- SALVAR RESULTADOS (Supabase DB) ----------
@app.post("/content-storage/save", response_model=RespSalvar)
def salvar_conteudo(req: SaveReq):
    try:
        res = salvar_texto(req.tabela, req.conteudo, req.meta)
        return RespSalvar(sucesso=True, data=res)
    except Exception as e:
        log.exception("Erro em /content-storage/save")
        raise HTTPException(status_code=500, detail=str(e))
