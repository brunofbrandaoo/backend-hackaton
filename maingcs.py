# main.py
import os
import io
import json
import base64
import logging
from typing import List, Optional, Literal, Dict, Any
import re

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel, Field
from supabase import create_client, Client
from google import genai
from dotenv import load_dotenv
from datetime import timedelta
from google.cloud.storage.blob import Blob

load_dotenv()
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

def gcs_upload_fileobj(
    bucket_name: str,
    blob_path: str,
    fileobj,
    content_type: Optional[str] = None,
    make_public: bool = False,
    cache_control: Optional[str] = None,
) -> dict:
    """
    Faz upload de um arquivo (file-like) para o GCS.
    """
    cli = gcs_client()
    bucket = cli.bucket(bucket_name)
    blob: Blob = bucket.blob(blob_path)

    # Para uploads grandes, pode ajustar chunk_size (ex.: 8 MB)
    # blob.chunk_size = 8 * 1024 * 1024

    blob.upload_from_file(fileobj, content_type=content_type, rewind=True)
    if cache_control:
        blob.cache_control = cache_control
        blob.patch()

    public_url = None
    if make_public:
        blob.make_public()
        public_url = blob.public_url

    return {
        "bucket": bucket_name,
        "path": blob_path,
        "size": blob.size,
        "content_type": content_type,
        "md5_hash": blob.md5_hash,
        "crc32c": blob.crc32c,
        "public_url": public_url,
        "gs_uri": f"gs://{bucket_name}/{blob_path}",
    }


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
            "Formato obrigatório de saída (JSON):\n"
            "{\n"
            '  "questions": [\n'
            "    {\n"
            '      "statement": "Enunciado da questão",\n'
            '      "alternatives": [\n'
            '        {"letter": "A", "text": "Alternativa A"},\n'
            '        {"letter": "B", "text": "Alternativa B"},\n'
            '        {"letter": "C", "text": "Alternativa C"},\n'
            '        {"letter": "D", "text": "Alternativa D"}\n'
            "      ],\n"
            '      "correct_answer": "A",\n'
            '      "difficulty": "medium",\n'
            '      "topic": "Nome do assunto/conteúdo extraído do material para a questão"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Retorne APENAS o JSON, sem explicações adicionais."
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
        "Formato obrigatório de saída (JSON):\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "statement": "Enunciado da questão",\n'
        '      "alternatives": [\n'
        '        {"letter": "A", "text": "Alternativa A"},\n'
        '        {"letter": "B", "text": "Alternativa B"},\n'
        '        {"letter": "C", "text": "Alternativa C"},\n'
        '        {"letter": "D", "text": "Alternativa D"}\n'
        "      ],\n"
        '      "correct_answer": "A",\n'
        '      "difficulty": "medium",\n'
        '      "topic": "Nome do assunto/conteúdo extraído do material para a questão"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Retorne APENAS o JSON, sem explicações adicionais."
    )
    if ineditas:
        base += "\nAs questões devem ser INÉDITAS; não copie trechos literais do material."
    return base

def prompt_ingestao_questoes() -> str:
    return (
        "Você receberá páginas de um PDF contendo questões. Faça OCR e extraia as questões exatamente como estão.\n"
        "Inclua enunciado, alternativas e gabarito se disponível.\n"
        "Saída no formato JSON:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "statement": "Enunciado da questão",\n'
        '      "alternatives": [\n'
        '        {"letter": "A", "text": "Alternativa A"},\n'
        '        {"letter": "B", "text": "Alternativa B"},\n'
        '        {"letter": "C", "text": "Alternativa C"},\n'
        '        {"letter": "D", "text": "Alternativa D"}\n'
        "      ],\n"
        '      "correct_answer": "A",\n'
        '      "difficulty": "medium",\n'
        '      "topic": "Nome do assunto/conteúdo da questão"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Retorne APENAS o JSON, sem comentários adicionais."
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

# ==== PARSEAR SAÍDA DO LLM ====
import uuid
import re
from datetime import datetime

def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """
    Varre o texto e extrai TODOS os objetos JSON bem-formados usando contagem de chaves.
    Suporta casos em que o LLM devolve {..}{..} (vários objetos colados) ou
    texto extra antes/depois.
    """
    # Remove cercas de código
    t = text.strip()
    # tira blocos ```json ... ``` ou ``` ... ```
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    objs = []
    stack = 0
    start = None
    for i, ch in enumerate(t):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    chunk = t[start:i+1]
                    try:
                        obj = json.loads(chunk)
                        objs.append(obj)
                    except json.JSONDecodeError:
                        # tenta limpar vírgulas finais, etc.
                        try:
                            cleaned = re.sub(r",\s*}", "}", chunk)
                            obj = json.loads(cleaned)
                            objs.append(obj)
                        except Exception:
                            pass
                    start = None
    return objs

def parsear_questoes_llm(texto_llm: str) -> dict:
    """
    Lê a saída do LLM e retorna UM dicionário com "questions" mescladas.
    Aceita múltiplos JSONs no texto e agrega tudo em {"questions": [...] }.
    """
    objetos = _extract_json_objects(texto_llm)
    if not objetos:
        # Última tentativa: talvez seja um único JSON válido com lixo lateral
        try:
            return json.loads(texto_llm.strip())
        except json.JSONDecodeError as e:
            raise ValueError(f"Falha ao parsear JSON do LLM: {e}")

    if len(objetos) == 1:
        # normal: um único JSON
        return objetos[0]

    # múltiplos JSONs → mescla arrays de "questions"
    merged = {"questions": []}
    for obj in objetos:
        qs = obj.get("questions", [])
        if isinstance(qs, list):
            merged["questions"].extend(qs)
    return merged

def transformar_para_db_questoes(
    texto_llm: str,
    created_by: Optional[str] = None,
    subject_id: Optional[str] = None,
    source_material_id: Optional[str] = None,
) -> List[dict]:
    """
    Transforma a saída do LLM em formato JSON para o schema da tabela 'questions'.
    
    Args:
        texto_llm: Texto JSON retornado pelo LLM
        created_by: UUID do professor que criou
        subject_id: UUID da matéria
        source_material_id: UUID do material de origem (opcional)
    
    Returns:
        Lista de dicionários prontos para inserção na tabela 'questions'
    """
    dados = parsear_questoes_llm(texto_llm)
    questoes_db = []
    
    for q in dados.get("questions", []):
        # Montar o JSONB da questão no formato esperado pelo banco
        question_jsonb = {
            "statement": q.get("statement", ""),
            "alternatives": q.get("alternatives", []),
            "correct_answer": q.get("correct_answer", ""),
            "metadata": {
                "created_from_llm": True,
                "raw_response": q
            }
        }
        
        # Criar registro para o banco
        questao_db = {
            "source_material_id": source_material_id,
            "created_by": created_by,
            "subject_id": subject_id,
            "topic_id": None,  # Será necessário mapear o tópico posteriormente
            "difficulty": q.get("difficulty", "medium"),
            "available": False,
            "question": question_jsonb
        }
        
        questoes_db.append(questao_db)
    
    return questoes_db

def salvar_material_db(
    teacher_id: str,
    subject_id: str,
    material_data: dict,
) -> dict:
    """
    Salva material na tabela 'materials' do banco de dados.
    
    Args:
        teacher_id: UUID do professor
        subject_id: UUID da matéria
        material_data: Dados do material (será salvo como JSONB)
    
    Returns:
        Dicionário com o resultado da inserção
    """
    sb = supabase_client()
    payload = {
        "teacher_id": teacher_id,
        "subject_id": subject_id,
        "material": material_data
    }
    try:
        data = sb.table("materials").insert(payload).execute()
        return {"sucesso": True, "data": data.data}
    except Exception as e:
        raise RuntimeError(f"Falha ao salvar material no Supabase: {e}")

def salvar_questoes_db(questoes: List[dict]) -> dict:
    """
    Salva questões na tabela 'questions' do banco de dados.
    
    Args:
        questoes: Lista de dicionários com os dados das questões
    
    Returns:
        Dicionário com o resultado da inserção
    """
    sb = supabase_client()
    try:
        data = sb.table("questions").insert(questoes).execute()
        return {"sucesso": True, "data": data.data, "quantidade": len(questoes)}
    except Exception as e:
        raise RuntimeError(f"Falha ao salvar questões no Supabase: {e}")

# ==== SALVAR (Supabase DB) - LEGACY ====
def salvar_texto(tabela: str, conteudo: str, meta: Optional[dict] = None) -> dict:
    sb = supabase_client()
    payload = {"conteudo": conteudo, "meta": meta or {}}
    try:
        data = sb.table(tabela).insert(payload).execute()
        return {"sucesso": True, "data": data.data}
    except Exception as e:
        raise RuntimeError(f"Falha ao salvar no Supabase: {e}")
    
def _sb() -> Client:
    return supabase_client()

def get_question_by_id(qid: str) -> Optional[dict]:
    sb = _sb()
    res = sb.table("questions").select("*").eq("id", qid).limit(1).execute()
    rows = res.data or []
    return rows[0] if rows else None

def normalize_correct_answer(letter: str) -> str:
    return (letter or "").strip().upper()


# ==== SCHEMAS ====

# --- SCHEMAS PARA MODERAÇÃO/VALIDAÇÃO DE QUESTÕES ---
class QuestionOut(BaseModel):
    id: Any
    available: bool
    subject_id: Optional[str] = None
    created_by: Optional[str] = None
    source_material_id: Optional[str] = None
    difficulty: Optional[str] = None
    question: Dict[str, Any]

class ValidateQuestionReq(BaseModel):
    correct_answer: str = Field(..., description="Letra A/B/C/D/E")
    available: bool = True
    # campos opcionais para ajustes pelo professor
    statement: Optional[str] = None
    alternatives: Optional[List[Dict[str, Any]]] = None  # [{letter, text}, ...]
    difficulty: Optional[str] = None
    topic: Optional[str] = None

class PendingQueryParams(BaseModel):
    subject_id: Optional[str] = None
    created_by: Optional[str] = None
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)

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

class SalvarQuestoesReq(BaseModel):
    texto_llm: str
    created_by: Optional[str] = Field(None, description="UUID do professor")
    subject_id: Optional[str] = Field(None, description="UUID da matéria")
    source_material_id: Optional[str] = Field(None, description="UUID do material de origem")

class SalvarMaterialReq(BaseModel):
    teacher_id: Optional[str] = Field(None, description="UUID do professor")
    subject_id: Optional[str] = Field(None, description="UUID da matéria")
    material_data: dict = Field(..., description="Dados do material em formato livre")

class RespSalvarQuestoes(BaseModel):
    sucesso: bool
    quantidade: int
    questoes: List[dict] = []

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

# ---------- SALVAR QUESTÕES NO BANCO ----------
@app.post("/questions/save", response_model=RespSalvarQuestoes)
def salvar_questoes_endpoint(req: SalvarQuestoesReq):
    """
    Parseia a saída JSON do LLM e salva as questões na tabela 'questions'.
    """
    try:
        # Transformar saída do LLM para formato do banco
        questoes = transformar_para_db_questoes(
            texto_llm=req.texto_llm,
            created_by=req.created_by,
            subject_id=req.subject_id,
            source_material_id=req.source_material_id
        )
        
        # Salvar no banco
        resultado = salvar_questoes_db(questoes)
        
        return RespSalvarQuestoes(
            sucesso=True,
            quantidade=len(questoes),
            questoes=resultado.get("data", [])
        )
    except Exception as e:
        log.exception("Erro em /questions/save")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- SALVAR MATERIAL NO BANCO ----------
@app.post("/materials/save", response_model=RespSalvar)
def salvar_material_endpoint(req: SalvarMaterialReq):
    """
    Salva material na tabela 'materials'.
    """
    try:
        resultado = salvar_material_db(
            teacher_id=req.teacher_id,
            subject_id=req.subject_id,
            material_data=req.material_data
        )
        return RespSalvar(sucesso=True, data=resultado.get("data"))
    except Exception as e:
        log.exception("Erro em /materials/save")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- GERAR E SALVAR QUESTÕES (FLUXO COMPLETO) ----------
@app.post("/question-generation/complete", response_model=RespSalvarQuestoes)
def gerar_e_salvar_questoes(req: GenBucketReq, created_by: Optional[str] = None,
    subject_id: Optional[str] = None, source_material_id: Optional[str] = None):
    """
    Fluxo completo: baixa PDF, gera questões com LLM e salva diretamente no banco.
    """
    try:
        # 1. Baixar e processar PDF
        pdf = gcs_baixar_pdf_bytes(req.bucket, req.caminho_no_bucket)
        imgs = pdf_para_imagens(pdf, req.dpi, req.max_paginas, req.como_png, req.qualidade_jpeg)
        if not imgs:
            raise HTTPException(status_code=422, detail="Nenhuma página renderizada.")
        
        # 2. Gerar questões com LLM
        mime = "image/png" if req.como_png else "image/jpeg"
        prompt = prompt_gerar_questoes(req.serie, req.qtd_questoes, req.ineditas)
        texto_llm = gemini_processar_imagens(imgs, req.modelo, prompt, mime, req.tamanho_lote)
        
        # 3. Transformar e salvar no banco
        questoes = transformar_para_db_questoes(
            texto_llm=texto_llm,
            created_by=created_by,
            subject_id=subject_id,
            source_material_id=source_material_id
        )
        resultado = salvar_questoes_db(questoes)
        
        return RespSalvarQuestoes(
            sucesso=True,
            quantidade=len(questoes),
            questoes=resultado.get("data", [])
        )
    except Exception as e:
        log.exception("Erro em /question-generation/complete")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/storage/gcs/upload")
async def upload_gcs_file(
    file: UploadFile = File(...),
    blob_path: str = Form(..., description="Caminho no bucket, ex: materiais/2025/prova.pdf"),
    bucket: Optional[str] = Form(None),
    make_public: bool = Form(False),
    cache_control: Optional[str] = Form(None),
):
    """
    Upload direto de um arquivo (multipart/form-data) para o GCS.
    """
    try:
        target_bucket = bucket or GCS_DEFAULT_BUCKET
        if not target_bucket:
            raise HTTPException(status_code=400, detail="Bucket do GCS não informado.")

        # Lê o stream e envia
        # Dica: para arquivos grandes, é melhor usar signed URL (abaixo)
        file.file.seek(0)
        result = gcs_upload_fileobj(
            bucket_name=target_bucket,
            blob_path=blob_path,
            fileobj=file.file,
            content_type=file.content_type or "application/octet-stream",
            make_public=make_public,
            cache_control=cache_control,
        )
        return {"ok": True, "file": result}
    except Exception as e:
        log.exception("Erro em /storage/gcs/upload")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/question-ingestion/complete/gcs", response_model=RespSalvarQuestoes)
def ingestao_e_salvar_questoes_gcs(
    req: GCSBaseReq,
    created_by: Optional[str] = None,
    subject_id: Optional[str] = None,
    source_material_id: Optional[str] = None,
):
    """
    Fluxo completo (GCS): baixa o PDF do GCS, extrai questões exatamente como no material (OCR),
    parseia o JSON e salva na tabela 'questions'.
    """
    try:
        # 1) Baixar PDF do GCS
        bucket = req.bucket or GCS_DEFAULT_BUCKET
        if not bucket:
            raise HTTPException(status_code=400, detail="Bucket do GCS não informado.")
        pdf = gcs_baixar_pdf_bytes(bucket, req.blob_path)

        # 2) Renderizar em imagens
        imgs = pdf_para_imagens(pdf, req.dpi, req.max_paginas, req.como_png, req.qualidade_jpeg)
        if not imgs:
            raise HTTPException(status_code=422, detail="Nenhuma página renderizada.")

        # 3) Rodar OCR + extração com prompt de ingestão
        mime = "image/png" if req.como_png else "image/jpeg"
        texto_llm = gemini_processar_imagens(
            imagens=imgs,
            modelo=req.modelo,
            prompt=prompt_ingestao_questoes(),
            mime=mime,
            tamanho_lote=req.tamanho_lote,
        )

        # 4) Transformar para schema do banco
        questoes = transformar_para_db_questoes(
            texto_llm=texto_llm,
            created_by=created_by,
            subject_id=subject_id,
            source_material_id=source_material_id,
        )

        # 5) Salvar
        resultado = salvar_questoes_db(questoes)

        return RespSalvarQuestoes(
            sucesso=True,
            quantidade=len(questoes),
            questoes=resultado.get("data", []),
        )
    except Exception as e:
        log.exception("Erro em /question-ingestion/complete/gcs")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/questions/pending", response_model=List[QuestionOut])
def list_pending_questions(
    subject_id: Optional[str] = Query(None),
    created_by: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Lista questões que ainda NÃO estão disponíveis para alunos (available = false).
    Suporta filtros por subject_id e created_by, além de paginação (limit/offset).
    """
    try:
        sb = _sb()
        q = sb.table("questions").select("*").eq("available", False)

        if subject_id:
            q = q.eq("subject_id", subject_id)
        if created_by:
            q = q.eq("created_by", created_by)

        # ordena por criação desc (se tiver created_at); senão por id desc
        try:
            q = q.order("created_at", desc=True)
        except Exception:
            q = q.order("id", desc=True)

        if offset:
            q = q.range(offset, offset + limit - 1)
        else:
            q = q.limit(limit)

        res = q.execute()
        return (res.data or [])
    except Exception as e:
        log.exception("Erro em /questions/pending")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/questions/{question_id}", response_model=QuestionOut)
def validate_question(question_id: str, body: ValidateQuestionReq):
    """
    Professor valida/ajusta uma questão:
    - define 'question.correct_answer'
    - (opcional) ajusta enunciado, alternativas, difficulty, topic
    - define 'available' (default True)
    """
    try:
        # 1) Busca atual
        row = get_question_by_id(question_id)
        if not row:
            raise HTTPException(status_code=404, detail="Questão não encontrada.")

        qobj = row.get("question") or {}
        # 2) Normaliza e valida a resposta
        corr = normalize_correct_answer(body.correct_answer)
        if corr not in {"A", "B", "C", "D", "E"}:
            raise HTTPException(status_code=422, detail="correct_answer deve ser A, B, C, D ou E.")

        # 3) Aplica ajustes opcionais
        if body.statement is not None:
            qobj["statement"] = body.statement
        if body.alternatives is not None:
            # valida formato mínimo
            if not isinstance(body.alternatives, list) or len(body.alternatives) < 2:
                raise HTTPException(status_code=422, detail="alternatives deve ser lista com 2+ itens.")
            qobj["alternatives"] = body.alternatives
        if body.topic is not None:
            # topic pode morar dentro de question.metadata ou na raiz do question; segue seu prompt:
            qobj["topic"] = body.topic

        qobj["correct_answer"] = corr

        # 4) Difficulty opcional (na coluna de linha, não só no JSON)
        new_difficulty = body.difficulty if body.difficulty is not None else row.get("difficulty")

        # 5) Update no Supabase
        sb = _sb()
        updated = sb.table("questions").update({
            "question": qobj,
            "available": bool(body.available),
            "difficulty": new_difficulty
        }).eq("id", question_id).execute()

        data = (updated.data or [])
        if not data:
            raise HTTPException(status_code=500, detail="Falha ao atualizar a questão.")
        return data[0]
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Erro em PATCH /questions/{id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/questions/{question_id}")
def delete_question(question_id: str):
    """
    Remove uma questão (quando está mal formulada ou sem resposta correta).
    """
    try:
        # opcional: garantir que existe antes
        row = get_question_by_id(question_id)
        if not row:
            raise HTTPException(status_code=404, detail="Questão não encontrada.")

        sb = _sb()
        res = sb.table("questions").delete().eq("id", question_id).execute()
        count = len(res.data or [])
        return {"deleted": count, "id": question_id}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Erro em DELETE /questions/{id}")
        raise HTTPException(status_code=500, detail=str(e))
