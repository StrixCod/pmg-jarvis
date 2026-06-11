from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import FastAPI, WebSocket, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import json
import logging
import re
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import sqlite3
from pathlib import Path

# ============================================================================
# 1. CONFIGURATION & IMPORTS
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "L9KyHhbmAEYTDmEOwIUcPCBkpWf6xeS3")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pmg_jarvis.db")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")

# Dossier où la mémoire de Jarvis est stockée en fichiers texte bruts
MEMORY_DIR = Path("jarvis_memory")
MEMORY_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="PMG Jarvis API",
    description="Assistant IA personnel - Pony Mounted Games",
    version="2.0.0"
)

# ============================================================================
# 2. MIDDLEWARE
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# ============================================================================
# 3. MODÈLES PYDANTIC
# ============================================================================

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    is_voice: bool = False
    id: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    user_id: str
    is_voice: bool = False
    language: str = "fr"
    include_sources: bool = True

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    timestamp: datetime
    message_id: str
    is_voice_response: bool = False

class PMGEntry(BaseModel):
    title: str
    content: str
    category: str
    source: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = []

class PMGQuery(BaseModel):
    query: str
    user_id: str
    return_source: bool = True

class PMGResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    from_pmg_memory: bool

class SettingsUpdate(BaseModel):
    assistant_name: str = "Jarvis"
    voice_speed: float = 1.0
    voice_volume: float = 1.0
    voice_id: str = "default"
    language: str = "fr"
    theme: str = "dark"

class UserSettings(BaseModel):
    user_id: str
    assistant_name: str
    voice_speed: float
    voice_volume: float
    voice_id: str
    language: str
    theme: str
    created_at: datetime
    updated_at: datetime

# ============================================================================
# 4. FILE MEMORY SYSTEM — Système de mémoire en fichiers texte bruts
# ============================================================================

class FileMemorySystem:
    """
    Stocke et lit la mémoire de Jarvis dans des fichiers .txt bruts sur disque.
    Chaque document uploadé devient un fichier lisible et indexable.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.index_file = memory_dir / "_index.json"
        self._ensure_index()

    def _ensure_index(self):
        """Crée le fichier index s'il n'existe pas"""
        if not self.index_file.exists():
            self._save_index({})

    def _load_index(self) -> dict:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, index: dict):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2, default=str)

    def _sanitize_filename(self, name: str) -> str:
        """Nettoie le nom de fichier"""
        name = re.sub(r'[^\w\s\-.]', '_', name)
        name = name[:80].strip()
        return name or "document"

    def save_document(self, filename: str, content: str, category: str = "document", tags: list = None) -> str:
        """
        Sauvegarde un document en fichier texte brut.
        Retourne l'ID du fichier créé.
        """
        import uuid
        doc_id = str(uuid.uuid4())[:8]
        safe_name = self._sanitize_filename(Path(filename).stem)
        file_path = self.memory_dir / f"{doc_id}_{safe_name}.txt"

        # Écriture du fichier texte brut avec en-tête lisible par l'IA
        header = (
            f"=== DOCUMENT JARVIS ===\n"
            f"ID: {doc_id}\n"
            f"Source: {filename}\n"
            f"Catégorie: {category}\n"
            f"Tags: {', '.join(tags or [])}\n"
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"======================\n\n"
        )

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(header + content)

        # Mise à jour de l'index
        index = self._load_index()
        index[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "file_path": str(file_path),
            "category": category,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "size_chars": len(content),
            "preview": content[:200].replace("\n", " ")
        }
        self._save_index(index)

        logger.info(f"✅ Document sauvegardé: {file_path} ({len(content)} caractères)")
        return doc_id

    def search_documents(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Cherche dans tous les fichiers texte de la mémoire.
        Retourne les passages pertinents pour la question posée.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results = []

        index = self._load_index()

        for doc_id, meta in index.items():
            file_path = Path(meta["file_path"])
            if not file_path.exists():
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            content_lower = content.lower()

            # Score de pertinence basé sur le nombre de mots trouvés
            score = 0
            for word in query_words:
                if len(word) > 2:  # ignore les mots trop courts
                    count = content_lower.count(word)
                    score += count

            if score == 0:
                continue

            # Extrait les passages les plus pertinents
            passages = self._extract_relevant_passages(content, query_words)

            results.append({
                "id": doc_id,
                "filename": meta["filename"],
                "category": meta["category"],
                "score": score,
                "passages": passages,
                "file_path": str(file_path)
            })

        # Trie par score décroissant
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    def _extract_relevant_passages(self, content: str, query_words: set, passage_size: int = 400) -> List[str]:
        """Extrait les paragraphes contenant les mots clés"""
        paragraphs = content.split("\n\n")
        scored_paragraphs = []

        for para in paragraphs:
            if len(para.strip()) < 20:
                continue
            para_lower = para.lower()
            score = sum(1 for w in query_words if len(w) > 2 and w in para_lower)
            if score > 0:
                scored_paragraphs.append((score, para.strip()))

        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)

        passages = []
        total_chars = 0
        for _, para in scored_paragraphs:
            if total_chars + len(para) > 2000:
                break
            passages.append(para)
            total_chars += len(para)

        return passages[:3]

    def get_all_documents(self) -> List[Dict]:
        """Retourne la liste de tous les documents en mémoire"""
        index = self._load_index()
        docs = list(index.values())
        docs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return docs

    def delete_document(self, doc_id: str) -> bool:
        """Supprime un document de la mémoire"""
        index = self._load_index()
        if doc_id not in index:
            return False

        file_path = Path(index[doc_id]["file_path"])
        if file_path.exists():
            file_path.unlink()

        del index[doc_id]
        self._save_index(index)
        return True

    def get_full_content(self, doc_id: str) -> Optional[str]:
        """Retourne le contenu complet d'un document"""
        index = self._load_index()
        if doc_id not in index:
            return None
        file_path = Path(index[doc_id]["file_path"])
        if not file_path.exists():
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()


# Instance globale du système de mémoire fichiers
file_memory = FileMemorySystem(MEMORY_DIR)

# ============================================================================
# 5. DATABASE LAYER
# ============================================================================

class Database:
    def __init__(self, db_path: str = "pmg_jarvis.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                messages TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pmg_memory (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                source TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                tags TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                assistant_name TEXT,
                voice_speed REAL,
                voice_volume REAL,
                voice_id TEXT,
                language TEXT,
                theme TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                message_content TEXT,
                response_content TEXT,
                is_voice BOOLEAN,
                created_at TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

db = Database()

# ============================================================================
# 6. SERVICES LAYER
# ============================================================================

class MistralService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "mistral-small"
        self.base_url = "https://api.mistral.ai/v1"

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0
                )

                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Mistral API error: {response.text}")
                    return "Erreur lors de la communication avec l'IA"

        except Exception as e:
            logger.error(f"Mistral service error: {e}")
            return f"Erreur: {str(e)}"


class PMGMemory:
    def __init__(self, db: Database):
        self.db = db

    def add_entry(self, entry: PMGEntry) -> str:
        import uuid
        entry_id = str(uuid.uuid4())

        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO pmg_memory
            (id, title, content, category, source, created_at, updated_at, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry_id,
            entry.title,
            entry.content,
            entry.category,
            entry.source or "Manual Entry",
            entry.created_at,
            entry.updated_at,
            json.dumps(entry.tags)
        ))

        conn.commit()
        conn.close()
        return entry_id

    def search(self, query: str, category: Optional[str] = None) -> List[Dict]:
        conn = self.db.get_connection()
        cursor = conn.cursor()

        if category:
            cursor.execute("""
                SELECT id, title, content, category, source, created_at, tags
                FROM pmg_memory
                WHERE (title LIKE ? OR content LIKE ? OR tags LIKE ?)
                AND category = ?
                ORDER BY updated_at DESC
            """, (f"%{query}%", f"%{query}%", f"%{query}%", category))
        else:
            cursor.execute("""
                SELECT id, title, content, category, source, created_at, tags
                FROM pmg_memory
                WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
                ORDER BY updated_at DESC
            """, (f"%{query}%", f"%{query}%", f"%{query}%"))

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "title": r[1],
                "content": r[2],
                "category": r[3],
                "source": r[4],
                "created_at": r[5],
                "tags": json.loads(r[6]) if r[6] else []
            }
            for r in results
        ]

    def get_all_entries(self) -> List[Dict]:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, content, category, source, created_at, tags
            FROM pmg_memory
            ORDER BY created_at DESC
        """)
        results = cursor.fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "title": r[1],
                "content": r[2],
                "category": r[3],
                "source": r[4],
                "created_at": r[5],
                "tags": json.loads(r[6]) if r[6] else []
            }
            for r in results
        ]


class HistoryService:
    def __init__(self, db: Database):
        self.db = db

    def save_message(self, user_id: str, message: str, response: str, is_voice: bool = False):
        import uuid
        msg_id = str(uuid.uuid4())

        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO history
            (id, user_id, message_content, response_content, is_voice, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (msg_id, user_id, message, response, is_voice, datetime.now()))

        conn.commit()
        conn.close()

    def get_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, message_content, response_content, is_voice, created_at
            FROM history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (user_id, limit))

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "message": r[1],
                "response": r[2],
                "is_voice": bool(r[3]),
                "timestamp": r[4]
            }
            for r in reversed(results)
        ]

    def delete_message(self, message_id: str):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history WHERE id = ?", (message_id,))
        conn.commit()
        conn.close()


class SettingsService:
    def __init__(self, db: Database):
        self.db = db

    def get_settings(self, user_id: str) -> UserSettings:
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, assistant_name, voice_speed, voice_volume,
                   voice_id, language, theme, created_at, updated_at
            FROM user_settings
            WHERE user_id = ?
        """, (user_id,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return UserSettings(
                user_id=result[0],
                assistant_name=result[1],
                voice_speed=result[2],
                voice_volume=result[3],
                voice_id=result[4],
                language=result[5],
                theme=result[6],
                created_at=datetime.fromisoformat(str(result[7])),
                updated_at=datetime.fromisoformat(str(result[8]))
            )
        else:
            self.create_default_settings(user_id)
            return self.get_settings(user_id)

    def create_default_settings(self, user_id: str):
        conn = self.db.get_connection()
        cursor = conn.cursor()
        now = datetime.now()
        cursor.execute("""
            INSERT INTO user_settings
            (user_id, assistant_name, voice_speed, voice_volume, voice_id, language, theme, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, "Jarvis", 1.0, 1.0, "default", "fr", "dark", now, now))
        conn.commit()
        conn.close()

    def update_settings(self, user_id: str, settings: SettingsUpdate) -> UserSettings:
        conn = self.db.get_connection()
        cursor = conn.cursor()
        now = datetime.now()
        cursor.execute("""
            UPDATE user_settings
            SET assistant_name = ?, voice_speed = ?, voice_volume = ?,
                voice_id = ?, language = ?, theme = ?, updated_at = ?
            WHERE user_id = ?
        """, (
            settings.assistant_name,
            settings.voice_speed,
            settings.voice_volume,
            settings.voice_id,
            settings.language,
            settings.theme,
            now,
            user_id
        ))
        conn.commit()
        conn.close()
        return self.get_settings(user_id)


# ============================================================================
# 7. INITIALISATION
# ============================================================================

mistral_service = MistralService(MISTRAL_API_KEY)
pmg_memory = PMGMemory(db)
history_service = HistoryService(db)
settings_service = SettingsService(db)

# ============================================================================
# 8. FONCTION CENTRALE — Recherche dans la mémoire fichiers
# ============================================================================

def build_memory_context(query: str) -> tuple[str, List[str]]:
    """
    Cherche dans les fichiers de mémoire de Jarvis.
    Retourne (contexte_texte, liste_sources).
    """
    results = file_memory.search_documents(query, max_results=4)

    if not results:
        return "", []

    context_parts = []
    sources = []

    for r in results:
        source_name = r["filename"]
        sources.append(source_name)

        if r["passages"]:
            passages_text = "\n---\n".join(r["passages"])
            context_parts.append(
                f"📄 Source: {source_name}\n{passages_text}"
            )

    return "\n\n".join(context_parts), sources


# ============================================================================
# 9. PROMPT SYSTÈME AMÉLIORÉ
# ============================================================================

SYSTEM_PROMPT = """Tu es PMG Jarvis, un assistant IA personnel intelligent et professionnel.

RÈGLES DE FORMATAGE DES RÉPONSES :
- Sois clair, concis et structuré
- Utilise des listes à puces (•) pour les points multiples
- Utilise des titres courts en majuscules suivi de ":" pour les sections
- Sépare bien les idées avec des lignes vides
- Pour les chiffres et règles importantes, mets-les bien en évidence
- Évite les réponses en bloc de texte compact
- Réponds toujours en français sauf si on te parle dans une autre langue
- Maximum 3-4 phrases par paragraphe

COMPORTEMENT :
- Si tu as des informations dans ta mémoire sur le sujet, utilise-les EN PRIORITÉ
- Indique toujours la source quand tu utilises ta mémoire
- Si tu n'as pas l'information, dis-le clairement et propose une aide alternative
- Pour les messages vocaux, réponds de façon plus concise et conversationnelle (pas de listes)
- Tu es spécialisé dans le Pony Mounted Games (PMG) mais tu peux aider sur tous les sujets"""


def build_voice_prompt() -> str:
    return """Tu es PMG Jarvis. Réponds de façon naturelle et conversationnelle, comme si tu parlais à voix haute.
Pas de listes ni de symboles. Phrases courtes. Réponds en français. Maximum 3-4 phrases."""


# ============================================================================
# 10. ENDPOINTS API
# ============================================================================

@app.get("/")
async def root():
    docs = file_memory.get_all_documents()
    return {
        "name": "PMG Jarvis API",
        "version": "2.0.0",
        "status": "online",
        "memory_documents": len(docs),
        "memory_dir": str(MEMORY_DIR.absolute()),
        "timestamp": datetime.now()
    }


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    import uuid
    message_id = str(uuid.uuid4())

    try:
        # 1. Cherche d'abord dans la mémoire fichiers
        memory_context, memory_sources = build_memory_context(request.message)

        # 2. Aussi chercher dans la mémoire SQLite PMG
        pmg_results = pmg_memory.search(request.message)
        pmg_sources = []
        pmg_context = ""
        if pmg_results:
            pmg_context = "\n\n".join([
                f"[{r['category'].upper()}] {r['title']}: {r['content']}"
                for r in pmg_results[:3]
            ])
            pmg_sources = [r["source"] or r["title"] for r in pmg_results]

        # 3. Construction du contexte complet
        full_context = ""
        if memory_context:
            full_context += f"\n\n=== MÉMOIRE JARVIS (fichiers) ===\n{memory_context}"
        if pmg_context:
            full_context += f"\n\n=== BASE PMG ===\n{pmg_context}"

        all_sources = list(set(memory_sources + pmg_sources))

        # 4. Choix du prompt selon vocal ou texte
        system_content = (build_voice_prompt() if request.is_voice else SYSTEM_PROMPT)
        if full_context:
            system_content += f"\n\nINFORMATIONS DISPONIBLES DANS TA MÉMOIRE :{full_context}\n\nUtilise ces informations pour répondre à la question."

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": request.message}
        ]

        response_text = await mistral_service.chat_completion(messages)

        history_service.save_message(
            user_id=request.user_id,
            message=request.message,
            response=response_text,
            is_voice=request.is_voice
        )

        return ChatResponse(
            response=response_text,
            sources=all_sources if request.include_sources else None,
            timestamp=datetime.now(),
            message_id=message_id,
            is_voice_response=request.is_voice
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pmg/upload-document")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = "user_default"
):
    """
    Upload, extrait et sauvegarde un document dans la mémoire fichiers de Jarvis.
    Le contenu est stocké en texte brut lisible et interrogeable.
    """
    try:
        content_bytes = await file.read()
        filename = file.filename or "unknown"
        extension = filename.split('.')[-1].lower()
        extracted_text = ""

        # --- Extraction du texte ---
        if extension == 'pdf':
            try:
                import io
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
                pages_text = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages_text.append(f"[Page {i+1}]\n{page_text}")
                extracted_text = "\n\n".join(pages_text)
            except ImportError:
                extracted_text = f"[PDF reçu: {filename} — installez PyPDF2 pour l'extraction de texte]"
            except Exception as e:
                extracted_text = f"[PDF non lisible: {e}]"

        elif extension in ['txt', 'md', 'csv']:
            try:
                extracted_text = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                extracted_text = content_bytes.decode('latin-1', errors='replace')

        elif extension in ['png', 'jpg', 'jpeg', 'webp']:
            # Pour les images, on essaie d'utiliser OCR si disponible
            try:
                import pytesseract
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(content_bytes))
                extracted_text = pytesseract.image_to_string(img, lang='fra')
                if not extracted_text.strip():
                    extracted_text = f"[Image reçue: {filename} — aucun texte détecté]"
            except ImportError:
                extracted_text = f"[Image reçue: {filename} — installez pytesseract pour extraire le texte des images]"
            except Exception as e:
                extracted_text = f"[Image: {filename} — erreur OCR: {e}]"
        else:
            try:
                extracted_text = content_bytes.decode('utf-8', errors='replace')
            except Exception:
                extracted_text = f"[Fichier binaire: {filename}]"

        if not extracted_text.strip():
            extracted_text = f"[Fichier vide ou non lisible: {filename}]"

        # Nettoyage du texte extrait
        extracted_text = extracted_text.strip()

        # Sauvegarde dans le système de mémoire fichiers
        doc_id = file_memory.save_document(
            filename=filename,
            content=extracted_text,
            category="document",
            tags=[extension, user_id, "uploaded"]
        )

        # Aussi sauvegarder en chunks dans SQLite pour compatibilité
        max_chunk = 2000
        chunks = [extracted_text[i:i + max_chunk] for i in range(0, len(extracted_text), max_chunk)]
        for i, chunk in enumerate(chunks):
            title = f"{filename} (partie {i+1}/{len(chunks)})" if len(chunks) > 1 else filename
            entry = PMGEntry(
                title=title,
                content=chunk,
                category="document",
                source=filename,
                tags=[extension, user_id]
            )
            pmg_memory.add_entry(entry)

        logger.info(f"✅ Document uploadé et indexé: {filename} — {len(extracted_text)} caractères")

        return {
            "status": "success",
            "filename": filename,
            "doc_id": doc_id,
            "characters_extracted": len(extracted_text),
            "file_saved": str(MEMORY_DIR / f"{doc_id}_{filename[:30]}.txt"),
            "message": f"'{filename}' a été lu et enregistré dans la mémoire de Jarvis ({len(extracted_text)} caractères)",
            "preview": extracted_text[:300] + "..." if len(extracted_text) > 300 else extracted_text,
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Upload document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/documents")
async def list_memory_documents():
    """Liste tous les documents dans la mémoire fichiers de Jarvis"""
    docs = file_memory.get_all_documents()
    return {
        "total": len(docs),
        "memory_dir": str(MEMORY_DIR.absolute()),
        "documents": docs,
        "timestamp": datetime.now()
    }


@app.get("/memory/documents/{doc_id}")
async def get_memory_document(doc_id: str):
    """Retourne le contenu complet d'un document"""
    content = file_memory.get_full_content(doc_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    return {"doc_id": doc_id, "content": content}


@app.delete("/memory/documents/{doc_id}")
async def delete_memory_document(doc_id: str):
    """Supprime un document de la mémoire"""
    success = file_memory.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document non trouvé")
    return {"status": "success", "message": f"Document {doc_id} supprimé"}


@app.post("/memory/search")
async def search_memory(query: str):
    """Cherche dans la mémoire fichiers de Jarvis"""
    results = file_memory.search_documents(query)
    return {
        "query": query,
        "results_count": len(results),
        "results": [
            {
                "id": r["id"],
                "filename": r["filename"],
                "score": r["score"],
                "passages": r["passages"]
            }
            for r in results
        ]
    }


@app.post("/pmg/add")
async def add_pmg_entry(entry: PMGEntry):
    try:
        entry_id = pmg_memory.add_entry(entry)
        return {
            "status": "success",
            "entry_id": entry_id,
            "message": f"Entrée PMG ajoutée: {entry.title}",
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Add PMG Entry error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pmg/batch-import")
async def batch_import_pmg(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text_content = content.decode('utf-8')

        if file.filename.endswith('.json'):
            entries = json.loads(text_content)
        else:
            entries = []
            blocks = text_content.split('\n---\n')
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    entries.append({
                        "title": lines[0],
                        "content": '\n'.join(lines[1:-1]),
                        "category": lines[-1]
                    })

        added_ids = []
        for entry_data in entries:
            entry = PMGEntry(
                title=entry_data.get("title", ""),
                content=entry_data.get("content", ""),
                category=entry_data.get("category", "general"),
                source=file.filename,
                tags=entry_data.get("tags", [])
            )
            entry_id = pmg_memory.add_entry(entry)
            added_ids.append(entry_id)

        return {
            "status": "success",
            "entries_added": len(added_ids),
            "entry_ids": added_ids,
            "message": f"{len(added_ids)} entrées importées depuis {file.filename}",
            "timestamp": datetime.now()
        }

    except Exception as e:
        logger.error(f"Batch import error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pmg/memory")
async def get_pmg_memory():
    try:
        entries = pmg_memory.get_all_entries()
        return {
            "total_entries": len(entries),
            "entries": entries,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/settings/name")
async def change_assistant_name(user_id: str, new_name: str):
    try:
        current_settings = settings_service.get_settings(user_id)
        update = SettingsUpdate(
            assistant_name=new_name,
            voice_speed=current_settings.voice_speed,
            voice_volume=current_settings.voice_volume,
            voice_id=current_settings.voice_id,
            language=current_settings.language,
            theme=current_settings.theme
        )
        updated = settings_service.update_settings(user_id, update)
        return {
            "status": "success",
            "message": f"Nom changé en '{new_name}'",
            "settings": updated.dict(),
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/settings/{user_id}")
async def get_user_settings(user_id: str):
    try:
        settings = settings_service.get_settings(user_id)
        return settings.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/settings/{user_id}")
async def update_user_settings(user_id: str, settings: SettingsUpdate):
    try:
        updated = settings_service.update_settings(user_id, settings)
        return {
            "status": "success",
            "settings": updated.dict(),
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/{user_id}")
async def get_user_history(user_id: str, limit: int = 50):
    try:
        history = history_service.get_history(user_id, limit)
        return {
            "total_messages": len(history),
            "messages": history,
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{message_id}")
async def delete_history_message(message_id: str):
    try:
        history_service.delete_message(message_id)
        return {
            "status": "success",
            "message": "Message supprimé",
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            message_text = data.get("message", "")
            is_voice = data.get("is_voice", False)

            request = ChatRequest(
                message=message_text,
                user_id=user_id,
                is_voice=is_voice
            )

            response = await chat(request)

            await websocket.send_json({
                "response": response.response,
                "sources": response.sources,
                "message_id": response.message_id,
                "is_voice_response": response.is_voice_response,
                "timestamp": response.timestamp.isoformat()
            })

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1000)


# ============================================================================
# 11. HEALTH CHECK & STATUS
# ============================================================================

@app.get("/health")
async def health_check():
    docs = file_memory.get_all_documents()
    return {
        "status": "healthy",
        "database": "connected",
        "mistral_api": "configured",
        "memory_files": len(docs),
        "memory_dir": str(MEMORY_DIR.absolute()),
        "timestamp": datetime.now()
    }


@app.get("/stats")
async def get_stats():
    all_pmg = pmg_memory.get_all_entries()
    all_docs = file_memory.get_all_documents()

    return {
        "pmg_entries_total": len(all_pmg),
        "memory_files_total": len(all_docs),
        "memory_dir": str(MEMORY_DIR.absolute()),
        "api_version": "2.0.0",
        "database": "SQLite + FileSystem",
        "timestamp": datetime.now()
    }

# ============================
# FLUTTER WEB FRONTEND
# ============================

FLUTTER_DIR = Path(r"C:\Perso\Code\pmg_jarvis_app\build\web")

@app.get("/app")
async def flutter_app():
    return FileResponse(str(FLUTTER_DIR / "index.html"))


@app.get("/app/{full_path:path}")
async def flutter_assets(full_path: str):
    file_path = FLUTTER_DIR / full_path

    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))

    return FileResponse(str(FLUTTER_DIR / "index.html"))


# ============================================================================
# 12. DÉMARRAGE
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 PMG Jarvis API v2.0 démarrage...")
    logger.info(f"📁 Base de données: {db.db_path}")
    logger.info(f"🧠 Mémoire fichiers: {MEMORY_DIR.absolute()}")
    logger.info(f"📄 Documents en mémoire: {len(file_memory.get_all_documents())}")
    logger.info(f"🤖 Mistral API: Configurée")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )