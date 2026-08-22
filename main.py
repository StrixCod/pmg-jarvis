from pathlib import Path
from fastapi import (
    FastAPI,
    WebSocket,
    HTTPException,
    File,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum

import os
import json
import logging
import re
import sqlite3
import uuid


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("PMG_JARVIS")


# ============================================================================
# MISTRAL
# ============================================================================

# IMPORTANT :
# Configure cette variable dans les variables d'environnement de ton serveur.
#
# Exemple :
# MISTRAL_API_KEY = ta_cle
#
# Ne mets pas la clé directement dans GitHub.

MISTRAL_API_KEY = "L9KyHhbmAEYTDmEOwIUcPCBkpWf6xeS3"

MISTRAL_MODEL = os.getenv(
    "MISTRAL_MODEL",
    "mistral-small-latest"
)

MISTRAL_BASE_URL = os.getenv(
    "MISTRAL_BASE_URL",
    "https://api.mistral.ai/v1"
)


# ============================================================================
# GITHUB MEMORY
# ============================================================================

# Dépôt GitHub contenant les fichiers de mémoire.
GITHUB_OWNER = os.getenv(
    "GITHUB_OWNER",
    "StrixCod"
)

GITHUB_REPO = os.getenv(
    "GITHUB_REPO",
    "pmg-jarvis"
)

GITHUB_BRANCH = os.getenv(
    "GITHUB_BRANCH",
    "main"
)

GITHUB_MEMORY_PATH = os.getenv(
    "GITHUB_MEMORY_PATH",
    "jarvis_memory"
)


# ============================================================================
# CHEMINS LOCAUX
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent

# Si jarvis_memory existe dans le dépôt cloné, on l'utilise directement.
LOCAL_GITHUB_MEMORY_DIR = (
    BASE_DIR / GITHUB_MEMORY_PATH
)

# Cache local des fichiers récupérés depuis GitHub.
DOWNLOADED_MEMORY_DIR = (
    BASE_DIR / ".jarvis_github_memory"
)

# Documents envoyés via l'application.
LOCAL_MEMORY_DIR = (
    BASE_DIR / "jarvis_memory_local"
)

DATABASE_PATH = (
    BASE_DIR / "pmg_jarvis.db"
)

FLUTTER_DIR = (
    BASE_DIR / "build" / "web"
)


LOCAL_MEMORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOWNLOADED_MEMORY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="PMG Jarvis API",
    description="Assistant IA personnel - Pony Mounted Games",
    version="3.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],
)


# ============================================================================
# 2. MODÈLES
# ============================================================================

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(
        default_factory=datetime.now
    )
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
    created_at: datetime = Field(
        default_factory=datetime.now
    )
    updated_at: datetime = Field(
        default_factory=datetime.now
    )
    tags: List[str] = Field(
        default_factory=list
    )


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
# 3. GITHUB MEMORY
# ============================================================================

class GitHubMemoryLoader:
    """
    Charge les fichiers .txt présents dans :

    https://github.com/StrixCod/pmg-jarvis/tree/main/jarvis_memory

    Fonctionnement :

    1. Si jarvis_memory existe localement :
       -> utilisation directe.

    2. Sinon :
       -> interrogation de l'API GitHub
       -> récupération de tous les .txt
       -> stockage dans .jarvis_github_memory
       -> utilisation par Jarvis.

    3. Les sous-dossiers sont également parcourus.
    """

    def __init__(self):
        self.owner = GITHUB_OWNER
        self.repo = GITHUB_REPO
        self.branch = GITHUB_BRANCH
        self.memory_path = GITHUB_MEMORY_PATH

        self.local_dir = LOCAL_GITHUB_MEMORY_DIR
        self.cache_dir = DOWNLOADED_MEMORY_DIR

    # ------------------------------------------------------------------------
    # URL RAW
    # ------------------------------------------------------------------------

    def raw_url(
        self,
        path: str,
    ) -> str:

        return (
            f"https://raw.githubusercontent.com/"
            f"{self.owner}/"
            f"{self.repo}/"
            f"{self.branch}/"
            f"{path}"
        )

    # ------------------------------------------------------------------------
    # API GITHUB
    # ------------------------------------------------------------------------

    def api_url(
        self,
        path: str,
    ) -> str:

        return (
            f"https://api.github.com/repos/"
            f"{self.owner}/"
            f"{self.repo}/contents/"
            f"{path}"
        )

    # ------------------------------------------------------------------------
    # RECHERCHE LOCALE
    # ------------------------------------------------------------------------

    def get_local_files(self) -> List[Path]:

        if not self.local_dir.exists():
            return []

        return sorted(
            [
                p
                for p in self.local_dir.rglob("*.txt")
                if p.is_file()
            ],
            key=lambda p: str(p).lower(),
        )

    # ------------------------------------------------------------------------
    # CACHE GITHUB
    # ------------------------------------------------------------------------

    def get_cached_files(self) -> List[Path]:

        if not self.cache_dir.exists():
            return []

        return sorted(
            [
                p
                for p in self.cache_dir.rglob("*.txt")
                if p.is_file()
            ],
            key=lambda p: str(p).lower(),
        )

    # ------------------------------------------------------------------------
    # SYNCHRONISATION GITHUB
    # ------------------------------------------------------------------------

    async def sync_from_github(self):

        import httpx

        logger.info(
            "🌐 Vérification de la mémoire GitHub..."
        )

        logger.info(
            "📦 Repository : %s/%s",
            self.owner,
            self.repo,
        )

        logger.info(
            "📁 Dossier : %s",
            self.memory_path,
        )

        try:

            async with httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
            ) as client:

                await self._download_directory(
                    client,
                    self.memory_path,
                )

            files = self.get_cached_files()

            logger.info(
                "✅ Mémoire GitHub synchronisée : %s fichier(s)",
                len(files),
            )

            for file_path in files:

                logger.info(
                    "   📄 %s",
                    file_path.relative_to(
                        self.cache_dir
                    ),
                )

            return True

        except Exception as e:

            logger.error(
                "❌ Impossible de synchroniser GitHub : %s",
                e,
            )

            cached = self.get_cached_files()

            if cached:

                logger.warning(
                    "⚠️ Utilisation du cache GitHub existant : %s fichier(s)",
                    len(cached),
                )

                return True

            return False

    # ------------------------------------------------------------------------
    # TÉLÉCHARGEMENT RÉCURSIF
    # ------------------------------------------------------------------------

    async def _download_directory(
        self,
        client,
        github_path: str,
    ):

        url = self.api_url(
            github_path
        )

        response = await client.get(
            url,
            headers={
                "Accept": "application/vnd.github+json"
            },
        )

        if response.status_code != 200:

            raise Exception(
                f"GitHub API HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        items = response.json()

        if not isinstance(items, list):

            raise Exception(
                "Réponse GitHub inattendue."
            )

        for item in items:

            item_type = item.get(
                "type"
            )

            item_path = item.get(
                "path",
                ""
            )

            if item_type == "dir":

                await self._download_directory(
                    client,
                    item_path,
                )

            elif (
                item_type == "file"
                and item_path.lower().endswith(".txt")
            ):

                await self._download_file(
                    client,
                    item_path,
                )

    # ------------------------------------------------------------------------
    # TÉLÉCHARGEMENT FICHIER
    # ------------------------------------------------------------------------

    async def _download_file(
        self,
        client,
        github_path: str,
    ):

        raw = self.raw_url(
            github_path
        )

        response = await client.get(
            raw
        )

        if response.status_code != 200:

            logger.warning(
                "⚠️ Impossible de télécharger %s",
                github_path,
            )

            return

        relative_path = Path(
            github_path
        ).relative_to(
            self.memory_path
        )

        destination = (
            self.cache_dir
            / relative_path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_bytes(
            response.content
        )

    # ------------------------------------------------------------------------
    # FICHIERS DISPONIBLES
    # ------------------------------------------------------------------------

    def get_memory_files(self) -> List[Path]:

        local_files = self.get_local_files()

        if local_files:

            logger.info(
                "📚 Utilisation directe de jarvis_memory : %s fichier(s)",
                len(local_files),
            )

            return local_files

        cached_files = self.get_cached_files()

        if cached_files:

            return cached_files

        return []


github_memory = GitHubMemoryLoader()


# ============================================================================
# 4. MÉMOIRE TEXTE
# ============================================================================

class FileMemorySystem:

    def __init__(
        self,
        github_loader: GitHubMemoryLoader,
        local_memory_dir: Path,
    ):

        self.github_loader = github_loader
        self.local_memory_dir = local_memory_dir

    # ------------------------------------------------------------------------
    # FICHIERS
    # ------------------------------------------------------------------------

    def get_github_files(self):

        return self.github_loader.get_memory_files()

    def get_local_files(self):

        if not self.local_memory_dir.exists():
            return []

        return sorted(
            [
                p
                for p in self.local_memory_dir.rglob("*.txt")
                if p.is_file()
            ],
            key=lambda p: str(p).lower(),
        )

    def get_all_memory_files(self):

        github_files = self.get_github_files()

        local_files = self.get_local_files()

        return (
            github_files
            + local_files
        )

    # ------------------------------------------------------------------------
    # LECTURE
    # ------------------------------------------------------------------------

    def read_file(
        self,
        path: Path,
    ) -> str:

        try:

            return path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            try:

                return path.read_text(
                    encoding="latin-1"
                )

            except Exception as e:

                logger.error(
                    "Erreur lecture %s : %s",
                    path,
                    e,
                )

                return ""

        except Exception as e:

            logger.error(
                "Erreur lecture %s : %s",
                path,
                e,
            )

            return ""

    # ------------------------------------------------------------------------
    # NORMALISATION
    # ------------------------------------------------------------------------

    def normalize_text(
        self,
        text: str,
    ) -> str:

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    # ------------------------------------------------------------------------
    # MOTS
    # ------------------------------------------------------------------------

    def tokenize(
        self,
        text: str,
    ) -> List[str]:

        words = re.findall(
            r"[a-zA-ZÀ-ÿ0-9]+",
            text.lower(),
        )

        stop_words = {
            "les",
            "des",
            "une",
            "dans",
            "pour",
            "avec",
            "sans",
            "sur",
            "sous",
            "entre",
            "être",
            "avoir",
            "est",
            "sont",
            "qui",
            "que",
            "quoi",
            "comment",
            "quel",
            "quelle",
            "quels",
            "quelles",
            "cela",
            "cette",
            "ce",
            "ces",
            "mon",
            "ma",
            "mes",
            "ton",
            "ta",
            "tes",
            "son",
            "sa",
            "ses",
            "notre",
            "votre",
            "leur",
            "leurs",
            "une",
            "des",
            "aux",
            "du",
            "de",
            "le",
            "la",
            "un",
            "et",
            "ou",
            "en",
            "au",
            "a",
            "à",
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
        }

        return [
            word
            for word in words
            if len(word) >= 3
            and word not in stop_words
        ]

    # ------------------------------------------------------------------------
    # PASSAGES
    # ------------------------------------------------------------------------

    def extract_passages(
        self,
        content: str,
        query_words: List[str],
    ) -> List[str]:

        paragraphs = re.split(
            r"\n\s*\n",
            content,
        )

        scored = []

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if len(paragraph) < 20:
                continue

            lower = paragraph.lower()

            score = 0

            for word in query_words:

                count = lower.count(
                    word
                )

                if count:
                    score += min(
                        count,
                        5,
                    )

            if score > 0:

                scored.append(
                    (
                        score,
                        paragraph,
                    )
                )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        passages = []

        total = 0

        for score, paragraph in scored:

            if len(passages) >= 5:
                break

            if total >= 6000:
                break

            remaining = (
                6000 - total
            )

            paragraph = paragraph[
                :remaining
            ]

            passages.append(
                paragraph
            )

            total += len(
                paragraph
            )

        return passages

    # ------------------------------------------------------------------------
    # RECHERCHE
    # ------------------------------------------------------------------------

    def search_documents(
        self,
        query: str,
        max_results: int = 6,
    ) -> List[Dict]:

        query_words = self.tokenize(
            query
        )

        if not query_words:
            return []

        files = (
            self.get_all_memory_files()
        )

        logger.info(
            "🔎 Recherche '%s' dans %s fichier(s)",
            query,
            len(files),
        )

        results = []

        for path in files:

            content = self.read_file(
                path
            )

            if not content:
                continue

            content = self.normalize_text(
                content
            )

            lower = content.lower()

            score = 0

            matched = 0

            # --------------------------------------------------------------
            # CONTENU
            # --------------------------------------------------------------

            for word in query_words:

                count = lower.count(
                    word
                )

                if count:

                    matched += 1

                    score += min(
                        count,
                        10,
                    )

            # --------------------------------------------------------------
            # BONUS MOTS MULTIPLES
            # --------------------------------------------------------------

            if matched >= 2:

                score += (
                    matched * 5
                )

            # --------------------------------------------------------------
            # BONUS NOM FICHIER
            # --------------------------------------------------------------

            filename = path.stem.lower()

            for word in query_words:

                if word in filename:

                    score += 8

            if score <= 0:
                continue

            passages = (
                self.extract_passages(
                    content,
                    query_words,
                )
            )

            if not passages:

                passages = [
                    content[:6000]
                ]

            # Détermine la provenance

            is_local_upload = (
                self.local_memory_dir in path.parents
                or path.parent
                == self.local_memory_dir
            )

            results.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "score": score,
                    "passages": passages,
                    "source_type": (
                        "upload"
                        if is_local_upload
                        else "github"
                    ),
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        results = results[
            :max_results
        ]

        for result in results:

            logger.info(
                "   📄 %s | score=%s | %s",
                result["filename"],
                result["score"],
                result["source_type"],
            )

        return results

    # ------------------------------------------------------------------------
    # DOCUMENT UPLOADÉ
    # ------------------------------------------------------------------------

    def save_document(
        self,
        filename: str,
        content: str,
    ) -> str:

        doc_id = str(
            uuid.uuid4()
        )[:8]

        safe_name = re.sub(
            r"[^\w\s\-.]",
            "_",
            Path(filename).stem,
        )

        safe_name = safe_name[
            :100
        ]

        if not safe_name:
            safe_name = "document"

        path = (
            self.local_memory_dir
            / f"{doc_id}_{safe_name}.txt"
        )

        path.write_text(
            content,
            encoding="utf-8",
        )

        logger.info(
            "📥 Document ajouté : %s",
            path,
        )

        return doc_id

    # ------------------------------------------------------------------------
    # LISTE
    # ------------------------------------------------------------------------

    def get_all_documents(self):

        result = []

        for path in self.get_all_memory_files():

            try:

                stat = path.stat()

                result.append(
                    {
                        "filename": path.name,
                        "path": str(path),
                        "source_type": (
                            "upload"
                            if self.local_memory_dir in path.parents
                            else "github"
                        ),
                        "size_bytes": stat.st_size,
                        "modified_at": datetime.fromtimestamp(
                            stat.st_mtime
                        ).isoformat(),
                    }
                )

            except Exception:
                pass

        return result

    # ------------------------------------------------------------------------
    # CONTENU
    # ------------------------------------------------------------------------

    def get_full_content(
        self,
        filename: str,
    ):

        for path in self.get_all_memory_files():

            if path.name == filename:

                return self.read_file(
                    path
                )

        return None

    # ------------------------------------------------------------------------
    # SUPPRESSION UPLOAD
    # ------------------------------------------------------------------------

    def delete_document(
        self,
        filename: str,
    ) -> bool:

        for path in self.get_local_files():

            if path.name == filename:

                path.unlink()

                return True

        return False


file_memory = FileMemorySystem(
    github_memory,
    LOCAL_MEMORY_DIR,
)


# ============================================================================
# 5. DATABASE SQLITE
# ============================================================================

class Database:

    def __init__(
        self,
        path: Path,
    ):

        self.path = str(path)

        self.init_db()

    def connection(self):

        return sqlite3.connect(
            self.path
        )

    def init_db(self):

        conn = self.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pmg_memory (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                category TEXT,
                source TEXT,
                created_at TEXT,
                updated_at TEXT,
                tags TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                message_content TEXT,
                response_content TEXT,
                is_voice INTEGER,
                created_at TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                assistant_name TEXT,
                voice_speed REAL,
                voice_volume REAL,
                voice_id TEXT,
                language TEXT,
                theme TEXT,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )

        conn.commit()

        conn.close()


db = Database(
    DATABASE_PATH
)


# ============================================================================
# 6. PMG MEMORY SQLITE
# ============================================================================

class PMGMemory:

    def __init__(
        self,
        database: Database,
    ):

        self.db = database

    def add_entry(
        self,
        entry: PMGEntry,
    ) -> str:

        entry_id = str(
            uuid.uuid4()
        )

        conn = self.db.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO pmg_memory
            (
                id,
                title,
                content,
                category,
                source,
                created_at,
                updated_at,
                tags
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                entry.title,
                entry.content,
                entry.category,
                entry.source or "",
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                json.dumps(
                    entry.tags,
                    ensure_ascii=False,
                ),
            ),
        )

        conn.commit()

        conn.close()

        return entry_id

    def search(
        self,
        query: str,
    ):

        conn = self.db.connection()

        cursor = conn.cursor()

        pattern = (
            f"%{query}%"
        )

        cursor.execute(
            """
            SELECT
                id,
                title,
                content,
                category,
                source,
                created_at,
                tags
            FROM pmg_memory
            WHERE
                title LIKE ?
                OR content LIKE ?
                OR tags LIKE ?
            ORDER BY updated_at DESC
            LIMIT 10
            """,
            (
                pattern,
                pattern,
                pattern,
            ),
        )

        rows = cursor.fetchall()

        conn.close()

        result = []

        for row in rows:

            try:
                tags = json.loads(
                    row[6]
                )
            except Exception:
                tags = []

            result.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "category": row[3],
                    "source": row[4],
                    "created_at": row[5],
                    "tags": tags,
                }
            )

        return result

    def get_all_entries(self):

        conn = self.db.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                title,
                content,
                category,
                source,
                created_at,
                tags
            FROM pmg_memory
            ORDER BY created_at DESC
            """
        )

        rows = cursor.fetchall()

        conn.close()

        result = []

        for row in rows:

            try:
                tags = json.loads(
                    row[6]
                )
            except Exception:
                tags = []

            result.append(
                {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "category": row[3],
                    "source": row[4],
                    "created_at": row[5],
                    "tags": tags,
                }
            )

        return result


pmg_memory = PMGMemory(
    db
)


# ============================================================================
# 7. HISTORIQUE
# ============================================================================

class HistoryService:

    def __init__(
        self,
        database: Database,
    ):

        self.db = database

    def save_message(
        self,
        user_id: str,
        message: str,
        response: str,
        is_voice: bool,
    ):

        conn = self.db.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO history
            (
                id,
                user_id,
                message_content,
                response_content,
                is_voice,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                user_id,
                message,
                response,
                1 if is_voice else 0,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()

        conn.close()

    def get_history(
        self,
        user_id: str,
        limit: int = 50,
    ):

        conn = self.db.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                message_content,
                response_content,
                is_voice,
                created_at
            FROM history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (
                user_id,
                limit,
            ),
        )

        rows = cursor.fetchall()

        conn.close()

        rows.reverse()

        return [
            {
                "id": row[0],
                "message": row[1],
                "response": row[2],
                "is_voice": bool(row[3]),
                "timestamp": row[4],
            }
            for row in rows
        ]

    def delete_message(
        self,
        message_id: str,
    ):

        conn = self.db.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM history
            WHERE id = ?
            """,
            (message_id,),
        )

        conn.commit()

        conn.close()


history_service = HistoryService(
    db
)


# ============================================================================
# 8. SETTINGS
# ============================================================================

class SettingsService:

    def __init__(
        self,
        database: Database,
    ):

        self.db = database

    def create_default(
        self,
        user_id: str,
    ):

        now = datetime.now().isoformat()

        conn = self.db.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR IGNORE INTO user_settings
            (
                user_id,
                assistant_name,
                voice_speed,
                voice_volume,
                voice_id,
                language,
                theme,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                "Jarvis",
                1.0,
                1.0,
                "default",
                "fr",
                "dark",
                now,
                now,
            ),
        )

        conn.commit()

        conn.close()

    def get(
        self,
        user_id: str,
    ):

        self.create_default(
            user_id
        )

        conn = self.db.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                user_id,
                assistant_name,
                voice_speed,
                voice_volume,
                voice_id,
                language,
                theme,
                created_at,
                updated_at
            FROM user_settings
            WHERE user_id = ?
            """,
            (user_id,),
        )

        row = cursor.fetchone()

        conn.close()

        return UserSettings(
            user_id=row[0],
            assistant_name=row[1],
            voice_speed=row[2],
            voice_volume=row[3],
            voice_id=row[4],
            language=row[5],
            theme=row[6],
            created_at=datetime.fromisoformat(
                row[7]
            ),
            updated_at=datetime.fromisoformat(
                row[8]
            ),
        )

    def update(
        self,
        user_id: str,
        settings: SettingsUpdate,
    ):

        self.create_default(
            user_id
        )

        now = datetime.now().isoformat()

        conn = self.db.connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE user_settings
            SET
                assistant_name = ?,
                voice_speed = ?,
                voice_volume = ?,
                voice_id = ?,
                language = ?,
                theme = ?,
                updated_at = ?
            WHERE user_id = ?
            """,
            (
                settings.assistant_name,
                settings.voice_speed,
                settings.voice_volume,
                settings.voice_id,
                settings.language,
                settings.theme,
                now,
                user_id,
            ),
        )

        conn.commit()

        conn.close()

        return self.get(
            user_id
        )


settings_service = SettingsService(
    db
)


# ============================================================================
# 9. MISTRAL
# ============================================================================

class MistralService:

    def __init__(
        self,
        api_key: str,
    ):

        self.api_key = api_key
        self.model = MISTRAL_MODEL
        self.base_url = MISTRAL_BASE_URL

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:

        if not self.api_key:

            raise HTTPException(
                status_code=500,
                detail=(
                    "MISTRAL_API_KEY n'est pas configurée."
                ),
            )

        import httpx

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:

            async with httpx.AsyncClient(
                timeout=60.0
            ) as client:

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if response.status_code != 200:

                logger.error(
                    "Mistral error %s: %s",
                    response.status_code,
                    response.text,
                )

                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Erreur API Mistral : "
                        + response.text
                    ),
                )

            data = response.json()

            return (
                data["choices"][0]["message"]["content"]
                .strip()
            )

        except HTTPException:
            raise

        except Exception as e:

            logger.exception(
                "Erreur Mistral"
            )

            raise HTTPException(
                status_code=502,
                detail=str(e),
            )


mistral_service = MistralService(
    MISTRAL_API_KEY
)


# ============================================================================
# 10. PROMPTS
# ============================================================================

SYSTEM_PROMPT = """
Tu es PMG Jarvis, l'assistant IA personnel du projet Pony Mounted Games.

IMPORTANT :

Les informations contenues dans la section MÉMOIRE JARVIS viennent des
fichiers TXT de référence du projet.

Tu dois utiliser ces informations en priorité.

NE FABRIQUE PAS de règles, chiffres, dates ou informations qui ne sont
pas présentes dans la mémoire.

Si la réponse est présente dans la mémoire, réponds directement.

Si plusieurs documents donnent des informations différentes, signale-le.

Si aucune information pertinente n'est présente dans la mémoire, dis-le
clairement au lieu d'inventer.

Réponds en français sauf demande contraire.

Sois précis, clair et utile.

Lorsque tu utilises une information provenant de la mémoire, mentionne
le nom du fichier source lorsque cela est pertinent.

Tu es spécialisé en Pony Mounted Games mais peux aussi répondre aux
questions générales.
"""


VOICE_SYSTEM_PROMPT = """
Tu es PMG Jarvis.

Réponds comme un assistant vocal naturel.

Utilise en priorité la mémoire fournie.

Ne fabrique aucune information.

Réponds en français.

Fais des phrases courtes et naturelles.

Maximum 4 phrases.
"""


# ============================================================================
# 11. CONTEXTE MÉMOIRE
# ============================================================================

def build_memory_context(
    query: str,
):

    results = file_memory.search_documents(
        query,
        max_results=6,
    )

    if not results:

        return "", []

    context_parts = []

    sources = []

    for result in results:

        filename = result[
            "filename"
        ]

        sources.append(
            filename
        )

        passages = result[
            "passages"
        ]

        if not passages:
            continue

        text = "\n\n---\n\n".join(
            passages
        )

        context_parts.append(
            f"""
SOURCE : {filename}
TYPE : {result["source_type"]}
SCORE : {result["score"]}

{text}
"""
        )

    return (
        "\n\n".join(
            context_parts
        ),
        sources,
    )


# ============================================================================
# 12. STARTUP
# ============================================================================

@app.on_event("startup")
async def startup_event():

    logger.info(
        "=========================================="
    )

    logger.info(
        "🚀 PMG JARVIS démarrage"
    )

    logger.info(
        "=========================================="
    )

    logger.info(
        "📁 BASE_DIR : %s",
        BASE_DIR,
    )

    logger.info(
        "📚 GitHub : %s/%s",
        GITHUB_OWNER,
        GITHUB_REPO,
    )

    logger.info(
        "📁 Mémoire : %s",
        GITHUB_MEMORY_PATH,
    )

    # ------------------------------------------------------------------------
    # Vérification mémoire locale
    # ------------------------------------------------------------------------

    local_files = (
        github_memory.get_local_files()
    )

    if local_files:

        logger.info(
            "✅ jarvis_memory trouvé localement : %s fichier(s)",
            len(local_files),
        )

    else:

        logger.info(
            "ℹ️ jarvis_memory absent localement."
        )

        # Télécharge depuis GitHub

        await github_memory.sync_from_github()

    # ------------------------------------------------------------------------
    # Affichage final
    # ------------------------------------------------------------------------

    memory_files = (
        file_memory.get_github_files()
    )

    logger.info(
        "🧠 Fichiers mémoire disponibles : %s",
        len(memory_files),
    )

    for path in memory_files:

        try:

            content = file_memory.read_file(
                path
            )

            logger.info(
                "   📄 %s (%s caractères)",
                path.name,
                len(content),
            )

        except Exception:
            pass

    if MISTRAL_API_KEY:

        logger.info(
            "🤖 Mistral API : configurée"
        )

    else:

        logger.warning(
            "⚠️ MISTRAL_API_KEY absente"
        )

    logger.info(
        "=========================================="
    )


# ============================================================================
# 13. ROOT
# ============================================================================

@app.get("/")
async def root():

    documents = (
        file_memory.get_all_documents()
    )

    github_documents = [
        d
        for d in documents
        if d["source_type"] == "github"
    ]

    uploads = [
        d
        for d in documents
        if d["source_type"] == "upload"
    ]

    return {
        "name": "PMG Jarvis API",
        "version": "3.0.0",
        "status": "online",

        "github_repository": (
            f"{GITHUB_OWNER}/{GITHUB_REPO}"
        ),

        "github_memory_path": (
            GITHUB_MEMORY_PATH
        ),

        "github_memory_files": len(
            github_documents
        ),

        "uploaded_memory_files": len(
            uploads
        ),

        "total_memory_files": len(
            documents
        ),

        "mistral_configured": bool(
            MISTRAL_API_KEY
        ),

        "timestamp": datetime.now(),
    }


# ============================================================================
# 14. CHAT
# ============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
):

    message_id = str(
        uuid.uuid4()
    )

    try:

        # ================================================================
        # 1. RECHERCHE DANS LES TXT GITHUB
        # ================================================================

        memory_context, memory_sources = (
            build_memory_context(
                request.message
            )
        )

        # ================================================================
        # 2. RECHERCHE SQLITE
        # ================================================================

        pmg_results = (
            pmg_memory.search(
                request.message
            )
        )

        pmg_context = ""

        pmg_sources = []

        for result in pmg_results[:3]:

            pmg_context += (
                f"\n"
                f"[{result['category']}]\n"
                f"{result['title']}\n"
                f"{result['content']}\n"
            )

            if result["source"]:

                pmg_sources.append(
                    result["source"]
                )

        # ================================================================
        # 3. PROMPT
        # ================================================================

        if request.is_voice:

            system_prompt = (
                VOICE_SYSTEM_PROMPT
            )

        else:

            system_prompt = (
                SYSTEM_PROMPT
            )

        # ================================================================
        # 4. MÉMOIRE
        # ================================================================

        if memory_context:

            system_prompt += f"""

============================================================
MÉMOIRE JARVIS — FICHIERS DE RÉFÉRENCE
============================================================

{memory_context}

============================================================
FIN DE LA MÉMOIRE
============================================================

Utilise cette mémoire pour répondre à la question.
"""

        else:

            system_prompt += """

============================================================
MÉMOIRE JARVIS
============================================================

Aucune information pertinente n'a été trouvée dans les
fichiers TXT de mémoire pour cette question.

Ne prétends pas avoir trouvé une information dans la mémoire.
============================================================
"""

        # ================================================================
        # 5. SQLITE
        # ================================================================

        if pmg_context:

            system_prompt += f"""

============================================================
BASE PMG SQLITE
============================================================

{pmg_context}

============================================================
"""

        # ================================================================
        # 6. MESSAGES
        # ================================================================

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": request.message,
            },
        ]

        # ================================================================
        # 7. MISTRAL
        # ================================================================

        response_text = (
            await mistral_service.chat(
                messages
            )
        )

        # ================================================================
        # 8. HISTORIQUE
        # ================================================================

        history_service.save_message(
            user_id=request.user_id,
            message=request.message,
            response=response_text,
            is_voice=request.is_voice,
        )

        # ================================================================
        # 9. SOURCES
        # ================================================================

        sources = list(
            dict.fromkeys(
                memory_sources
                + pmg_sources
            )
        )

        return ChatResponse(
            response=response_text,
            sources=(
                sources
                if request.include_sources
                else None
            ),
            timestamp=datetime.now(),
            message_id=message_id,
            is_voice_response=request.is_voice,
        )

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "Erreur chat"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================================
# 15. LISTE MÉMOIRE
# ============================================================================

@app.get(
    "/memory/documents"
)
async def list_memory_documents():

    documents = (
        file_memory.get_all_documents()
    )

    return {
        "total": len(documents),
        "github_repository": (
            f"{GITHUB_OWNER}/{GITHUB_REPO}"
        ),
        "github_path": (
            GITHUB_MEMORY_PATH
        ),
        "documents": documents,
        "timestamp": datetime.now(),
    }


# ============================================================================
# 16. LECTURE DOCUMENT
# ============================================================================

@app.get(
    "/memory/documents/{filename:path}"
)
async def get_memory_document(
    filename: str,
):

    content = (
        file_memory.get_full_content(
            filename
        )
    )

    if content is None:

        raise HTTPException(
            status_code=404,
            detail="Document non trouvé",
        )

    return {
        "filename": filename,
        "content": content,
    }


# ============================================================================
# 17. RECHERCHE MÉMOIRE
# ============================================================================

@app.post(
    "/memory/search"
)
async def search_memory(
    query: str,
):

    results = (
        file_memory.search_documents(
            query,
            max_results=10,
        )
    )

    return {
        "query": query,
        "results_count": len(results),
        "results": results,
    }


# ============================================================================
# 18. SYNCHRONISATION GITHUB MANUELLE
# ============================================================================

@app.post(
    "/memory/sync-github"
)
async def sync_github_memory():

    success = (
        await github_memory.sync_from_github()
    )

    if not success:

        raise HTTPException(
            status_code=500,
            detail=(
                "Impossible de synchroniser "
                "la mémoire GitHub."
            ),
        )

    documents = (
        file_memory.get_github_files()
    )

    return {
        "status": "success",
        "github_repository": (
            f"{GITHUB_OWNER}/{GITHUB_REPO}"
        ),
        "path": GITHUB_MEMORY_PATH,
        "files": len(documents),
        "documents": [
            p.name
            for p in documents
        ],
        "timestamp": datetime.now(),
    }


# ============================================================================
# 19. UPLOAD DOCUMENT
# ============================================================================

@app.post(
    "/pmg/upload-document"
)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = "user_default",
):

    try:

        filename = (
            file.filename
            or "document"
        )

        content_bytes = (
            await file.read()
        )

        extension = (
            filename
            .split(".")[-1]
            .lower()
        )

        extracted_text = ""

        # ================================================================
        # TXT / MD / CSV
        # ================================================================

        if extension in {
            "txt",
            "md",
            "csv",
        }:

            try:

                extracted_text = (
                    content_bytes.decode(
                        "utf-8"
                    )
                )

            except UnicodeDecodeError:

                extracted_text = (
                    content_bytes.decode(
                        "latin-1",
                        errors="replace",
                    )
                )

        # ================================================================
        # PDF
        # ================================================================

        elif extension == "pdf":

            try:

                import io
                import PyPDF2

                reader = (
                    PyPDF2.PdfReader(
                        io.BytesIO(
                            content_bytes
                        )
                    )
                )

                pages = []

                for index, page in enumerate(
                    reader.pages
                ):

                    text = (
                        page.extract_text()
                        or ""
                    )

                    if text.strip():

                        pages.append(
                            f"[Page {index + 1}]\n{text}"
                        )

                extracted_text = (
                    "\n\n".join(pages)
                )

            except ImportError:

                extracted_text = (
                    "PDF reçu mais PyPDF2 "
                    "n'est pas installé."
                )

        # ================================================================
        # AUTRES FICHIERS
        # ================================================================

        else:

            try:

                extracted_text = (
                    content_bytes.decode(
                        "utf-8",
                        errors="replace",
                    )
                )

            except Exception:

                extracted_text = (
                    f"[Fichier non lisible : {filename}]"
                )

        if not extracted_text.strip():

            extracted_text = (
                f"[Aucun texte détecté dans {filename}]"
            )

        # ================================================================
        # SAUVEGARDE FICHIER
        # ================================================================

        doc_id = (
            file_memory.save_document(
                filename,
                extracted_text,
            )
        )

        # ================================================================
        # SQLITE
        # ================================================================

        entry = PMGEntry(
            title=filename,
            content=extracted_text,
            category="document",
            source=filename,
            tags=[
                extension,
                user_id,
                "uploaded",
            ],
        )

        entry_id = (
            pmg_memory.add_entry(
                entry
            )
        )

        return {
            "status": "success",
            "filename": filename,
            "doc_id": doc_id,
            "entry_id": entry_id,
            "characters_extracted": len(
                extracted_text
            ),
            "message": (
                f"{filename} ajouté à la mémoire."
            ),
            "preview": (
                extracted_text[:500]
            ),
            "timestamp": datetime.now(),
        }

    except Exception as e:

        logger.exception(
            "Erreur upload"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================================
# 20. AJOUT PMG SQLITE
# ============================================================================

@app.post(
    "/pmg/add"
)
async def add_pmg_entry(
    entry: PMGEntry,
):

    try:

        entry_id = (
            pmg_memory.add_entry(
                entry
            )
        )

        return {
            "status": "success",
            "entry_id": entry_id,
            "message": (
                f"Entrée PMG ajoutée : {entry.title}"
            ),
            "timestamp": datetime.now(),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================================
# 21. MÉMOIRE SQLITE
# ============================================================================

@app.get(
    "/pmg/memory"
)
async def get_pmg_memory():

    entries = (
        pmg_memory.get_all_entries()
    )

    return {
        "total_entries": len(
            entries
        ),
        "entries": entries,
        "timestamp": datetime.now(),
    }


# ============================================================================
# 22. SETTINGS
# ============================================================================

@app.get(
    "/settings/{user_id}"
)
async def get_settings(
    user_id: str,
):

    settings = (
        settings_service.get(
            user_id
        )
    )

    return settings.model_dump()


@app.put(
    "/settings/{user_id}"
)
async def update_settings(
    user_id: str,
    settings: SettingsUpdate,
):

    updated = (
        settings_service.update(
            user_id,
            settings,
        )
    )

    return {
        "status": "success",
        "settings": updated.model_dump(),
        "timestamp": datetime.now(),
    }


@app.put(
    "/settings/name"
)
async def change_assistant_name(
    user_id: str,
    new_name: str,
):

    current = (
        settings_service.get(
            user_id
        )
    )

    settings = SettingsUpdate(
        assistant_name=new_name,
        voice_speed=current.voice_speed,
        voice_volume=current.voice_volume,
        voice_id=current.voice_id,
        language=current.language,
        theme=current.theme,
    )

    updated = (
        settings_service.update(
            user_id,
            settings,
        )
    )

    return {
        "status": "success",
        "settings": updated.model_dump(),
        "timestamp": datetime.now(),
    }


# ============================================================================
# 23. HISTORY
# ============================================================================

@app.get(
    "/history/{user_id}"
)
async def get_history(
    user_id: str,
    limit: int = 50,
):

    limit = max(
        1,
        min(limit, 200)
    )

    history = (
        history_service.get_history(
            user_id,
            limit,
        )
    )

    return {
        "total_messages": len(
            history
        ),
        "messages": history,
        "timestamp": datetime.now(),
    }


@app.delete(
    "/history/{message_id}"
)
async def delete_history(
    message_id: str,
):

    history_service.delete_message(
        message_id
    )

    return {
        "status": "success",
        "message": "Message supprimé",
        "timestamp": datetime.now(),
    }


# ============================================================================
# 24. WEBSOCKET
# ============================================================================

@app.websocket(
    "/ws/chat/{user_id}"
)
async def websocket_chat(
    websocket: WebSocket,
    user_id: str,
):

    await websocket.accept()

    logger.info(
        "🔌 WebSocket connecté : %s",
        user_id,
    )

    try:

        while True:

            data = (
                await websocket.receive_json()
            )

            message = data.get(
                "message",
                "",
            )

            is_voice = data.get(
                "is_voice",
                False,
            )

            if not message.strip():

                await websocket.send_json(
                    {
                        "error": (
                            "Message vide"
                        )
                    }
                )

                continue

            request = ChatRequest(
                message=message,
                user_id=user_id,
                is_voice=is_voice,
            )

            response = await chat(
                request
            )

            await websocket.send_json(
                {
                    "response": response.response,
                    "sources": response.sources,
                    "message_id": response.message_id,
                    "is_voice_response": (
                        response.is_voice_response
                    ),
                    "timestamp": (
                        response.timestamp.isoformat()
                    ),
                }
            )

    except Exception as e:

        logger.error(
            "WebSocket fermé : %s",
            e,
        )

        try:
            await websocket.close(
                code=1000
            )
        except Exception:
            pass


# ============================================================================
# 25. HEALTH
# ============================================================================

@app.get(
    "/health"
)
async def health():

    memory_files = (
        file_memory.get_github_files()
    )

    return {
        "status": "healthy",
        "database": "connected",
        "mistral_api": (
            "configured"
            if MISTRAL_API_KEY
            else "not_configured"
        ),
        "github_repository": (
            f"{GITHUB_OWNER}/{GITHUB_REPO}"
        ),
        "github_memory_path": (
            GITHUB_MEMORY_PATH
        ),
        "memory_files": len(
            memory_files
        ),
        "timestamp": datetime.now(),
    }


# ============================================================================
# 26. STATS
# ============================================================================

@app.get(
    "/stats"
)
async def stats():

    github_files = (
        file_memory.get_github_files()
    )

    local_files = (
        file_memory.get_local_files()
    )

    sqlite_entries = (
        pmg_memory.get_all_entries()
    )

    return {
        "api_version": "3.0.0",

        "github_repository": (
            f"{GITHUB_OWNER}/{GITHUB_REPO}"
        ),

        "github_memory_path": (
            GITHUB_MEMORY_PATH
        ),

        "github_files": len(
            github_files
        ),

        "local_uploaded_files": len(
            local_files
        ),

        "sqlite_entries": len(
            sqlite_entries
        ),

        "total_memory_files": (
            len(github_files)
            + len(local_files)
        ),

        "database": "SQLite",

        "timestamp": datetime.now(),
    }


# ============================================================================
# 27. FLUTTER WEB
# ============================================================================

@app.get(
    "/app"
)
async def flutter_app():

    index = (
        FLUTTER_DIR
        / "index.html"
    )

    if not index.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Frontend Flutter non trouvé."
            ),
        )

    return FileResponse(
        str(index)
    )


@app.get(
    "/app/{full_path:path}"
)
async def flutter_assets(
    full_path: str,
):

    file_path = (
        FLUTTER_DIR
        / full_path
    )

    if (
        file_path.exists()
        and file_path.is_file()
    ):

        return FileResponse(
            str(file_path)
        )

    index = (
        FLUTTER_DIR
        / "index.html"
    )

    if index.exists():

        return FileResponse(
            str(index)
        )

    raise HTTPException(
        status_code=404,
        detail="Frontend non trouvé.",
    )


# ============================================================================
# 28. DÉMARRAGE LOCAL
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    logger.info(
        "🚀 Démarrage PMG Jarvis..."
    )

    logger.info(
        "📁 Base : %s",
        BASE_DIR,
    )

    logger.info(
        "📚 GitHub : %s/%s/%s",
        GITHUB_OWNER,
        GITHUB_REPO,
        GITHUB_MEMORY_PATH,
    )

    logger.info(
        "🗃️ SQLite : %s",
        DATABASE_PATH,
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        log_level="info",
    )
