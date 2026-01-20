"""
Face Recognition Module using InsightFace/ArcFace.
Handles embedding extraction and matching.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import logging
from scipy.spatial.distance import cosine

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None
    logging.warning("InsightFace not installed. Face recognition will not work.")


@dataclass
class MatchResult:
    """Result of face matching."""
    matched: bool
    user_id: Optional[str]
    user_name: Optional[str]
    user_code: Optional[str]
    score: float  # Cosine similarity score
    all_scores: Dict[str, float]  # Scores for all users


@dataclass
class EmbeddingResult:
    """Result of embedding extraction."""
    success: bool
    embedding: Optional[np.ndarray]
    error: Optional[str] = None


class FaceRecognition:
    """
    Face Recognition module using InsightFace/ArcFace.
    
    Features:
    - Embedding extraction (512D)
    - Cosine similarity matching
    - Multi-template matching per user
    """
    
    def __init__(
        self,
        model_name: str = "buffalo_l",
        threshold: float = 0.4,
        embedding_dim: int = 512
    ):
        """
        Initialize face recognition module.
        
        Args:
            model_name: InsightFace model name
            threshold: Matching threshold (cosine similarity)
            embedding_dim: Embedding dimension
        """
        self.model_name = model_name
        self.threshold = threshold
        self.embedding_dim = embedding_dim
        
        self._model = None
        self._initialized = False
        
        # User database: {user_id: {"name": str, "code": str, "embeddings": List[np.ndarray]}}
        self._user_db: Dict[str, Dict] = {}
    
    def _initialize(self):
        """Lazy initialization."""
        if self._initialized:
            return
        
        if FaceAnalysis is None:
            raise RuntimeError("InsightFace is not installed")
        
        self._model = FaceAnalysis(
            name=self.model_name,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._model.prepare(ctx_id=0, det_size=(640, 640))
        self._initialized = True
        logging.info("Face recognition model initialized")
    
    def extract_embedding(self, frame: np.ndarray) -> EmbeddingResult:
        """
        Extract face embedding from frame.
        
        Args:
            frame: Input BGR frame with face
            
        Returns:
            EmbeddingResult with 512D embedding
        """
        self._initialize()
        
        try:
            # Detect and get embedding
            faces = self._model.get(frame)
            
            if len(faces) == 0:
                return EmbeddingResult(
                    success=False,
                    embedding=None,
                    error="No face detected"
                )
            
            if len(faces) > 1:
                return EmbeddingResult(
                    success=False,
                    embedding=None,
                    error="Multiple faces detected"
                )
            
            # Get embedding (normalized)
            embedding = faces[0].embedding
            
            if embedding is None:
                return EmbeddingResult(
                    success=False,
                    embedding=None,
                    error="Failed to extract embedding"
                )
            
            # Normalize if not already
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return EmbeddingResult(
                success=True,
                embedding=embedding.astype(np.float32)
            )
            
        except Exception as e:
            logging.error(f"Embedding extraction failed: {e}")
            return EmbeddingResult(
                success=False,
                embedding=None,
                error=str(e)
            )
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Cosine similarity = 1 - cosine distance
        return float(1 - cosine(embedding1, embedding2))
    
    def register_user(
        self,
        user_id: str,
        name: str,
        code: str,
        embeddings: List[np.ndarray]
    ):
        """
        Register a user with their face embeddings.
        
        Args:
            user_id: Unique user ID
            name: User name
            code: User code (employee/student ID)
            embeddings: List of face embeddings
        """
        self._user_db[user_id] = {
            "name": name,
            "code": code,
            "embeddings": embeddings
        }
        logging.info(f"Registered user {name} ({code}) with {len(embeddings)} embeddings")
    
    def unregister_user(self, user_id: str):
        """Remove a user from the database."""
        if user_id in self._user_db:
            del self._user_db[user_id]
            logging.info(f"Unregistered user {user_id}")
    
    def load_users_from_db(self, users_with_embeddings: List[Tuple]):
        """
        Load users from database.
        
        Args:
            users_with_embeddings: List of (User, FaceTemplate) tuples
        """
        self._user_db.clear()
        
        for user, template in users_with_embeddings:
            if user.user_id not in self._user_db:
                self._user_db[user.user_id] = {
                    "name": user.name,
                    "code": user.code,
                    "embeddings": []
                }
            
            # Convert bytes to numpy array
            embedding = np.frombuffer(template.embedding, dtype=np.float32)
            if len(embedding) == self.embedding_dim:
                self._user_db[user.user_id]["embeddings"].append(embedding)
        
        logging.info(f"Loaded {len(self._user_db)} users from database")
    
    def match(self, embedding: np.ndarray) -> MatchResult:
        """
        Match embedding against registered users.
        Uses maximum similarity among all templates per user.
        
        Args:
            embedding: Query face embedding
            
        Returns:
            MatchResult with best match
        """
        if len(self._user_db) == 0:
            return MatchResult(
                matched=False,
                user_id=None,
                user_name=None,
                user_code=None,
                score=0.0,
                all_scores={}
            )
        
        all_scores = {}
        best_user_id = None
        best_score = -1.0
        
        for user_id, user_data in self._user_db.items():
            user_embeddings = user_data["embeddings"]
            
            if len(user_embeddings) == 0:
                all_scores[user_id] = 0.0
                continue
            
            # Get maximum similarity across all templates
            max_similarity = max(
                self.compute_similarity(embedding, emb)
                for emb in user_embeddings
            )
            
            all_scores[user_id] = max_similarity
            
            if max_similarity > best_score:
                best_score = max_similarity
                best_user_id = user_id
        
        # Check if above threshold
        matched = best_score >= self.threshold
        
        if matched and best_user_id:
            user_data = self._user_db[best_user_id]
            return MatchResult(
                matched=True,
                user_id=best_user_id,
                user_name=user_data["name"],
                user_code=user_data["code"],
                score=best_score,
                all_scores=all_scores
            )
        else:
            return MatchResult(
                matched=False,
                user_id=None,
                user_name=None,
                user_code=None,
                score=best_score,
                all_scores=all_scores
            )
    
    def match_from_frame(self, frame: np.ndarray) -> MatchResult:
        """
        Extract embedding from frame and match.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            MatchResult
        """
        emb_result = self.extract_embedding(frame)
        
        if not emb_result.success:
            return MatchResult(
                matched=False,
                user_id=None,
                user_name=None,
                user_code=None,
                score=0.0,
                all_scores={}
            )
        
        return self.match(emb_result.embedding)
    
    def get_user_count(self) -> int:
        """Get number of registered users."""
        return len(self._user_db)
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """Get user information."""
        return self._user_db.get(user_id)


# Singleton instance
_recognition_instance: Optional[FaceRecognition] = None


def get_face_recognition(
    model_name: str = "buffalo_l",
    threshold: float = 0.4
) -> FaceRecognition:
    """Get or create face recognition singleton."""
    global _recognition_instance
    if _recognition_instance is None:
        _recognition_instance = FaceRecognition(
            model_name=model_name,
            threshold=threshold
        )
    return _recognition_instance
