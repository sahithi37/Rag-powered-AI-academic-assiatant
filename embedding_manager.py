import json
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime

class EmbeddingManager:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """Initialize the embedding manager with configurable model"""
        self.model_name = model_name
        self.embedding_model = None
        self.embeddings_file = "chunk_embeddings.npy"
        self.metadata_file = "embeddings_metadata.json"
        self.chunks_hash_file = "chunks_hash.txt"
        
    def initialize_model(self) -> bool:
        """Initialize the embedding model dynamically"""
        try:
            print(f"🔄 Loading embedding model ({self.model_name})...")
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"✅ Embedding model loaded successfully")
            print(f"   Model dimensions: {self.embedding_model.get_sentence_embedding_dimension()}")
            return True
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            return False
    
    def calculate_chunks_hash(self, chunks: List[Dict]) -> str:
        """Calculate a hash of chunks to detect changes"""
        try:
            # Create a stable representation of chunks for hashing
            chunks_data = []
            for chunk in chunks:
                chunk_info = {
                    'course': chunk.get('course', ''),
                    'instructor': chunk.get('instructor', ''),
                    'manual_name': chunk.get('manual_name', ''),
                    'section_title': chunk.get('section_title', ''),
                    'chunk_text': chunk.get('chunk_text', ''),
                    'slide_number': chunk.get('slide_number', '')
                }
                chunks_data.append(chunk_info)
            
            # Sort for consistent hashing
            chunks_data.sort(key=lambda x: (x['course'], x['instructor'], x['manual_name'], x['slide_number']))
            
            # Create hash
            chunks_str = json.dumps(chunks_data, sort_keys=True)
            return hashlib.md5(chunks_str.encode()).hexdigest()
            
        except Exception as e:
            print(f"⚠️ Error calculating chunks hash: {e}")
            return ""
    
    def should_recreate_embeddings(self, chunks: List[Dict]) -> bool:
        """Check if embeddings need to be recreated"""
        try:
            # Check if all required files exist
            if not all(os.path.exists(f) for f in [self.embeddings_file, self.metadata_file, self.chunks_hash_file]):
                print("🔄 Missing embedding files, will create new embeddings")
                return True
            
            # Check if chunks have changed
            current_hash = self.calculate_chunks_hash(chunks)
            
            with open(self.chunks_hash_file, 'r') as f:
                stored_hash = f.read().strip()
            
            if current_hash != stored_hash:
                print("🔄 Chunks have changed, will recreate embeddings")
                return True
            
            # Check if model has changed
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            if metadata.get('model_name') != self.model_name:
                print(f"🔄 Model changed from {metadata.get('model_name')} to {self.model_name}, will recreate embeddings")
                return True
            
            print("✅ Using existing embeddings (no changes detected)")
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking embeddings status: {e}")
            return True
    
    def create_embeddings(self, chunks: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Create embeddings for chunks efficiently"""
        try:
            if not self.embedding_model:
                raise ValueError("Embedding model not initialized")
            
            print(f"🔄 Creating embeddings for {len(chunks)} chunks...")
            
            # Extract text for embedding
            texts = [chunk.get('chunk_text', '') for chunk in chunks]
            
            # Create embeddings in batches for efficiency
            batch_size = 32  # Adjust based on memory
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts)
                embeddings.extend(batch_embeddings)
                
                # Progress indicator
                if (i + batch_size) % 100 == 0 or i + batch_size >= len(texts):
                    print(f"   Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
            
            embeddings_array = np.array(embeddings)
            
            # Create metadata
            metadata = {
                'model_name': self.model_name,
                'chunks_count': len(chunks),
                'embedding_dimensions': embeddings_array.shape[1],
                'created_at': datetime.now().isoformat(),
                'chunks_hash': self.calculate_chunks_hash(chunks)
            }
            
            print(f"✅ Successfully created {len(embeddings)} embeddings")
            return embeddings_array, metadata
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: Dict) -> bool:
        """Save embeddings and metadata to disk"""
        try:
            print("💾 Saving embeddings to disk...")
            
            # Save embeddings
            np.save(self.embeddings_file, embeddings)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save chunks hash
            with open(self.chunks_hash_file, 'w') as f:
                f.write(metadata['chunks_hash'])
            
            print("✅ Embeddings saved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self) -> Tuple[np.ndarray, Dict]:
        """Load existing embeddings from disk"""
        try:
            print("📂 Loading existing embeddings...")
            
            # Load embeddings
            embeddings = np.load(self.embeddings_file)
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Loaded {len(embeddings)} embeddings")
            print(f"   Model: {metadata.get('model_name')}")
            print(f"   Created: {metadata.get('created_at')}")
            
            return embeddings, metadata
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            raise
    
    def get_or_create_embeddings(self, chunks: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Main method: get existing embeddings or create new ones"""
        try:
            # Initialize model if needed
            if not self.embedding_model:
                if not self.initialize_model():
                    raise RuntimeError("Failed to initialize embedding model")
            
            # Check if we need to recreate embeddings
            if self.should_recreate_embeddings(chunks):
                # Create new embeddings
                embeddings, metadata = self.create_embeddings(chunks)
                
                # Save them
                if not self.save_embeddings(embeddings, metadata):
                    raise RuntimeError("Failed to save embeddings")
                
                return embeddings, metadata
            else:
                # Load existing embeddings
                return self.load_embeddings()
                
        except Exception as e:
            print(f"❌ Error in get_or_create_embeddings: {e}")
            raise
    
    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions from the model"""
        if self.embedding_model:
            return self.embedding_model.get_sentence_embedding_dimension()
        return 0
    

