import json
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGChatbot:
    def __init__(self):
        """Initialize the RAG chatbot with basic components"""
        self.chunks = []
        self.embeddings = None
        self.qdrant_client = None
        self.collection_name = "lecture_notes"
        self.embedding_model = None
        
    def setup_qdrant(self, embedding_dimensions: int = 768) -> bool:
        """Initialize Qdrant client and create collection"""
        try:
            # Initialize Qdrant client (in-memory for now)
            self.qdrant_client = QdrantClient(":memory:")
            
            # Create collection with proper vector configuration
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dimensions,
                    distance=Distance.COSINE
                )
            )
            
            print("✅ Qdrant initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up Qdrant: {e}")
            return False
    
    def initialize_with_embeddings(self, chunks: List[Dict], embeddings: np.ndarray, metadata: Dict) -> bool:
        """Initialize the RAG system with pre-loaded embeddings"""
        try:
            print("🔄 Initializing RAG system with embeddings...")
            
            # Store chunks and embeddings
            self.chunks = chunks
            self.embeddings = embeddings
            
            # Setup Qdrant with correct dimensions
            embedding_dimensions = metadata.get('embedding_dimensions', 768)
            if not self.setup_qdrant(embedding_dimensions):
                return False
            
            # Store embeddings in Qdrant
            if not self._store_embeddings_in_qdrant():
                return False
            
            print("✅ RAG system initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RAG system: {e}")
            return False
    
    def _store_embeddings_in_qdrant(self) -> bool:
        """Store pre-loaded embeddings in Qdrant"""
        try:
            print("🔄 Storing embeddings in Qdrant...")
            
            # Prepare data for Qdrant
            points = []
            for i, chunk in enumerate(self.chunks):
                if i < len(self.embeddings):
                    embedding = self.embeddings[i].tolist()
                    
                    # Create point with metadata
                    point = PointStruct(
                        id=i,
                        vector=embedding,
                        payload={
                            'course': chunk['course'],
                            'instructor': chunk['instructor'],
                            'document_name': chunk['manual_name'],
                            'section_title': chunk['section_title'],
                            'slide_number': chunk['slide_number'],
                            'chunk_text': chunk['chunk_text']
                        }
                    )
                    points.append(point)
            
            # Store all points in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            print(f"✅ Successfully stored {len(points)} embeddings in Qdrant")
            return True
            
        except Exception as e:
            print(f"❌ Error storing embeddings in Qdrant: {e}")
            return False
    
    def _get_embedding_model(self) -> SentenceTransformer:
        """Get or create embedding model (lazy loading)"""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        return self.embedding_model
    
    def find_similar_course_and_professor(self, query_course: Optional[str], query_professor: Optional[str], threshold: float = 0.6) -> Tuple[Optional[str], Optional[str]]:
        """Find the most similar course ID and professor name using semantic similarity simultaneously"""
        try:
            similar_course = None
            similar_professor = None
            
            # Get unique course IDs and instructor names from chunks
            available_courses = list(set(chunk['course'] for chunk in self.chunks))
            available_instructors = list(set(chunk['instructor'] for chunk in self.chunks))
            
            if not available_courses and not available_instructors:
                return None, None
            
            # Create embeddings for queries and available options
            model = self._get_embedding_model()
            
            # Process course similarity if query provided
            if query_course and available_courses:
                query_course_embedding = model.encode(query_course.lower())
                course_embeddings = model.encode([course.lower() for course in available_courses])
                
                # Calculate course similarities
                course_similarities = np.dot(course_embeddings, query_course_embedding) / (
                    np.linalg.norm(course_embeddings, axis=1) * np.linalg.norm(query_course_embedding)
                )
                
                # Find best course match
                best_course_idx = np.argmax(course_similarities)
                best_course_similarity = course_similarities[best_course_idx]
                
                if best_course_similarity >= threshold:
                    similar_course = available_courses[best_course_idx]
                    print(f"   🎯 Course '{query_course}' matched to '{similar_course}' (similarity: {best_course_similarity:.3f})")
            
            # Process professor similarity if query provided
            if query_professor and available_instructors:
                query_professor_embedding = model.encode(query_professor.lower())
                instructor_embeddings = model.encode([instr.lower() for instr in available_instructors])
                
                # Calculate professor similarities
                professor_similarities = np.dot(instructor_embeddings, query_professor_embedding) / (
                    np.linalg.norm(instructor_embeddings, axis=1) * np.linalg.norm(query_professor_embedding)
                )
                
                # Find best professor match
                best_professor_idx = np.argmax(professor_similarities)
                best_professor_similarity = professor_similarities[best_professor_idx]
                
                if best_professor_similarity >= threshold:
                    similar_professor = available_instructors[best_professor_idx]
                    print(f"   👨‍🏫 Professor '{query_professor}' matched to '{similar_professor}' (similarity: {best_professor_similarity:.3f})")
            
            return similar_course, similar_professor
            
        except Exception as e:
            print(f"⚠️ Error finding similar course and professor: {e}")
            return None, None
    
    def filter_by_course_and_professor(self, course: Optional[str], professor: Optional[str]) -> List[Dict]:
        """Efficiently filter documents by course and professor using similarity search for both"""
        try:
            print(f"🔄 Filtering documents by course: '{course}' and professor: '{professor}'")
            
            # Step 1: Find similar course and professor names
            similar_course, similar_professor = self.find_similar_course_and_professor(course, professor)
            
            # Step 2: Use similar names for filtering (or original if no similarity found)
            filter_course = similar_course if similar_course else course
            filter_professor = similar_professor if similar_professor else professor
            
            filters = []
            
            # Add course filter if we have a course (similar or original)
            if filter_course:
                filters.append(FieldCondition(key="course", match=MatchValue(value=filter_course)))
                print(f"   📚 Filtering by course: {filter_course}")
            
            # Add professor filter if we have a professor (similar or original)
            if filter_professor:
                filters.append(FieldCondition(key="instructor", match=MatchValue(value=filter_professor)))
                print(f"   👨‍🏫 Filtering by professor: {filter_professor}")
            
            # If no filters, return all documents
            if not filters:
                print("   ⚠️ No filters applied, returning all documents")
                return self.chunks
            
            # Create combined filter
            combined_filter = Filter(must=filters)
            
            # Search with filter (we'll use a dummy vector since we only care about filtering)
            dummy_vector = [0.0] * self.embeddings.shape[1] if self.embeddings is not None else [0.0] * 768
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=dummy_vector,
                query_filter=combined_filter,
                limit=1000,  # Get all matching documents
                with_payload=True,
                with_vectors=False
            )
            
            print(f"   ✅ Found {len(results)} documents after filtering")
            
            # Convert results back to chunk format
            filtered_chunks = []
            for result in results:
                chunk_data = {
                    'id': result.id,
                    'course': result.payload['course'],
                    'instructor': result.payload['instructor'],
                    'document_name': result.payload['document_name'],
                    'section_title': result.payload['section_title'],
                    'slide_number': result.payload['slide_number'],
                    'chunk_text': result.payload['chunk_text']
                }
                filtered_chunks.append(chunk_data)
            
            return filtered_chunks
            
        except Exception as e:
            print(f"❌ Error filtering by course and professor: {e}")
            return []
    
    def search_concept_within_filtered(self, concept: str, filtered_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Search for concept within filtered chunks using semantic similarity"""
        try:
            if not concept or not filtered_chunks or self.embeddings is None:
                return []
            
            # Create embedding for the concept
            model = self._get_embedding_model()
            concept_embedding = model.encode(concept)
            
            # Get IDs of filtered chunks
            chunk_ids = [chunk['id'] for chunk in filtered_chunks]
            
            # Search in Qdrant with the concept embedding
            print(f"   🔍 Executing Qdrant semantic search...")
            
            # FIXED: Search in Qdrant without filter first, then filter results
            # This avoids the filter bug that was preventing results
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=concept_embedding.tolist(),
                limit=top_k * 3,  # Get more results to filter from
                with_payload=True,
                with_vectors=False
            )
            
            # Filter results to only include chunks from our filtered set
            filtered_results = []
            chunk_id_set = set(chunk_ids)  # Convert to set for faster lookup
            
            for result in search_results:
                if result.id in chunk_id_set:
                    filtered_results.append(result)
                    if len(filtered_results) >= top_k:
                        break
            
            search_results = filtered_results
            
            # Format results with source information
            formatted_results = []
            for result in search_results:
                formatted_result = {
                    'score': result.score,
                    'course': result.payload['course'],
                    'instructor': result.payload['instructor'],
                    'document_name': result.payload['document_name'],
                    'section_title': result.payload['section_title'],
                    'slide_number': result.payload['slide_number'],
                    'chunk_text': result.payload['chunk_text']
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            print(f"❌ Error searching concept: {e}")
            return []
    
    def search_lecture_notes(self, course: Optional[str] = None, professor: Optional[str] = None, 
                            concept: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Main search method: 3-step process - similarity search, filter, then concept search"""
        try:
            print(f"\n🔍 Starting 3-step search process...")
            print(f"   Query: Course='{course}', Professor='{professor}', Concept='{concept}'")
            
            # Step 1: Similarity search for course and professor names
            print("\n📋 Step 1: Finding similar course and professor names...")
            similar_course, similar_professor = self.find_similar_course_and_professor(course, professor)
            
            # Step 2: Filter documents by course and professor (using similar names if found)
            print("\n🔍 Step 2: Filtering documents by course and professor...")
            filtered_chunks = self.filter_by_course_and_professor(
                similar_course if similar_course else course, 
                similar_professor if similar_professor else professor
            )
            
            if not filtered_chunks:
                return {
                    'success': False,
                    'message': f"No documents found for course: '{course}' → '{similar_course or 'No match'}', professor: '{professor}' → '{similar_professor or 'No match'}'",
                    'results': [],
                    'total_filtered': 0,
                    'similarity_matches': {
                        'course': similar_course,
                        'professor': similar_professor
                    }
                }
            
            # Step 3: Search for concept within filtered chunks
            if concept:
                print(f"\n🎯 Step 3: Searching for concept '{concept}' within {len(filtered_chunks)} filtered documents...")
                search_results = self.search_concept_within_filtered(concept, filtered_chunks, top_k)
                
                if not search_results:
                    return {
                        'success': False,
                        'message': f"No results found for concept '{concept}' in the filtered documents",
                        'results': [],
                        'total_filtered': len(filtered_chunks),
                        'similarity_matches': {
                            'course': similar_course,
                            'professor': similar_professor
                        },
                        'search_query': {
                            'course': course,
                            'professor': professor,
                            'concept': concept
                        }
                    }
                
                print(f"   ✅ Found {len(search_results)} relevant results for concept '{concept}'")
                return {
                    'success': True,
                    'message': f"Found {len(search_results)} relevant results",
                    'results': search_results,
                    'total_filtered': len(filtered_chunks),
                    'similarity_matches': {
                        'course': similar_course,
                        'professor': similar_professor
                    },
                    'search_query': {
                        'course': course,
                        'professor': professor,
                        'concept': concept
                    }
                }
            else:
                # If no concept provided, return filtered chunks
                print(f"   ✅ No concept specified, returning {len(filtered_chunks)} filtered documents")
                return {
                    'success': True,
                    'message': f"Found {len(filtered_chunks)} documents",
                    'results': filtered_chunks[:top_k],
                    'total_filtered': len(filtered_chunks),
                    'similarity_matches': {
                        'course': similar_course,
                        'professor': similar_professor
                    },
                    'search_query': {
                        'course': course,
                        'professor': professor,
                        'concept': None
                    }
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error during search: {str(e)}",
                'results': [],
                'total_filtered': 0,
                'similarity_matches': {
                    'course': None,
                    'professor': None
                }
            }


