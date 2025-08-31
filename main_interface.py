import json
import os
from typing import List, Dict, Any, Optional
from input_parser import InputParser
from embedding_manager import EmbeddingManager
from rag_chatbot import RAGChatbot

class AcademicChatbot:
    def __init__(self):
        """Initialize the complete academic chatbot system"""
        self.input_parser = None
        self.embedding_manager = None
        self.rag_chatbot = None
        self.chunks = []
        self.embeddings = None
        self.metadata = None
        
    def initialize_system(self, chunks_file: str = "all_chunks.json") -> bool:
        """Initialize the complete system step by step"""
        try:
            print("🚀 Initializing Academic Chatbot System...")
            
            # Step 1: Load chunks
            if not self._load_chunks(chunks_file):
                return False
            
            # Step 2: Initialize input parser
            if not self._initialize_input_parser():
                return False
            
            # Step 3: Initialize embedding manager
            if not self._initialize_embedding_manager():
                return False
            
            # Step 4: Initialize RAG chatbot
            if not self._initialize_rag_chatbot():
                return False
            
            print("✅ Academic Chatbot System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            return False
    
    def _load_chunks(self, chunks_file: str) -> bool:
        """Load document chunks"""
        try:
            if not os.path.exists(chunks_file):
                print(f"❌ Error: {chunks_file} not found!")
                return False
                
            with open(chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            print(f"✅ Successfully loaded {len(self.chunks)} chunks from {chunks_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading chunks: {e}")
            return False
    
    def _initialize_input_parser(self) -> bool:
        """Initialize the input parser"""
        try:
            print("🔄 Initializing Input Parser...")
            self.input_parser = InputParser()
            print("✅ Input Parser initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing input parser: {e}")
            return False
    
    def _initialize_embedding_manager(self) -> bool:
        """Initialize the embedding manager"""
        try:
            print("🔄 Initializing Embedding Manager...")
            self.embedding_manager = EmbeddingManager()
            
            # Get or create embeddings
            self.embeddings, self.metadata = self.embedding_manager.get_or_create_embeddings(self.chunks)
            
            print("✅ Embedding Manager initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing embedding manager: {e}")
            return False
    
    def _initialize_rag_chatbot(self) -> bool:
        """Initialize the RAG chatbot"""
        try:
            print("🔄 Initializing RAG Chatbot...")
            self.rag_chatbot = RAGChatbot()
            
            # Initialize with pre-loaded embeddings
            if not self.rag_chatbot.initialize_with_embeddings(self.chunks, self.embeddings, self.metadata):
                return False
            
            print("✅ RAG Chatbot initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing RAG chatbot: {e}")
            return False
    
    def process_student_query(self, student_query: str) -> Dict[str, Any]:
        """Process a student query through the complete pipeline"""
        try:
            print(f"\n🎓 Processing Student Query: {student_query}")
            print("=" * 60)
            
            # Step 1: Extract components using LLM
            print("🔄 Step 1: Extracting query components...")
            extracted_data = self.input_parser.extract_query_components(student_query)
            
            if 'error' in extracted_data:
                return {
                    'success': False,
                    'message': f"Error parsing query: {extracted_data['error']}",
                    'extracted_data': extracted_data
                }
            
            print(f"✅ Extracted: Course={extracted_data['course_id']}, Professor={extracted_data['professor']}, Concept={extracted_data['concept']}")
            
            # Step 2: Search using RAG system
            print("🔄 Step 2: Searching lecture notes...")
            search_results = self.rag_chatbot.search_lecture_notes(
                course=extracted_data['course_id'],
                professor=extracted_data['professor'],
                concept=extracted_data['concept']
            )
            
            # Step 3: Format results
            if search_results['success']:
                formatted_results = self._format_search_results(search_results)
                return {
                    'success': True,
                    'extracted_data': extracted_data,
                    'search_results': search_results,
                    'formatted_results': formatted_results
                }
            else:
                return {
                    'success': False,
                    'message': search_results['message'],
                    'extracted_data': extracted_data,
                    'search_results': search_results
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"Error processing query: {str(e)}",
                'extracted_data': None,
                'search_results': None
            }
    
    def _format_search_results(self, search_results: Dict) -> str:
        """Format search results for display"""
        try:
            if not search_results['results']:
                return "No relevant results found."
            
            formatted = f"\n📚 Found {len(search_results['results'])} relevant results:\n"
            formatted += "=" * 60 + "\n"
            
            for i, result in enumerate(search_results['results'], 1):
                formatted += f"\n{i}. 📄 Document: {result['document_name']}\n"
                formatted += f"   🎓 Course: {result['course']}\n"
                formatted += f"   👨‍🏫 Professor: {result['instructor']}\n"
                formatted += f"   📍 Section: {result['section_title']}\n"
                formatted += f"   🎯 Slide: {result['slide_number']}\n"
                formatted += f"   📊 Relevance Score: {result.get('score', 'N/A'):.4f}\n"
                formatted += f"   📝 Content: {result['chunk_text'][:200]}...\n"
                formatted += "-" * 40 + "\n"
            
            return formatted
            
        except Exception as e:
            return f"Error formatting results: {str(e)}"
    
    def test_system(self) -> bool:
        """Test the complete system with sample queries"""
        print("\n🧪 Testing Complete System...")
        print("=" * 60)
        
        test_queries = [
            "Find lecture notes about convolutional neural networks in DSCI6011 with Professor Animulislam",
            "Show me DSCI6004 materials about machine learning from khaledsayed",
            "What does Professor khaledsayed teach about deep learning?",
            "Find notes on neural networks in DSCI6011",
            "DSCI6004 khaledsayed artificial intelligence"
        ]
        
        success_count = 0
        for i, query in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}: {query}")
            result = self.process_student_query(query)
            
            if result['success']:
                print("✅ Query processed successfully")
                print(f"   Results: {result['search_results']['message']}")
                success_count += 1
            else:
                print(f"❌ Query failed: {result['message']}")
        
        print(f"\n📊 Test Results: {success_count}/{len(test_queries)} successful")
        return success_count == len(test_queries)
    
    def interactive_mode(self):
        """Run the chatbot in interactive mode"""
        print("\n🎓 Academic Chatbot - Interactive Mode")
        print("=" * 60)
        print("Type 'quit' to exit, 'test' to run system tests")
        print("Ask questions like:")
        print("- 'Find CNN notes from prof khaled in DSCI6004'")
        print("- 'What does Professor Animulislam teach about neural networks?'")
        print("- 'Show me machine learning materials from khaledsayed'")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n🎓 Student Query: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye! Thank you for using Academic Chatbot!")
                    break
                
                if user_input.lower() == 'test':
                    self.test_system()
                    continue
                
                if not user_input:
                    continue
                
                # Process the query
                result = self.process_student_query(user_input)
                
                if result['success']:
                    print("\n" + result['formatted_results'])
                else:
                    print(f"\n❌ {result['message']}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Thank you for using Academic Chatbot!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def main():
    """Main function to run the academic chatbot"""
    print("🚀 Starting Academic Chatbot...")
    
    try:
        # Create and initialize the chatbot
        chatbot = AcademicChatbot()
        
        if not chatbot.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        # Run interactive mode
        chatbot.interactive_mode()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
