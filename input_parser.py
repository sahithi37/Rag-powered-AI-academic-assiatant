import os
import json
from typing import Dict, Optional
import google.generativeai as genai

class InputParser:
    def __init__(self):
        """Initialize the input parser with Gemini 2.0 Flash"""
        # Get API key from environment
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
        # If not found in environment, try to read from .env file manually
        if not self.api_key:
            try:
                with open('.env', 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('GOOGLE_API_KEY='):
                            self.api_key = line.split('=', 1)[1].strip()
                            break
            except Exception as e:
                print(f"⚠️ Warning: Could not read .env file: {e}")
        
        # If still not found, ask user to input manually
        if not self.api_key or self.api_key == 'your_actual_api_key_here':
            print("🔑 Google API Key not found. Please enter it manually:")
            self.api_key = input("Enter your Google API Key: ").strip()
            
        if not self.api_key:
            raise ValueError("Google API Key is required to use Gemini 2.0 Flash")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def extract_query_components(self, student_query: str) -> Dict[str, Optional[str]]:
        """
        Extract course ID, professor name, and concept from student query using LLM
        
        Args:
            student_query: Natural language query from student
            
        Returns:
            Dictionary with 'course_id', 'professor', 'concept' keys
        """
        try:
            # Create the prompt for the LLM
            prompt = self._create_extraction_prompt(student_query)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response
            extracted_data = self._parse_llm_response(response.text)
            
            # Clean the extracted data
            cleaned_data = self._clean_extracted_data(extracted_data)
            
            return cleaned_data
            
        except Exception as e:
            print(f"❌ Error extracting query components: {e}")
            return {
                'course_id': None,
                'professor': None,
                'concept': None,
                'error': str(e)
            }
    
    def _create_extraction_prompt(self, student_query: str) -> str:
        """Create the prompt for the LLM to extract components"""
        prompt = f"""
You are an academic assistant that helps extract information from student queries about lecture notes.

Extract the following information from this student query:
1. Course ID (any course identifier mentioned, like DSCI6004, CS101, etc.)
2. Professor/Instructor name (any instructor name mentioned)
3. Concept or topic being searched for

Student Query: "{student_query}"

Respond ONLY with a JSON object in this exact format:
{{
    "course_id": "DSCI6004",
    "professor": "Professor Smith", 
    "concept": "neural networks"
}}

If any information is missing or unclear, use null for that field.
Extract exactly what the student mentioned - don't restrict or validate against any predefined lists.
"""
        return prompt
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Optional[str]]:
        """Parse the LLM response to extract the structured data"""
        try:
            # Clean the response text
            cleaned_response = llm_response.strip()
            
            # Remove markdown formatting if present
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response.split('```json')[1]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response.rsplit('```', 1)[0]
            
            cleaned_response = cleaned_response.strip()
            
            # Parse JSON
            parsed_data = json.loads(cleaned_response)
            
            return {
                'course_id': parsed_data.get('course_id'),
                'professor': parsed_data.get('professor'),
                'concept': parsed_data.get('concept')
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing LLM response: {e}")
            return {
                'course_id': None,
                'professor': None,
                'concept': None
            }
    
    def _clean_extracted_data(self, data: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
        """Clean the extracted data (just trim whitespace)"""
        cleaned = data.copy()
        
        # Clean all fields (just trim whitespace)
        for key in cleaned:
            if cleaned[key]:
                cleaned[key] = cleaned[key].strip()
        
        return cleaned
