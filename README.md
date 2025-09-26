# 📚 Lecture Notes RAG Chatbot

A RAG-powered chatbot that helps students find relevant lecture notes based on course ID, professor name, and concept queries. Built with LangChain, Google Gemini 2.0 Flash, and FAISS vector database.

## 🚀 Features

- **Smart Search**: Find lecture notes by course, professor, and concept
- **RAG Integration**: Uses Retrieval-Augmented Generation for accurate answers
- **Vector Search**: Qdrant-based semantic search through document chunks
- **User-Friendly Interface**: Streamlit web interface for easy interaction
- **Source Attribution**: Shows exactly which documents and sections contain the information

## 🛠️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in your project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

**To get a Google API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### 3. Run the Chatbot

```bash
streamlit run rag_chatbot.py
```

## 📁 File Structure

```
├── rag_chatbot.py          # Main chatbot application
├── extracting_files.py      # Document extraction from Google Drive
├── all_chunks.json         # Extracted document chunks
├── requirements.txt         # Python dependencies
├── credentials.json         # Google Drive API credentials
└── token.pickle            # Google Drive authentication token
```

## 🎯 How It Works

1. **Document Processing**: Your existing `extracting_files.py` extracts chunks from Google Drive documents
2. **Vector Storage**: Chunks are converted to embeddings and stored in Qdrant vector database
3. **RAG Search**: When a student asks a question:
   - Filters chunks by course ID and professor name
   - Searches for relevant content using semantic similarity
   - Generates answers using Google Gemini 2.0 Flash
   - Provides source attribution

## 💬 Usage Example

1. **Select Course**: Choose from available courses (e.g., "DSCI6011")
2. **Select Professor**: Choose the professor (e.g., "Animulislam")
3. **Ask Question**: Enter your concept query (e.g., "What are convolutional neural networks?")
4. **Get Results**: Receive relevant lecture notes with source information

## 🔧 Customization

### Adding New Documents

1. Run `extracting_files.py` to extract new chunks
2. The chatbot automatically loads the updated `all_chunks.json`

### Modifying Search Parameters

- Adjust `search_kwargs={"k": 5}` in the code to change how many chunks are retrieved
- Modify the chunk filtering logic in `filter_chunks_by_course_and_professor()`

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure your `.env` file contains the correct Google API key
2. **No Chunks Found**: Verify that `all_chunks.json` exists and contains data
3. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`

### Fallback Mode

If the RAG system fails to initialize, the chatbot falls back to simple text-based search.

## 📊 Performance

- **Vector Search**: Fast semantic similarity search using Qdrant
- **Caching**: Streamlit session state maintains chatbot instance
- **Efficient Filtering**: Pre-filters chunks by course/professor before semantic search

## 🔮 Future Enhancements

- [ ] Add conversation history
- [ ] Support for more document types
- [ ] Advanced filtering options
- [ ] Export search results
- [ ] Integration with learning management systems

## 📝 License

This project is open source and available under the MIT License.


