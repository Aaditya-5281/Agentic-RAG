import os
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field, ConfigDict
from markitdown import MarkItDown
from chonkie import SemanticChunker, SentenceTransformerEmbeddings as SentenceTransformerEmbedding
from qdrant_client import QdrantClient, models
from firecrawl import FirecrawlApp

class DocumentSearchToolInput(BaseModel):
    """Input schema for DocumentSearchTool."""
    query: str = Field(..., description="Query to search the document.")

class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search the document for the given query."
    args_schema: Type[BaseModel] = DocumentSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        """Initialize the searcher with a PDF file path and set up the Qdrant collection."""
        super().__init__()
        self.file_path = file_path
        self.client = QdrantClient(":memory:")
        self.embedding_model = SentenceTransformerEmbedding("minishlab/potion-base-8M")
        self._process_document()

    def _extract_text(self) -> str:
        """Extract raw text from PDF using MarkItDown."""
        md = MarkItDown()
        result = md.convert(self.file_path)
        return result.text_content

    def _create_chunks(self, raw_text: str):
        """Create semantic chunks from raw text."""
        chunker = SemanticChunker(
            embedding_model=self.embedding_model,
            threshold=0.5,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _process_document(self):
        """Process the document and add chunks to Qdrant collection."""
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)

        docs = [chunk.text for chunk in chunks]
        ids = list(range(len(chunks)))
        metadata = [{"source": os.path.basename(self.file_path)} for _ in range(len(chunks))]

        # Create collection if not exists
        self.client.create_collection(
            collection_name="demo_collection",
            vectors_config=models.VectorParams(
                size=256,
                distance=models.Distance.COSINE
            )
        )

        # Embed documents
        embeddings = self.embedding_model.embed_batch(docs)

        # Upload with pre-computed embeddings
        points = []
        for i in range(len(docs)):
            vector = embeddings[i]
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            points.append(
                models.PointStruct(
                    id=ids[i],
                    vector=vector,
                    payload={"text": docs[i], "metadata": metadata[i]}
                )
            )

        self.client.upsert(
            collection_name="demo_collection",
            points=points
        )

    def _run(self, query: str) -> str:
        """Search the document with a query string."""
        query_embedding = self.embedding_model.embed(query)
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        search_results = self.client.query_points(
            collection_name="demo_collection",
            query=query_embedding,
            limit=5
        )

        docs = [hit.payload["text"] for hit in search_results.points]
        separator = "\n___\n"
        return separator.join(docs)

class FireCrawlWebSearchToolInput(BaseModel):
    """Input schema for FireCrawlWebSearchTool."""
    query: str = Field(..., description="Query to search the web.")

class FireCrawlWebSearchTool(BaseTool):
    name: str = "FireCrawlWebSearchTool"
    description: str = "Search the web using FireCrawl for the given query."
    args_schema: Type[BaseModel] = FireCrawlWebSearchToolInput

    model_config = ConfigDict(extra="allow")

    def __init__(self, api_key: str = None):
        """Initialize the FireCrawl web search tool."""
        super().__init__()
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        self.app = None
        if self.api_key:
            self.app = FirecrawlApp(api_key=self.api_key)

    def _run(self, query: str) -> str:
        """Search the web with a query string using FireCrawl."""
        if not self.app:
            return "Web search is not available. FIRECRAWL_API_KEY is not set."
        try:
            search_result = self.app.search(query, params={"pageOptions": {"onlyMainContent": True}})
            if search_result and 'data' in search_result:
                results = search_result['data']
                formatted_results = []
                for result in results[:5]:
                    formatted_results.append(f"Title: {result.get('title', 'N/A')}\nURL: {result.get('url', 'N/A')}\nContent: {result.get('markdown', 'N/A')[:1000]}")
                return "\n\n".join(formatted_results)
            return "No results found"
        except Exception as e:
            return f"Error searching web: {str(e)}"

# Test the implementation
def test_document_searcher():
    # Test file path
    pdf_path = "Users\aadit\OneDrive\Desktop\Agentic RAG\agentic_rag\knowledge\dspy.pdf"

    # Create instance
    searcher = DocumentSearchTool(file_path=pdf_path)

    # Test search
    result = searcher._run("What is the purpose of DSpy?")
    print("Search Results:", result)

if __name__ == "__main__":
    test_document_searcher()
