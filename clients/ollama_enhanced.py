import asyncio
from typing import List, Optional

from clients.ollama_client import OllamaClient
from clients.semantic_scholar_client import PaperMetadata


class EnhancedOllamaClient:
    """Enhanced Ollama client with research-specific capabilities"""

    def __init__(self, model_name: str = "llama3.2:1b"):
        self.client = OllamaClient()
        self.model_name = model_name

    async def enhance_search_query(self, query: str) -> str:
        """Use LLM to enhance the search query"""
        prompt = f"""You are a research paper search specialist. Enhance this search query: "{query}"
        Add 2-3 relevant academic terms and remove non-essential words.
        Keep the query focused and concise. Return only the enhanced query.
        Do NOT add medical terms unless the query is specifically about medicine.
        
        Example input: "ai in education"
        Example output: artificial intelligence educational technology machine learning pedagogy
        
        Your enhanced query (max 10 words):"""

        try:
            enhanced_query = await self.client.generate(
                prompt=prompt, model=self.model_name, max_tokens=50
            )
            # Clean up the query
            cleaned_query = enhanced_query.strip().replace("\n", " ").strip('"')
            return cleaned_query if cleaned_query else query
        except Exception as e:
            print(f"Query enhancement failed: {str(e)}")
            return query  # Fallback to original query

    async def summarize_results(
        self, papers: List[PaperMetadata], max_papers: int = 3
    ) -> str:
        """Generate a summary of search results"""
        papers_text = "\n".join(
            [
                f"Title: {p.title}\nYear: {p.year}\nCitations: {p.citationCount}\n"
                for p in papers[:max_papers]
            ]
        )

        prompt = f"""Analyze these academic papers and provide a brief summary:
        {papers_text}
        
        Focus on:
        1. Common themes
        2. Notable findings
        3. Research trends
        Keep it concise (2-3 sentences)."""

        try:
            summary = await self.client.generate(
                prompt=prompt, model=self.model_name, max_tokens=150
            )
            return f"\nSummary of top results:\n{summary.strip()}\n"
        except Exception as e:
            print(f"Summarization failed: {str(e)}")
            return ""  # Skip summary on error

    async def analyze_paper(self, paper: PaperMetadata) -> str:
        """Generate detailed analysis of a single paper"""
        paper_text = f"""Title: {paper.title}
        Authors: {', '.join(a.get('name', '') for a in paper.authors)}
        Year: {paper.year}
        Citations: {paper.citationCount}
        Abstract: {paper.abstract or 'Not available'}
        """

        prompt = f"""Analyze this academic paper and provide insights:
        {paper_text}
        
        Include:
        1. Main contributions
        2. Significance in the field
        3. Potential applications
        Make it concise but informative."""

        try:
            analysis = await self.client.generate(
                prompt=prompt, model=self.model_name, max_tokens=200
            )
            return analysis.strip()
        except Exception as e:
            print(f"Paper analysis failed: {str(e)}")
            return "Analysis not available due to an error."

    async def suggest_related_queries(self, query: str) -> List[str]:
        """Generate related search queries"""
        prompt = f"""For the academic query: "{query}"
        Generate 3 related academic search queries.
        IMPORTANT: Return ONLY the queries, one per line, no numbering or explanations.
        Example:
        deep learning vision architectures
        neural networks computer vision applications
        convolutional networks image recognition
        Your suggestions:"""

        try:
            suggestions = await self.client.generate(
                prompt=prompt, model=self.model_name, max_tokens=100
            )
            return [s.strip() for s in suggestions.strip().split("\n") if s.strip()]
        except Exception as e:
            print(f"Query suggestion failed: {str(e)}")
            return []


# Helper function for synchronous calls
def sync_enhance_query(query: str, model_name: str = "llama3.2:1b") -> str:
    """Synchronous wrapper for query enhancement"""

    async def _enhance():
        client = EnhancedOllamaClient(model_name=model_name)
        return await client.enhance_search_query(query)

    return asyncio.run(_enhance())


if __name__ == "__main__":
    # Test the enhanced client
    async def test_client():
        client = EnhancedOllamaClient()

        # Test query enhancement
        query = "machine learning for biology"
        enhanced = await client.enhance_search_query(query)
        print(f"Original query: {query}")
        print(f"Enhanced query: {enhanced}\n")

        # Test query suggestions
        suggestions = await client.suggest_related_queries(query)
        print("Related queries:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

    asyncio.run(test_client())
