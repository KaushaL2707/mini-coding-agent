"""
Semantic retrieval module.

STEP 4: Find relevant code chunks for a given prompt.
"""
from typing import Optional

from config import TOP_K_CHUNKS
from embed import VectorStore
from ingest import CodeChunk
from tools import format_file_for_context, estimate_tokens


class CodeRetriever:
    """
    Retrieves relevant code chunks for a given query.
    Handles deduplication and context window management.
    """
    
    def __init__(self, store: Optional[VectorStore] = None):
        self.store = store or VectorStore()
    
    def load_index(self, name: str = "default") -> bool:
        """Load a pre-built index."""
        return self.store.load(name)
    
    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
        max_tokens: int = 8000
    ) -> list[tuple[CodeChunk, float]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's prompt/question
            top_k: Maximum number of chunks to retrieve
            max_tokens: Approximate token budget for context
            
        Returns:
            List of (chunk, score) tuples within token budget
        """
        # Get raw results
        results = self.store.search(query, top_k=top_k * 2)  # Get extra for filtering
        
        # Filter and deduplicate
        seen_content = set()
        filtered_results = []
        total_tokens = 0
        
        for chunk, score in results:
            # Skip near-duplicates
            content_hash = hash(chunk.content[:500])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Check token budget
            chunk_tokens = estimate_tokens(chunk.content)
            if total_tokens + chunk_tokens > max_tokens:
                continue
            
            filtered_results.append((chunk, score))
            total_tokens += chunk_tokens
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def retrieve_as_context(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
        max_tokens: int = 8000
    ) -> str:
        """
        Retrieve chunks and format as LLM context.
        
        Args:
            query: User's prompt/question
            top_k: Maximum chunks
            max_tokens: Token budget
            
        Returns:
            Formatted string for LLM context
        """
        results = self.retrieve(query, top_k, max_tokens)
        
        if not results:
            return "No relevant code found in the repository."
        
        context_parts = [
            "Here are the most relevant code sections from the repository:\n"
        ]
        
        for i, (chunk, score) in enumerate(results, 1):
            location = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
            context_parts.append(f"\n--- [{i}] {location} (relevance: {score:.2f}) ---\n")
            context_parts.append(f"```\n{chunk.content}\n```\n")
        
        return "\n".join(context_parts)
    
    def get_file_context(self, file_path: str) -> Optional[str]:
        """
        Get full content of a specific file from indexed chunks.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Combined content of all chunks from that file
        """
        matching_chunks = [
            chunk for chunk in self.store.chunks
            if chunk.file_path == file_path
        ]
        
        if not matching_chunks:
            return None
        
        # Sort by line number and combine
        matching_chunks.sort(key=lambda c: c.start_line)
        
        # Avoid duplicating overlapping content
        combined = []
        last_end = 0
        
        for chunk in matching_chunks:
            if chunk.start_line > last_end:
                combined.append(chunk.content)
            elif chunk.end_line > last_end:
                # Partial overlap - get only new lines
                lines = chunk.content.split("\n")
                new_start = last_end - chunk.start_line + 1
                if new_start < len(lines):
                    combined.append("\n".join(lines[new_start:]))
            last_end = max(last_end, chunk.end_line)
        
        return "\n".join(combined)


def retrieve_for_prompt(
    query: str,
    index_name: str = "default",
    top_k: int = TOP_K_CHUNKS
) -> str:
    """
    Convenience function: load index and retrieve context for a prompt.
    
    Args:
        query: User's question/task
        index_name: Name of the stored index
        top_k: Number of chunks to retrieve
        
    Returns:
        Formatted context string
    """
    retriever = CodeRetriever()
    
    if not retriever.load_index(index_name):
        return "Error: No index found. Please run indexing first."
    
    return retriever.retrieve_as_context(query, top_k)


if __name__ == "__main__":
    import sys
    
    # Test retrieval
    query = sys.argv[1] if len(sys.argv) > 1 else "fix the bug in the main function"
    index_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    
    print(f"🔍 Query: {query}")
    print(f"📂 Index: {index_name}")
    print("\n" + "="*60 + "\n")
    
    context = retrieve_for_prompt(query, index_name)
    print(context)
