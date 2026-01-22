"""
Main Agent module - the reasoning loop.

This brings together all components:
1. Ingest → 2. Embed → 3. Retrieve → 4. LLM → 5. Output

Usage:
    python agent.py --repo /path/to/repo --prompt "fix the authentication bug"
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from config import TOP_K_CHUNKS
from embed import VectorStore, index_repository
from retrieve import CodeRetriever
from llm import LLM


class CodingAgent:
    """
    The main coding agent that orchestrates the entire pipeline.
    """
    
    def __init__(
        self,
        repo_path: Optional[str] = None,
        index_name: str = "default",
        llm_provider: Optional[str] = None
    ):
        self.repo_path = repo_path
        self.index_name = index_name
        self.retriever = CodeRetriever()
        self.llm = LLM()
        self.indexed = False
        
    def index(self, force: bool = False) -> bool:
        """
        Index the repository. Skip if already indexed.
        
        Args:
            force: Force re-indexing even if index exists
            
        Returns:
            True if indexing was successful
        """
        if not self.repo_path:
            print("❌ No repository path specified")
            return False
        
        # Check if index already exists
        if not force and self.retriever.load_index(self.index_name):
            print(f"📂 Loaded existing index: {self.index_name}")
            self.indexed = True
            return True
        
        # Create new index
        print(f"\n{'='*60}")
        print(f"🔄 Indexing repository: {self.repo_path}")
        print(f"{'='*60}\n")
        
        try:
            store = index_repository(self.repo_path, self.index_name)
            self.retriever.store = store
            self.indexed = True
            return True
        except Exception as e:
            print(f"❌ Indexing failed: {e}")
            return False
    
    def query(self, prompt: str, top_k: int = TOP_K_CHUNKS) -> str:
        """
        Process a user query: retrieve relevant code and get LLM response.
        
        Args:
            prompt: User's question or request
            top_k: Number of chunks to retrieve
            
        Returns:
            LLM's response
        """
        if not self.indexed:
            if not self.index():
                return "Error: Repository not indexed. Please index first."
        
        print(f"\n{'='*60}")
        print(f"🔍 Query: {prompt}")
        print(f"{'='*60}\n")
        
        # Step 1: Retrieve relevant code
        print("📚 Retrieving relevant code...")
        context = self.retriever.retrieve_as_context(prompt, top_k)
        
        # Show what was retrieved
        results = self.retriever.retrieve(prompt, top_k)
        print(f"\n📄 Found {len(results)} relevant chunks:")
        for chunk, score in results[:5]:
            print(f"   • {chunk.file_path}:{chunk.start_line}-{chunk.end_line} (score: {score:.2f})")
        if len(results) > 5:
            print(f"   ... and {len(results) - 5} more")
        
        # Step 2: Get LLM response
        print("\n🤖 Analyzing with LLM...")
        response = self.llm.analyze_code(context, prompt)
        
        return response
    
    def suggest_fix(self, issue: str, top_k: int = TOP_K_CHUNKS) -> str:
        """
        Dedicated method for bug fixes.
        
        Args:
            issue: Description of the bug/issue
            top_k: Number of chunks to retrieve
            
        Returns:
            LLM's fix suggestion
        """
        if not self.indexed:
            if not self.index():
                return "Error: Repository not indexed. Please index first."
        
        context = self.retriever.retrieve_as_context(issue, top_k)
        return self.llm.suggest_fix(context, issue)
    
    def interactive(self):
        """Run an interactive REPL for queries."""
        print(f"\n{'='*60}")
        print("🤖 Mini Coding Agent - Interactive Mode")
        print("='*60")
        print("Commands:")
        print("  /index   - Re-index the repository")
        print("  /quit    - Exit")
        print("  /help    - Show this help")
        print(f"{'='*60}\n")
        
        while True:
            try:
                prompt = input("\n🔹 Your query: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ["/quit", "/exit", "/q"]:
                    print("👋 Goodbye!")
                    break
                
                if prompt.lower() == "/index":
                    self.index(force=True)
                    continue
                
                if prompt.lower() == "/help":
                    print("Commands: /index, /quit, /help")
                    print("Or type any question about the code.")
                    continue
                
                # Process query
                response = self.query(prompt)
                
                print(f"\n{'='*60}")
                print("💡 Response:")
                print(f"{'='*60}\n")
                print(response)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Mini Coding Agent - Repo-aware AI coding assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index and query a repository
  python agent.py --repo ./my_project --prompt "find the authentication bug"
  
  # Interactive mode
  python agent.py --repo ./my_project --interactive
  
  # Use a specific LLM provider
  python agent.py --repo ./my_project --provider anthropic --prompt "explain the caching logic"
  
  # Force re-indexing
  python agent.py --repo ./my_project --reindex --prompt "what does main.py do?"
"""
    )
    
    parser.add_argument(
        "--repo", "-r",
        type=str,
        default=".",
        help="Path to the repository to analyze (default: current directory)"
    )
    
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Query or request to process"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "groq"],
        help="LLM provider to use (default: from config)"
    )
    
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of the repository"
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=TOP_K_CHUNKS,
        help=f"Number of code chunks to retrieve (default: {TOP_K_CHUNKS})"
    )
    
    parser.add_argument(
        "--index-name",
        type=str,
        default="default",
        help="Name for the vector store index (default: 'default')"
    )
    
    args = parser.parse_args()
    
    # Resolve repo path
    repo_path = str(Path(args.repo).resolve())
    
    # Create agent
    agent = CodingAgent(
        repo_path=repo_path,
        index_name=args.index_name,
        llm_provider=args.provider
    )
    
    # Index if needed
    if args.reindex or not agent.retriever.load_index(args.index_name):
        if not agent.index(force=args.reindex):
            sys.exit(1)
    else:
        agent.indexed = True
    
    # Run in interactive or single-query mode
    if args.interactive:
        agent.interactive()
    elif args.prompt:
        response = agent.query(args.prompt, args.top_k)
        print(f"\n{'='*60}")
        print("💡 Response:")
        print(f"{'='*60}\n")
        print(response)
    else:
        # Default to interactive if no prompt given
        agent.interactive()


if __name__ == "__main__":
    main()
