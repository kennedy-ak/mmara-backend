"""
Legal Document Chunker with Section-Based Strategy.
Uses LlamaParse for AI-powered PDF extraction.
"""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings


@dataclass
class LegalChunk:
    """Represents a chunk of legal text with metadata."""

    text: str
    metadata: Dict[str, Any]
    chunk_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "metadata": self.metadata, "chunk_id": self.chunk_id}


# Lazy-initialized LlamaParse client
_llama_parser = None


def _get_parser():
    """Lazy-init LlamaParse client."""
    global _llama_parser
    if _llama_parser is None:
        from llama_parse import LlamaParse

        _llama_parser = LlamaParse(
            api_key=settings.llamaparse_api_key,
            result_type="markdown",
            parsing_instruction=(
                "This is a Ghanaian legal document (Act, Legislative Instrument, or Regulation). "
                "Preserve all section numbers, article numbers, headings, and table formatting. "
                "Maintain the hierarchical structure of Parts, Chapters, Sections, and sub-sections."
            ),
        )
    return _llama_parser


class LegalDocumentChunker:
    """
    Intelligent chunker for legal documents.

    Strategies:
    1. Section-based: Split by legal sections (preferred for acts)
    2. Sentence-based: Split by sentences with overlap (fallback)
    3. Hybrid: Section detection with sentence-level refinement
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Patterns for detecting legal sections (includes markdown heading variants)
        self.section_patterns = [
            r"^#{1,3}\s*Section\s+\d+",
            r"^Section\s+\d+",
            r"^Sec\.\s*\d+",
            r"^\d+\.\s+",
            r"^#{1,3}\s*Article\s+\d+",
            r"^Article\s+\d+",
            r"^#{1,3}\s*Part\s+[IVXLCDM]+",
            r"^Part\s+[IVXLCDM]+",
            r"^#{1,3}\s*CHAPTER\s+[IVXLCDM]+",
            r"^CHAPTER\s+[IVXLCDM]+",
            r"^Act\s+\d+",
            r"^L\.I\.\s*\d+",
            r"^\(\d+\)",
        ]

    async def extract_text_from_pdf(self, pdf_path: Path) -> tuple[str, Dict[str, Any]]:
        """
        Extract text and basic metadata from PDF using LlamaParse.

        Returns:
            tuple: (full_text, base_metadata)
        """
        try:
            parser = _get_parser()
            documents = await parser.aload_data(str(pdf_path))

            # Combine all document pages
            full_text = "\n".join(doc.text for doc in documents)

            # Extract metadata
            metadata = {
                "source_file": pdf_path.name,
                "total_pages": len(documents),
                "file_size": pdf_path.stat().st_size,
            }

            return full_text, metadata

        except Exception as e:
            raise ValueError(f"Failed to process PDF {pdf_path}: {e}")

    def extract_legal_metadata(self, filename: str, text: str) -> Dict[str, Any]:
        """
        Extract legal metadata from filename and text content.

        Looks for:
        - Act numbers (ACT 30, Act 459)
        - Years (1960, 2003, 2017)
        - Document type (act, amendment, regulation)
        """
        metadata = {}

        # Extract Act number from filename or text
        act_pattern = r"(?:ACT|Act)\s*(\d+)"
        act_match = re.search(act_pattern, filename) or re.search(act_pattern, text[:500])
        if act_match:
            metadata["act_number"] = act_match.group(1)

        # Extract year
        year_pattern = r"\b(19\d{2}|20\d{2})\b"
        year_match = re.search(year_pattern, filename) or re.search(year_pattern, text[:500])
        if year_match:
            metadata["year"] = year_match.group(1)

        # Extract L.I. number (Legislative Instrument)
        li_pattern = r"L\.I\.?\s*(\d+)"
        li_match = re.search(li_pattern, filename) or re.search(li_pattern, text[:500])
        if li_match:
            metadata["legislative_instrument"] = f"L.I. {li_match.group(1)}"

        # Determine document type
        if "amendment" in filename.lower():
            metadata["doc_type"] = "amendment"
        elif "regulation" in filename.lower():
            metadata["doc_type"] = "regulation"
        elif "act" in filename.lower():
            metadata["doc_type"] = "act"
        elif "instrument" in filename.lower():
            metadata["doc_type"] = "legislative_instrument"
        else:
            metadata["doc_type"] = "other"

        # Determine category
        if any(
            term in filename.lower()
            for term in ["criminal", "offence", "procedure", "evidence", "juvenile"]
        ):
            metadata["category"] = "criminal"
        elif any(term in filename.lower() for term in ["road", "traffic", "driver", "vehicle"]):
            metadata["category"] = "road_traffic"

        return metadata

    def split_by_sections(self, text: str) -> List[tuple[str, str]]:
        """
        Split text into sections using legal section patterns.

        Returns a list of (header, section_text) tuples.
        """
        sections = []
        current_section = []
        current_header = ""

        lines = text.split("\n")

        for line in lines:
            # Check if this line is a section header
            is_section = False
            for pattern in self.section_patterns:
                if re.match(pattern, line.strip(), re.IGNORECASE):
                    is_section = True
                    break

            if is_section:
                # Save previous section
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if len(section_text) > self.min_chunk_size:
                        sections.append((current_header, section_text))

                # Start new section
                current_header = line.strip()
                current_section = [line]
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            section_text = "\n".join(current_section).strip()
            if len(section_text) > self.min_chunk_size:
                sections.append((current_header, section_text))

        return sections

    def split_by_sentences_with_overlap(self, text: str) -> List[str]:
        """
        Split text into chunks with sentence boundaries and overlap.

        Fallback strategy for documents without clear sections.
        """
        # Split by sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current_chunk = []

        for sentence in sentences:
            current_chunk.append(sentence)

            # Check if chunk exceeds size
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) > self.chunk_size:
                chunks.append(chunk_text)

                # Start new chunk with overlap
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    if overlap_size + len(sent) < self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += len(sent)
                    else:
                        break

                current_chunk = overlap_sentences

        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def chunk_document(self, pdf_path: Path, force_section_based: bool = False) -> List[LegalChunk]:
        """
        Chunk a legal document PDF.

        Args:
            pdf_path: Path to the PDF file
            force_section_based: If True, only use section-based chunking

        Returns:
            List of LegalChunk objects
        """
        # Extract text and base metadata (async with LlamaParse)
        full_text, base_metadata = await self.extract_text_from_pdf(pdf_path)

        # Extract legal-specific metadata
        legal_metadata = self.extract_legal_metadata(pdf_path.name, full_text)
        base_metadata.update(legal_metadata)

        chunks = []

        # Try section-based chunking first
        sections = self.split_by_sections(full_text)

        if sections and (force_section_based or len(sections) >= 3):
            # Section-based chunking successful
            for idx, (header, section_text) in enumerate(sections):
                chunk_metadata = base_metadata.copy()
                chunk_metadata["section_header"] = header
                chunk_metadata["section_number"] = idx + 1
                chunk_metadata["chunking_strategy"] = "section_based"

                # Further split large sections
                if len(section_text) > self.chunk_size * 2:
                    sub_chunks = self._split_large_section(section_text, header)
                    for sub_idx, sub_chunk in enumerate(sub_chunks):
                        sub_metadata = chunk_metadata.copy()
                        sub_metadata["sub_chunk"] = sub_idx + 1
                        chunks.append(
                            LegalChunk(
                                text=sub_chunk,
                                metadata=sub_metadata,
                                chunk_id=f"{pdf_path.stem}_sec{idx}_{sub_idx}",
                            )
                        )
                else:
                    chunks.append(
                        LegalChunk(
                            text=section_text,
                            metadata=chunk_metadata,
                            chunk_id=f"{pdf_path.stem}_sec{idx}",
                        )
                    )
        else:
            # Fall back to sentence-based chunking
            text_chunks = self.split_by_sentences_with_overlap(full_text)
            for idx, chunk_text in enumerate(text_chunks):
                chunk_metadata = base_metadata.copy()
                chunk_metadata["chunk_number"] = idx + 1
                chunk_metadata["chunking_strategy"] = "sentence_based"

                chunks.append(
                    LegalChunk(
                        text=chunk_text,
                        metadata=chunk_metadata,
                        chunk_id=f"{pdf_path.stem}_chunk{idx}",
                    )
                )

        return chunks

    def _split_large_section(self, text: str, header: str) -> List[str]:
        """Split a large section into smaller chunks."""
        chunks = []
        paragraphs = text.split("\n\n")

        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    async def chunk_document_async(self, pdf_path: Path) -> List[LegalChunk]:
        """
        Async chunking — delegates to chunk_document which is already async.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of LegalChunk objects
        """
        return await self.chunk_document(pdf_path)

    async def chunk_directory(
        self, directory: Path, pattern: str = "*.pdf", category: Optional[str] = None
    ) -> Dict[str, List[LegalChunk]]:
        """
        Chunk all PDF files in a directory asynchronously.

        Returns:
            Dict mapping file paths to their chunks
        """
        results = {}
        pdf_files = list(directory.rglob(pattern))

        tasks = [self.chunk_document_async(pdf_path) for pdf_path in pdf_files]
        chunks_list = await asyncio.gather(*tasks, return_exceptions=True)

        for pdf_path, chunks in zip(pdf_files, chunks_list):
            if isinstance(chunks, Exception):
                print(f"  ✗ {pdf_path.name}: {chunks}")
                continue

            # Add category to metadata if specified
            if category:
                for chunk in chunks:
                    chunk.metadata["category"] = category

            results[str(pdf_path)] = chunks
            print(f"  ✓ {pdf_path.name}: {len(chunks)} chunks")

        return results
