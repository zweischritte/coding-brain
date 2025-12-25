"""
Production-ready chunking implementations for Business Concept Development System.

This module provides three ready-to-use chunkers:
1. TranscriptChunker - For meeting recordings with speaker diarization
2. StructureAwareChunker - For PDFs and structured documents
3. HybridChunker - Auto-detects content type and uses appropriate strategy

Usage:
    from chunking_examples import HybridChunker

    chunker = HybridChunker()
    chunks = await chunker.chunk_file("meeting_recording.mp3")

    for chunk in chunks:
        await add_memories(text=chunk.text, vault="WLT", ...)
"""

import re
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

# Optional dependencies (install as needed)
try:
    import assemblyai as aai
    HAS_ASSEMBLYAI = True
except ImportError:
    HAS_ASSEMBLYAI = False

try:
    import pymupdf4llm
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Chunk:
    """Represents a single chunk of content."""
    text: str
    chunk_index: int
    metadata: Dict

    # Optional fields
    section_path: str = ""
    heading_level: int = 0
    speaker: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def chunk_id(self) -> str:
        """Generate stable chunk ID."""
        content = f"{self.text}{self.metadata.get('source_doc', '')}"
        return hashlib.sha256(content.encode()).hexdigest()


class ContentType(Enum):
    """Content type detection."""
    AUDIO = "audio"
    PDF = "pdf"
    MARKDOWN = "markdown"
    PLAIN_TEXT = "text"


# ============================================================================
# Transcript Chunker (Speaker-Aware)
# ============================================================================

class TranscriptChunker:
    """
    Chunks audio transcripts by speaker turns with semantic boundary detection.

    Features:
    - Uses AssemblyAI speaker diarization
    - Merges short utterances from same speaker
    - Splits long monologues at sentence boundaries
    - Adds 15% overlap at speaker transitions

    Example:
        chunker = TranscriptChunker(max_tokens=512)
        chunks = await chunker.transcribe_and_chunk("meeting.mp3")
    """

    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 100,
        overlap_sentences: int = 2,
        merge_same_speaker_gap: float = 2.0,
        api_key: Optional[str] = None
    ):
        if not HAS_ASSEMBLYAI:
            raise ImportError("AssemblyAI not installed. Run: pip install assemblyai")

        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_sentences = overlap_sentences
        self.merge_same_speaker_gap = merge_same_speaker_gap

        if api_key:
            aai.settings.api_key = api_key

    async def transcribe_and_chunk(
        self,
        audio_path: str,
        metadata: Optional[Dict] = None
    ) -> List[Chunk]:
        """Transcribe audio and chunk by speaker turns."""

        # 1. Transcribe with AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(
            audio_path,
            config=aai.TranscriptionConfig(
                speaker_labels=True,
                punctuate=True,
                format_text=True
            )
        )

        # 2. Merge consecutive utterances from same speaker
        merged = self._merge_speaker_turns(transcript.utterances)

        # 3. Chunk long utterances
        chunks = []
        for utterance in merged:
            token_count = self._count_tokens(utterance["text"])

            if token_count <= self.max_tokens:
                chunks.append(self._create_chunk(utterance, len(chunks), metadata))
            else:
                sub_chunks = self._split_long_utterance(utterance, len(chunks), metadata)
                chunks.extend(sub_chunks)

        # 4. Add overlap at speaker transitions
        chunks = self._add_overlap(chunks)

        return chunks

    def _merge_speaker_turns(self, utterances: List) -> List[Dict]:
        """Merge consecutive utterances from same speaker."""
        merged = []
        current = None

        for utt in utterances:
            utt_data = {
                "text": utt.text,
                "speaker": utt.speaker,
                "start": utt.start / 1000,  # ms to seconds
                "end": utt.end / 1000
            }

            if current is None:
                current = utt_data
            elif (utt.speaker == current["speaker"] and
                  (utt_data["start"] - current["end"]) < self.merge_same_speaker_gap):
                # Merge with current
                current["text"] += " " + utt.text
                current["end"] = utt_data["end"]
            else:
                # Save current and start new
                merged.append(current)
                current = utt_data

        if current:
            merged.append(current)

        return merged

    def _split_long_utterance(
        self,
        utterance: Dict,
        start_index: int,
        metadata: Optional[Dict]
    ) -> List[Chunk]:
        """Split long utterance at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', utterance["text"])

        chunks = []
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self.max_tokens and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=start_index + len(chunks),
                    metadata=metadata or {},
                    speaker=utterance["speaker"],
                    start_time=utterance["start"],
                    end_time=utterance["end"]
                ))
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        if current_sentences:
            chunks.append(Chunk(
                text=" ".join(current_sentences),
                chunk_index=start_index + len(chunks),
                metadata=metadata or {},
                speaker=utterance["speaker"],
                start_time=utterance["start"],
                end_time=utterance["end"]
            ))

        return chunks

    def _create_chunk(
        self,
        utterance: Dict,
        index: int,
        metadata: Optional[Dict]
    ) -> Chunk:
        """Create chunk from utterance."""
        return Chunk(
            text=utterance["text"],
            chunk_index=index,
            metadata=metadata or {},
            speaker=utterance["speaker"],
            start_time=utterance["start"],
            end_time=utterance["end"]
        )

    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add sentence overlap at speaker transitions."""
        for i in range(1, len(chunks)):
            if chunks[i].speaker != chunks[i-1].speaker:
                # Speaker transition - add overlap
                prev_sentences = re.split(r'(?<=[.!?])\s+', chunks[i-1].text)
                overlap_text = " ".join(prev_sentences[-self.overlap_sentences:])
                chunks[i].text = f"{overlap_text} {chunks[i].text}"

        return chunks

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Approximate token count (4 chars ≈ 1 token)."""
        return len(text) // 4


# ============================================================================
# Structure-Aware Document Chunker
# ============================================================================

class StructureAwareChunker:
    """
    Chunks documents by structure, preserving heading hierarchy.

    Features:
    - Parses markdown headings into hierarchical sections
    - Chunks within sections at paragraph boundaries
    - Includes heading path in metadata
    - Adds 15% overlap at section boundaries

    Example:
        chunker = StructureAwareChunker(max_tokens=512)
        chunks = chunker.chunk_pdf("business_plan.pdf")
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_percentage: float = 0.15,
        min_chunk_tokens: int = 50,
        include_section_in_text: bool = True
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = int(max_tokens * overlap_percentage)
        self.min_chunk_tokens = min_chunk_tokens
        self.include_section_in_text = include_section_in_text

    def chunk_pdf(self, pdf_path: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Extract and chunk PDF preserving structure."""
        if not HAS_PYMUPDF:
            raise ImportError("pymupdf4llm not installed. Run: pip install pymupdf4llm")

        # Extract as markdown
        markdown_text = pymupdf4llm.to_markdown(pdf_path)

        # Add PDF metadata
        pdf_metadata = metadata or {}
        pdf_metadata.update({
            "source_type": "pdf",
            "source_path": pdf_path
        })

        return self.chunk_markdown(markdown_text, pdf_metadata)

    def chunk_markdown(self, markdown_text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Chunk markdown preserving structure."""

        # 1. Parse into sections
        sections = self._parse_sections(markdown_text)

        # 2. Chunk each section
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section, metadata or {})
            chunks.extend(section_chunks)

        # 3. Add overlap at section boundaries
        chunks = self._add_section_overlap(chunks)

        return chunks

    def _parse_sections(self, markdown: str) -> List[Dict]:
        """Parse markdown into hierarchical sections."""
        lines = markdown.split('\n')
        sections = []
        current_section = None
        heading_stack = []

        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if heading_match:
                # Save previous section
                if current_section and current_section['content'].strip():
                    sections.append(current_section)

                # Start new section
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # Update heading stack
                heading_stack = heading_stack[:level-1]
                heading_stack.append(heading_text)

                current_section = {
                    'heading': heading_text,
                    'level': level,
                    'section_path': ' > '.join(heading_stack),
                    'content': ''
                }
            elif current_section is not None:
                current_section['content'] += line + '\n'

        # Add last section
        if current_section and current_section['content'].strip():
            sections.append(current_section)

        return sections

    def _chunk_section(self, section: Dict, metadata: Dict) -> List[Chunk]:
        """Chunk a single section at paragraph boundaries."""
        paragraphs = [p.strip() for p in section['content'].split('\n\n') if p.strip()]

        chunks = []
        current_paragraphs = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._count_tokens(para)

            if current_tokens + para_tokens > self.max_tokens and current_paragraphs:
                # Create chunk
                chunk_text = self._format_chunk(
                    section['section_path'],
                    '\n\n'.join(current_paragraphs)
                )

                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'heading': section['heading'],
                    'section_path': section['section_path'],
                    'heading_level': section['level']
                })

                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    metadata=chunk_metadata,
                    section_path=section['section_path'],
                    heading_level=section['level']
                ))

                current_paragraphs = [para]
                current_tokens = para_tokens
            else:
                current_paragraphs.append(para)
                current_tokens += para_tokens

        # Add remaining
        if current_paragraphs:
            chunk_text = self._format_chunk(
                section['section_path'],
                '\n\n'.join(current_paragraphs)
            )

            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'heading': section['heading'],
                'section_path': section['section_path'],
                'heading_level': section['level']
            })

            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=len(chunks),
                metadata=chunk_metadata,
                section_path=section['section_path'],
                heading_level=section['level']
            ))

        return chunks

    def _format_chunk(self, section_path: str, content: str) -> str:
        """Format chunk with optional section context."""
        if self.include_section_in_text:
            return f"[Section: {section_path}]\n\n{content}"
        return content

    def _add_section_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Add overlap at section boundaries."""
        for i in range(1, len(chunks)):
            if chunks[i].section_path != chunks[i-1].section_path:
                # Get last N tokens from previous chunk
                prev_tokens = chunks[i-1].text.split()[-self.overlap_tokens:]
                overlap_text = ' '.join(prev_tokens)
                chunks[i].text = f"{overlap_text} {chunks[i].text}"

        return chunks

    @staticmethod
    def _count_tokens(text: str) -> int:
        """Approximate token count."""
        return len(text) // 4


# ============================================================================
# Hybrid Chunker (Production Ready)
# ============================================================================

class HybridChunker:
    """
    Intelligent chunker that selects strategy based on content type.

    Decision tree:
    - Audio file? → TranscriptChunker (with speaker diarization)
    - PDF? → StructureAwareChunker (with markdown extraction)
    - Markdown with headings? → StructureAwareChunker
    - Plain text? → Simple paragraph-based chunking

    Example:
        chunker = HybridChunker()
        chunks = await chunker.chunk_file("meeting.mp3")
        # or
        chunks = chunker.chunk_file("business_plan.pdf")
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_percentage: float = 0.15,
        assemblyai_api_key: Optional[str] = None
    ):
        self.max_tokens = max_tokens
        self.overlap_percentage = overlap_percentage

        # Initialize sub-chunkers
        if HAS_ASSEMBLYAI:
            self.transcript_chunker = TranscriptChunker(
                max_tokens=max_tokens,
                api_key=assemblyai_api_key
            )
        else:
            self.transcript_chunker = None

        self.structure_chunker = StructureAwareChunker(
            max_tokens=max_tokens,
            overlap_percentage=overlap_percentage
        )

    async def chunk_file(self, file_path: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Auto-detect file type and chunk appropriately."""

        content_type = self._detect_file_type(file_path)
        base_metadata = metadata or {}
        base_metadata["source_path"] = file_path

        if content_type == ContentType.AUDIO:
            if not self.transcript_chunker:
                raise ImportError("AssemblyAI required for audio processing")
            return await self.transcript_chunker.transcribe_and_chunk(
                file_path,
                base_metadata
            )

        elif content_type == ContentType.PDF:
            return self.structure_chunker.chunk_pdf(file_path, base_metadata)

        elif content_type == ContentType.MARKDOWN:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.structure_chunker.chunk_markdown(text, base_metadata)

        else:  # Plain text
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self._chunk_plain_text(text, base_metadata)

    def _detect_file_type(self, file_path: str) -> ContentType:
        """Detect content type from file extension."""
        ext = file_path.lower().split('.')[-1]

        if ext in ['mp3', 'm4a', 'wav', 'flac', 'ogg']:
            return ContentType.AUDIO
        elif ext == 'pdf':
            return ContentType.PDF
        elif ext in ['md', 'markdown']:
            return ContentType.MARKDOWN
        else:
            return ContentType.PLAIN_TEXT

    def _chunk_plain_text(self, text: str, metadata: Dict) -> List[Chunk]:
        """Simple paragraph-based chunking for plain text."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        current_paragraphs = []
        current_tokens = 0
        overlap_tokens = int(self.max_tokens * self.overlap_percentage)

        for para in paragraphs:
            para_tokens = len(para) // 4  # Approximate

            if current_tokens + para_tokens > self.max_tokens and current_paragraphs:
                # Create chunk with overlap
                chunk_text = '\n\n'.join(current_paragraphs)

                # Add overlap from previous chunk if exists
                if chunks:
                    prev_tokens = chunks[-1].text.split()[-overlap_tokens:]
                    overlap = ' '.join(prev_tokens)
                    chunk_text = f"{overlap} {chunk_text}"

                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    metadata=metadata
                ))

                current_paragraphs = [para]
                current_tokens = para_tokens
            else:
                current_paragraphs.append(para)
                current_tokens += para_tokens

        # Add remaining
        if current_paragraphs:
            chunk_text = '\n\n'.join(current_paragraphs)

            if chunks:
                prev_tokens = chunks[-1].text.split()[-overlap_tokens:]
                overlap = ' '.join(prev_tokens)
                chunk_text = f"{overlap} {chunk_text}"

            chunks.append(Chunk(
                text=chunk_text,
                chunk_index=len(chunks),
                metadata=metadata
            ))

        return chunks


# ============================================================================
# Usage Examples
# ============================================================================

async def example_usage():
    """Example usage of the chunkers."""

    # Example 1: Process audio file with speaker diarization
    print("Example 1: Audio transcript chunking")
    chunker = HybridChunker(assemblyai_api_key="your-api-key")
    chunks = await chunker.chunk_file(
        "meeting_recording.mp3",
        metadata={"meeting_title": "Strategy Review", "date": "2024-12-20"}
    )

    for chunk in chunks[:3]:  # Show first 3
        print(f"Chunk {chunk.chunk_index}")
        print(f"Speaker: {chunk.speaker}")
        print(f"Time: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s")
        print(f"Text: {chunk.text[:100]}...")
        print()

    # Example 2: Process PDF document
    print("\nExample 2: PDF chunking")
    pdf_chunks = chunker.chunk_file(
        "business_plan.pdf",
        metadata={"document_type": "business_plan", "version": "2.0"}
    )

    for chunk in pdf_chunks[:3]:
        print(f"Chunk {chunk.chunk_index}")
        print(f"Section: {chunk.section_path}")
        print(f"Text: {chunk.text[:100]}...")
        print()

    # Example 3: Store in OpenMemory (pseudo-code)
    print("\nExample 3: Store in OpenMemory")
    for chunk in chunks:
        # await add_memories(
        #     text=chunk.text,
        #     vault="WLT",
        #     layer="cognitive",
        #     entity=chunk.speaker,
        #     tags={
        #         "chunk_index": chunk.chunk_index,
        #         "chunk_id": chunk.chunk_id,
        #         **chunk.metadata
        #     }
        # )
        print(f"Stored chunk {chunk.chunk_index} with ID {chunk.chunk_id[:16]}...")


if __name__ == "__main__":
    import asyncio

    print("Chunking Examples Module")
    print("=" * 60)
    print("\nAvailable chunkers:")
    print("- TranscriptChunker (requires: assemblyai)")
    print("- StructureAwareChunker (requires: pymupdf4llm)")
    print("- HybridChunker (auto-detects content type)")
    print("\nRun example_usage() to see examples")

    # Uncomment to run examples
    # asyncio.run(example_usage())
