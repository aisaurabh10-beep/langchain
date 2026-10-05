from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

client = genai.Client()

texts = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?",
]

result = client.models.embed_content(
    model="gemini-embedding-001",      # which embedding model to call | e.g. 'gemini-embedding-001', 'text-embedding-004'
    contents=texts,                    # text(s) to embed | a str, or a list of str/Part/Content (as in `texts` above)
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY",   # how the embedding will be used, tunes the vector for that purpose | 'SEMANTIC_SIMILARITY', 'RETRIEVAL_QUERY', 'RETRIEVAL_DOCUMENT', 'CLASSIFICATION', 'CLUSTERING', 'QUESTION_ANSWERING', 'FACT_VERIFICATION', 'CODE_RETRIEVAL_QUERY'
        title=None,                        # optional title for the text, only used with task_type='RETRIEVAL_DOCUMENT' | None or any string
        output_dimensionality=None,        # truncate/resize the output embedding vector | None (model default, e.g. 3072) or a positive int (e.g. 768, 1536)
        mime_type=None,                    # MIME type of the input content, for non-plain-text input | None (defaults to 'text/plain') or a valid MIME type string
        auto_truncate=None,                # whether to silently truncate input that exceeds the token limit | None/False (raise error) or True (truncate)
        document_ocr=None,                 # run OCR on document/image input before embedding | None or True/False
        audio_track_extraction=None,       # extract and embed the audio track from video/audio input | None or True/False
        http_options=None,                 # low-level HTTP client overrides (timeouts, headers, base_url) | None or a types.HttpOptions instance
    )
)
print(result.embeddings)