import json
from typing import Callable, Optional
from pathlib import Path
import hashlib

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

import config 
import store


SUPPORTED = {'.pdf', '.txt', '.md'}


def file_hash(path:Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_file(path:Path):

    if path.suffix.lower() == '.pdf':
        return PyPDFLoader(str(path)).load()
    return TextLoader(str(path), encoding = 'utf-8').load()



def chunk_documents(docs):

    splitter = RecursiveCharacterTextSplitter(chunk_size = config.CHUNK_SIZE, chunk_overlap = config.OVERLAP)

    #splitter = SemanticChunker(embeddings, breakpoint_threshold_type= "percentile", breakpoint_threshold_amount= 90 )

    return splitter.split_documents(docs)



def add_file(col, embeddings, path:Path, progress: Optional[Callable[[int,int],None]] = None) -> int:

    docs = load_file(path)

    if not any(d.page_content.strip() for d in docs):
        return 0

    chunks = chunk_documents(docs)

    for i in range(0, len(chunks), config.EMBED_BATCH):
        batch = chunks[i:i + config.EMBED_BATCH]
        vectors = embeddings.embed_documents([c.page_content for c in batch])

        rows = [
            {

                'text' : c.page_content,
                'source': path.name,
                'page' : int(c.metadata.get('page', -1)),
                'chunk' : i+j , 
                'vector' : v,

            }for j,(c,v) in enumerate (zip(batch,vectors))
        ]   


        store.insert_chunks(col, rows)

        if progress:
             progress(min(i + config.EMBED_BATCH, len(chunks)) , len (chunks))
        
    return len(chunks)


def sync_index(col, embeddings, docs_dir: Optional[Path] = None, log: Callable[[str], None] = print, 
               manifest_path : Optional[Path] = None, progress: Optional[Callable[[int,int], None]] = None  ) -> dict:


    docs_dir = docs_dir or config.DOCS_DIR

    manifest_path = manifest_path or config.MANIFEST

    manifest_path.parent.mkdir(parents = True, exist_ok = True)



    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    files = { p.name: p for p in sorted(docs_dir.glob("*"))

             if p.suffix.lower() in SUPPORTED
        }

    report = { "added" : [], "unreadable" : [], "removed" : [], "skipped" : [] }



    for name in sorted(set(manifest) - set(files)):

        log(f"...REMOVING {name}...")
        store.remove_file(name)

        manifest.pop(name)

        manifest_path.write_text(json.dumps(manifest, indent = 2))

        report['removed'].append(name)



    for name,path in files.items():

        h = file_hash(path)

        if manifest.get(name) == h and store.count_for(col, name) > 0:

            report['skipped'].append(name)

            continue

        store.remove_file(col, name)

        log(f'...Indexing {name}... ')



        if add_file(col, embeddings, path, progress) > 0:

            manifest[name] = h

            manifest_path.write_text(json.dumps(manifest, indent = 2))

            report['added'].append(name)

            log(f'...Added {name}...')

        else:
            log(f"...Skipping {name} as it is unreadable. OCR required if PDF")

            report['unreadable'].append(name)


    return report






