from langchain_ollama import OllamaEmbeddings

import chat
import config
import store
from ingest import sync_index
from retriever import retrieve, label


def show_progress(done: int, total:int) -> None:

    print(f"  embedded {done}/{total}", end = '\r' if done < total else '\n', flush = True )



def main() -> None:

    embeddings = OllamaEmbeddings(model = config.EMBED_MODEL)

    client = store.connect()


    try:

        col = store.get_collections(client)


        print("...Checking documents...")


        sync_index(col, embeddings, progress = show_progress)


        total = store.count_all(col)

        if total == 0:
            raise SystemExit('Nothing Indexed, Files missing in ./docs')

        print (f'Weaviate holds {total} chunks')


        chain = chat.build_chain()

        print('Ask a question or type exit\n')


        while True:

            q = input('You :').strip()

            if not q:
                continue

            if q.lower() in {'exit', 'quit'}:

                break


            chunks = retrieve(col, embeddings, q)

            if not chunks:

                print("Nothing Matched")


            for c in chunks[:9]:

                print('[retrieved] ', label(c), '|', c['text'][:100].replace('\n',''))

            print('Bot: ', chat.answer(chain, q , chunks), '\n')

    finally:

            client.close()

if __name__ == "__main__":

    main()









