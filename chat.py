import config

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama





PROMPT = ChatPromptTemplate.from_template(
    "You answer questions about a document using only the excerpts below. "
    "The excerpts may come from a novel, so they can contain a character's thoughts, "
    "daydreams, dialogue and flashbacks. Do not present imagined or hoped-for things "
    "as facts. When asked who someone is, combine what the excerpts show about the "
    "person: what they do, say, feel and how others treat them, in 3 to 5 sentences. "
    "Do not rely on a single excerpt. If the excerpts do not contain enough to answer, "
    "say you do not know.\n\n"
    "Excerpts:\n{context}\n\n"
    "Question: {question}"
)



def build_chain(model: str | None = None):

    llm = ChatOllama(model = model or config.CHAT_MODEL, temperature = 0, top_p = 0.9, num_ctx = config.NUM_CTX)

    return PROMPT | llm | StrOutputParser()



def format_context(chunks: list[dict]) -> str:

    return '\n\n'.join(c['text'] for c in chunks)


def answer(chain, question: str, chunks: list[dict]) -> str:

    return chain.invoke ( {"context": format_context(chunks), "question": question})

