from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
limit = os.getenv("CONCURRENT_TASK_LIMIT")
# optional debug
# print("API KEY:", os.getenv("OPENAI_API_KEY"))
print("LIMIT:", os.getenv("CONCURRENT_TASK_LIMIT"))

from fast_graphrag import GraphRAG

DOMAIN = """
Analyze the provided scientific and historical text to identify key discoveries, 
scientific concepts, technologies, and their relationships. 
Focus on how discoveries lead to scientific principles, how these principles enable 
technologies, and how these technologies contribute to medical applications, especially medical imaging. 
Capture cause-and-effect relationships and progression over time.
"""

EXAMPLE_QUERIES = [
    "What did Marie Curie discover?",
    "How did the discovery of radioactivity influence later technologies?",
    "What is the connection between radium and medical imaging?",
    "How did early radiation research contribute to X-ray imaging?",
    "What scientific principles enabled the development of medical imaging technologies?",
    "Explain the progression from Curie's discoveries to modern medical imaging"
]

ENTITY_TYPES = [
    "Scientist",
    "Discovery",
    "ScientificConcept",
    "Element",
    "Technology",
    "MedicalApplication",
    "Event"
]

grag = GraphRAG(
    working_dir="./marie_curie_graph",
    domain=DOMAIN,
    example_queries="\n".join(EXAMPLE_QUERIES),
    entity_types=ENTITY_TYPES
)

# Insert is the one doing the calls to :
# 1. Process text (LLM calls moneeyy)
# 2. Build graph
# 3. SAVE results into working_dir
with open("curie_things.txt", "r", encoding="utf-8") as f:
    grag.insert(f.read())

# Query :
# load graph -> run retrieval -> call LLM for answer
# NEVER rebuilds the graph
print(grag.query("What discoveries by Marie Curie led to later advances in medical imaging?").response)