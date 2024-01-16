import cohere  
import os

from dbConnection import get_session
from langchain.text_splitter import RecursiveCharacterTextSplitter

# build CQL statements
insertVectorCQL = "INSERT INTO taylor_swift_vectors (id, sentence, ts_vector) VALUES (UUID(),?,?)"
session = get_session()
insertVector = session.prepare(insertVectorCQL)
# clear out the table
session.execute("TRUNCATE TABLE taylor_swift_vectors")


co = cohere.Client(os.environ["COHERE_API_KEY"])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = False,
)

# iterate through all of the Taylor Swift text files
for counter in range(1,5):
	textfile = str(counter) + ".txt"

	with open(textfile) as ft:
		doc = ft.read()

	texts = text_splitter.split_text(doc)

	embeddings = co.embed(
		texts=texts,
		model="embed-english-light-v3.0",
		input_type="search_document").embeddings

	#for embedding in embeddings:
	for index in range(0,len(embeddings)):
		session.execute(insertVector,[texts[index],embeddings[index]])
