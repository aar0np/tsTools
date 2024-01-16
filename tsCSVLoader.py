import cohere  
import os
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter

co = cohere.Client(os.environ["COHERE_API_KEY"])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 0,
    length_function = len,
    is_separator_regex = False,
)

with open("ts_vectors.csv","w") as csv:
	csv.write("id,sentence,ts_vector" + "\n")

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
			#session.execute(insertVector,[texts[index],embeddings[index]])
			csv.write("%s,%s,%s\n" % (str(uuid.uuid4()),texts[index],embeddings[index]))
