import cohere 
import os

userInput = ""

co = cohere.Client(os.environ["COHERE_API_KEY"])

while userInput != "exit":
    userInput = input("Next question? ")

    #response = co.chat(message=userInput)
    #print(response)

    embedding = co.embed(
    texts=userInput.splitlines(),
    model="embed-english-light-v3.0",
    input_type="search_document").embeddings

    print(embedding)