import os
import json
import pandas as pd
from tqdm import tqdm
import tempfile

import streamlit as st
import streamlit.components.v1 as components
from streamlit.logger import get_logger

import networkx as nx
from pyvis.network import Network

from langchain.chains import RetrievalQA
from langchain_community.graphs import Neo4jGraph
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain.prompts import ChatPromptTemplate
from langchain.chains.openai_functions import (
    create_structured_output_chain
)
from langchain_openai import ChatOpenAI
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel

from chains import (
    load_embedding_model,
    load_llm,
)

class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    edges: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )

def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
      return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties

def map_to_base_node(node: Node) -> BaseNode:
    """Map the KnowledgeGraph Node to the base Node."""
    properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )

def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )
  

# load api key lib
from dotenv import load_dotenv

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)


embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

#llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))


# Function to process PDF files
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    pages = loader.load_and_split()
    return pages

# Function to process CSV files
def process_csv(file):
    df = pd.read_csv(file)
    return df

# Function to process JSON files
def process_json(file):
    file_content = file.read().decode("utf-8")
    data = json.load(file_content)
    return data

def main():
    uploaded_files = st.file_uploader("Choose a file", type=["pdf", "csv", "json"], accept_multiple_files=True)
    st.write("Upload a single or multiple files in either PDF, CSV or JSON format to generate a graph. For CSV format, it is required to upload separate CSV files of nodes and edges. For JSON format, ensure there are both node and edge data objects in a single JSON file.")

    if len(uploaded_files) is not 0:
        if st.button('Generate Graph'):
            for uploaded_file in uploaded_files:
                file_type = uploaded_file.type

                if file_type == "application/pdf":
                    raw_documents = process_pdf(uploaded_file)

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500, chunk_overlap=200, length_function=len
                    )
                    documents = text_splitter.split_documents(raw_documents)
                    
                    for i, d in tqdm(enumerate(documents), total=len(documents)):
                        extract_and_store_graph(d)

                elif file_type == "text/csv":
                    return
                elif file_type == "application/json":
                    json_data = process_json(uploaded_file)
                    df_nodes = pd.DataFrame(json_data['nodes'])
                    df_edges = pd.DataFrame(json_data['edges'])
                    insert_data(df_nodes, df_edges)
                    #insert_query_data(json_data)
                else:
                    st.write("Unsupported file type.")


            if uploaded_files[0] == "text/csv":
                df_nodes = process_csv(uploaded_files[0])
                df_edges = process_csv(uploaded_files[1])
                insert_data(df_nodes, df_edges)
        
            records = retrieve_graph()
            draw_graph(records)

def get_extraction_chain(
      allowed_nodes: Optional[List[str]] = None,
      allowed_rels: Optional[List[str]] = None
      ):
    
      prompt = ChatPromptTemplate.from_messages(
          [(
            "system",
            f"""
            Extract entities and relationships from the following text and format them as JSON with keys 'nodes' and 'edges'.
            {'- Allowed Node Labels:' + ", ".join(allowed_nodes) if allowed_nodes else ""}
            {'- Allowed Relationship Types:' + ", ".join(allowed_rels) if allowed_rels else ""}
            """),
              ("human", "Use the given format to extract information from the following input: {input}"),
              ("human", "Tip: Make sure to answer in the correct format and JSON compliant"),
          ])
      return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)

def extract_and_store_graph(
    document: Document,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]]=None) -> None:

    extract_chain = get_extraction_chain(nodes, rels)
    data = extract_chain.invoke(document.page_content)['function']

    # Construct a graph document
    graph_document = GraphDocument(
        nodes = [map_to_base_node(node) for node in data.nodes],
        relationships = [map_to_base_relationship(rel) for rel in data.edges],
        source = document
    )

    # Store information into a graph
    neo4j_graph.add_graph_documents([graph_document])

def insert_data(df_nodes, df_edges):

    def create_properties(row):
        return {key: value for key, value in row.items() if key != 'label'}

    for index, row in df_nodes.iterrows():
        row['properties'] = row.apply(create_properties, axis=1)
    
    ingest_graph_into_neo4j(df_nodes[['label', 'properties']], df_edges)

def ingest_graph_into_neo4j(node_data_objects, relationship_data_objects):
    # Ingest nodes
    for node_data in node_data_objects:
        label = ':'.join(node_data['label'])
        properties = ', '.join(f"{key}: ${key}" for key in node_data if key not in [])
        neo4j_graph.query(f"MERGE (n:{label} {{{properties}}})", **node_data['properties'])

    # Ingest relationships
    for rel_data in relationship_data_objects:
        neo4j_graph.query("""
            MATCH (source {elementId: $startNodeElementId})
            MATCH (target {elementId: $endNodeElementId})
            MERGE (source)-[r:{type}]->(target)
        """, startNodeElementId=rel_data['startNodeElementId'], endNodeElementId=rel_data['endNodeElementId'], type=rel_data['type'])


def insert_query_data(data: dict) -> None:
    # Cypher, the query language of Neo4j, is used to import the data
    # https://neo4j.com/docs/getting-started/cypher-intro/
    # https://neo4j.com/docs/cypher-cheat-sheet/5/auradb-enterprise/
    import_query = data['query']
    neo4j_graph.query(import_query)

def retrieve_graph():
    query = """
    MATCH (n)-[r]->(m) RETURN n, r, m
    """
    records = neo4j_graph.query(query)
    return records

def draw_graph(records):
    # Create a NetworkX graph
    G = nx.Graph()

    # Process records to add nodes and edges
    for record in records:
        node_n = record['n']
        node_m = record['m']
        edge = record['r']

        # Add nodes with attributes
        G.add_node(node_n['id'], **node_n)
        G.add_node(node_m['id'], **node_m)
        
        # Add edge with type
        G.add_edge(edge[0]['id'], edge[2]['id'], type=edge[1])


    # Create a PyVis network
    net = Network(notebook=True)
    net.from_nx(G)

    # Save and read graph as HTML file (streamlit can't directly display from PyVis)
    net.show("graph.html")
    HtmlFile = open("graph.html", "r", encoding="utf-8")

    # Load HTML file in Streamlit and display it
    source_code = HtmlFile.read()
    components.html(source_code, height=600)

if __name__ == "__main__":
    main()
