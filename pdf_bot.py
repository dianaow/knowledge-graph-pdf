import os

import streamlit as st
from streamlit.logger import get_logger
import streamlit.components.v1 as components

from langchain.callbacks.base import BaseCallbackHandler
#from langchain.graphs import Neo4jGraph
from neo4j import GraphDatabase
from dotenv import load_dotenv
import hashlib
import glob

#import networkx as nx
#from pyvis.network import Network

from chains import (
    load_embedding_model,
    load_llm,
    configure_llm_only_chain,
    configure_qa_structure_rag_chain,
)


from llmsherpa.readers import LayoutPDFReader
llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_reader = LayoutPDFReader(llmsherpa_api_url)

load_dotenv(".env")

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
#llm_name = os.getenv("LLM")
llm_name = 'gpt-3.5'
# Remapping for Langchain Neo4j integration
# os.environ["NEO4J_URL"] = url

logger = get_logger(__name__)

#neo4j_graph = Neo4jGraph(url=url, username=username, password=password, database=database)
embeddings, dimension = load_embedding_model(
    embedding_model_name, config={"ollama_base_url": ollama_base_url}, logger=logger
)
llm = load_llm(llm_name, logger=logger, config={"ollama_base_url": ollama_base_url})


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def chat_input():
    user_input = st.chat_input("What questions can I help you resolve today?")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.caption(f"RAG: {name}")
            stream_handler = StreamHandler(st.empty())

            # Call chain to generate answers
            result = output_function(
                {"question": user_input, "chat_history": []}, callbacks=[stream_handler]
            )["answer"]

            st.session_state[f"user_input"].append(user_input)
            st.session_state[f"generated"].append(result)
            st.session_state[f"rag_mode"].append(name)


def display_chat():
    # Session state
    if "generated" not in st.session_state:
        st.session_state[f"generated"] = []

    if "user_input" not in st.session_state:
        st.session_state[f"user_input"] = []

    if "rag_mode" not in st.session_state:
        st.session_state[f"rag_mode"] = []

    if st.session_state[f"generated"]:
        size = len(st.session_state[f"generated"])
        # Display only the last three exchanges
        for i in range(max(size - 3, 0), size):
            with st.chat_message("user"):
                st.write(st.session_state[f"user_input"][i])

            with st.chat_message("assistant"):
                st.caption(f"RAG: {st.session_state[f'rag_mode'][i]}")
                st.write(st.session_state[f"generated"][i])

        with st.container():
            st.write("&nbsp;")


def mode_select() -> str:
    options = ["Disabled", "Enabled"]
    return st.radio("Select RAG mode", options, horizontal=True)

# def open_sidebar():
#     st.session_state.open_sidebar = True


# def close_sidebar():
#     st.session_state.open_sidebar = False

# st.button(
#     "Show Graph",
#     type="primary",
#     key="show_graph",
#     on_click=open_sidebar,
# )

# def retrieve_graph():
#     query = """
#     MATCH (n)-[r]->(m) RETURN n, r, m
#     """
#     records = neo4j_graph.query(query)
#     return records

# def draw_graph(records):
#     # Create a NetworkX graph
#     G = nx.Graph()

#     # Process records to add nodes and edges
#     for record in records:
#         node_n = record['n']
#         node_m = record['m']
#         edge = record['r']
#         print('NODE', node_n)
#         # Add nodes with attributes
#         G.add_node(node_n['id'], **node_n)
#         G.add_node(node_m['id'], **node_m)
        
#         # Add edge with type
#         G.add_edge(edge[0]['id'], edge[2]['id'], type=edge[1])


#     # Create a PyVis network
#     net = Network(notebook=True)
#     net.from_nx(G)

#     # Save and read graph as HTML file (streamlit can't directly display from PyVis)
#     net.show("graph.html")
#     HtmlFile = open("graph.html", "r", encoding="utf-8")

#     # Load HTML file in Streamlit and display it
#     source_code = HtmlFile.read()
#     components.html(source_code, height=600)

# if not "open_sidebar" in st.session_state:
#     st.session_state.open_sidebar = False
# if st.session_state.open_sidebar:
#     records = retrieve_graph()
#     draw_graph(records)

def initialiseNeo4j():
    cypher_schema = [
        "CREATE CONSTRAINT sectionKey IF NOT EXISTS FOR (c:Section) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT chunkKey IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT documentKey IF NOT EXISTS FOR (c:Document) REQUIRE (c.url_hash) IS UNIQUE;",
        "CREATE CONSTRAINT tableKey IF NOT EXISTS FOR (c:Table) REQUIRE (c.key) IS UNIQUE;",
        "CALL db.index.vector.createNodeIndex('chunkVectorIndex', 'Embedding', 'value', 1536, 'COSINE');"
    ]

    driver = GraphDatabase.driver(url, database=database, auth=(username, password))

    with driver.session() as session:
        for cypher in cypher_schema:
            session.run("DROP INDEX chunkVectorIndex IF EXISTS")
            session.run(cypher)
    driver.close()

def ingestDocumentNeo4j(doc, doc_location):

    cypher_pool = [
        # 0 - Document
        "MERGE (d:Document {url_hash: $doc_url_hash_val}) ON CREATE SET d.url = $doc_url_val RETURN d;",  
        # 1 - Section
        "MERGE (p:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) ON CREATE SET p.page_idx = $page_idx_val, p.title_hash = $title_hash_val, p.block_idx = $block_idx_val, p.title = $title_val, p.tag = $tag_val, p.level = $level_val RETURN p;",
        # 2 - Link Section with the Document
        "MATCH (d:Document {url_hash: $doc_url_hash_val}) MATCH (s:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (d)<-[:HAS_DOCUMENT]-(s);",
        # 3 - Link Section with a parent section
        "MATCH (s1:Section {key: $doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_title_hash_val}) MATCH (s2:Section {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$title_hash_val}) MERGE (s1)<-[:UNDER_SECTION]-(s2);",
        # 4 - Chunk
        "MERGE (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) ON CREATE SET c.sentences = $sentences_val, c.sentences_hash = $sentences_hash_val, c.block_idx = $block_idx_val, c.page_idx = $page_idx_val, c.tag = $tag_val, c.level = $level_val RETURN c;",
        # 5 - Link Chunk to Section
        "MATCH (c:Chunk {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$sentences_hash_val}) MATCH (s:Section {key:$doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(c);",
        # 6 - Table
        "MERGE (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) ON CREATE SET t.name = $name_val, t.doc_url_hash = $doc_url_hash_val, t.block_idx = $block_idx_val, t.page_idx = $page_idx_val, t.html = $html_val, t.rows = $rows_val RETURN t;",
        # 7 - Link Table to Section
        "MATCH (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Section {key: $doc_url_hash_val+'|'+$parent_block_idx_val+'|'+$parent_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);",
        # 8 - Link Table to Document if no parent section
        "MATCH (t:Table {key: $doc_url_hash_val+'|'+$block_idx_val+'|'+$name_val}) MATCH (s:Document {url_hash: $doc_url_hash_val}) MERGE (s)<-[:HAS_PARENT]-(t);"
    ]

    driver = GraphDatabase.driver(url, database=database, auth=(username, password))

    with driver.session() as session:
        cypher = ""

        # 1 - Create Document node
        doc_url_val = doc_location
        doc_url_hash_val = hashlib.md5(doc_url_val.encode("utf-8")).hexdigest()

        cypher = cypher_pool[0]
        session.run(cypher, doc_url_hash_val=doc_url_hash_val, doc_url_val=doc_url_val)

        # 2 - Create Section nodes
        
        countSection = 0
        for sec in doc.sections():
            sec_title_val = sec.title
            sec_title_hash_val = hashlib.md5(sec_title_val.encode("utf-8")).hexdigest()
            sec_tag_val = sec.tag
            sec_level_val = sec.level
            sec_page_idx_val = sec.page_idx
            sec_block_idx_val = sec.block_idx

            # MERGE section node
            if not sec_tag_val == 'table':
                cypher = cypher_pool[1]
                session.run(cypher, page_idx_val=sec_page_idx_val
                                , title_hash_val=sec_title_hash_val
                                , title_val=sec_title_val
                                , tag_val=sec_tag_val
                                , level_val=sec_level_val
                                , block_idx_val=sec_block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )

                # Link Section with a parent section or Document

                sec_parent_val = str(sec.parent.to_text())

                if sec_parent_val == "None":    # use Document as parent

                    cypher = cypher_pool[2]
                    session.run(cypher, page_idx_val=sec_page_idx_val
                                    , title_hash_val=sec_title_hash_val
                                    , doc_url_hash_val=doc_url_hash_val
                                    , block_idx_val=sec_block_idx_val
                                )

                else:   # use parent section
                    sec_parent_title_hash_val = hashlib.md5(sec_parent_val.encode("utf-8")).hexdigest()
                    sec_parent_page_idx_val = sec.parent.page_idx
                    sec_parent_block_idx_val = sec.parent.block_idx

                    cypher = cypher_pool[3]
                    session.run(cypher, page_idx_val=sec_page_idx_val
                                    , title_hash_val=sec_title_hash_val
                                    , block_idx_val=sec_block_idx_val
                                    , parent_page_idx_val=sec_parent_page_idx_val
                                    , parent_title_hash_val=sec_parent_title_hash_val
                                    , parent_block_idx_val=sec_parent_block_idx_val
                                    , doc_url_hash_val=doc_url_hash_val
                                )
            # **** if sec_parent_val == "None":    

            countSection += 1
        # **** for sec in doc.sections():

        
        # ------- Continue within the blocks -------
        # 3 - Create Chunk nodes from chunks
            
        countChunk = 0
        for chk in doc.chunks():

            chunk_block_idx_val = chk.block_idx
            chunk_page_idx_val = chk.page_idx
            chunk_tag_val = chk.tag
            chunk_level_val = chk.level
            chunk_sentences = "\n".join(chk.sentences)

            # MERGE Chunk node
            if not chunk_tag_val == 'table':
                chunk_sentences_hash_val = hashlib.md5(chunk_sentences.encode("utf-8")).hexdigest()

                # MERGE chunk node
                cypher = cypher_pool[4]
                session.run(cypher, sentences_hash_val=chunk_sentences_hash_val
                                , sentences_val=chunk_sentences
                                , block_idx_val=chunk_block_idx_val
                                , page_idx_val=chunk_page_idx_val
                                , tag_val=chunk_tag_val
                                , level_val=chunk_level_val
                                , doc_url_hash_val=doc_url_hash_val
                            )
            
                # Link chunk with a section
                # Chunk always has a parent section 

                chk_parent_val = str(chk.parent.to_text())
                
                if not chk_parent_val == "None":
                    chk_parent_hash_val = hashlib.md5(chk_parent_val.encode("utf-8")).hexdigest()
                    chk_parent_page_idx_val = chk.parent.page_idx
                    chk_parent_block_idx_val = chk.parent.block_idx

                    cypher = cypher_pool[5]
                    session.run(cypher, sentences_hash_val=chunk_sentences_hash_val
                                    , block_idx_val=chunk_block_idx_val
                                    , parent_hash_val=chk_parent_hash_val
                                    , parent_block_idx_val=chk_parent_block_idx_val
                                    , doc_url_hash_val=doc_url_hash_val
                                )
                    
                # Link sentence 
                #   >> TO DO for smaller token length

                countChunk += 1
        # **** for chk in doc.chunks(): 

        # 4 - Create Table nodes

        countTable = 0
        for tb in doc.tables():
            page_idx_val = tb.page_idx
            block_idx_val = tb.block_idx
            name_val = 'block#' + str(block_idx_val) + '_' + tb.name
            html_val = tb.to_html()
            rows_val = len(tb.rows)

            # MERGE table node

            cypher = cypher_pool[6]
            session.run(cypher, block_idx_val=block_idx_val
                            , page_idx_val=page_idx_val
                            , name_val=name_val
                            , html_val=html_val
                            , rows_val=rows_val
                            , doc_url_hash_val=doc_url_hash_val
                        )
            
            # Link table with a section
            # Table always has a parent section 

            table_parent_val = str(tb.parent.to_text())
            
            if not table_parent_val == "None":
                table_parent_hash_val = hashlib.md5(table_parent_val.encode("utf-8")).hexdigest()
                table_parent_page_idx_val = tb.parent.page_idx
                table_parent_block_idx_val = tb.parent.block_idx

                cypher = cypher_pool[7]
                session.run(cypher, name_val=name_val
                                , block_idx_val=block_idx_val
                                , parent_page_idx_val=table_parent_page_idx_val
                                , parent_hash_val=table_parent_hash_val
                                , parent_block_idx_val=table_parent_block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )

            else:   # link table to Document
                cypher = cypher_pool[8]
                session.run(cypher, name_val=name_val
                                , block_idx_val=block_idx_val
                                , doc_url_hash_val=doc_url_hash_val
                            )
            countTable += 1

        # **** for tb in doc.tables():
        
        print(f'\'{doc_url_val}\' Done! Summary: ')
        print('#Sections: ' + str(countSection))
        print('#Chunks: ' + str(countChunk))
        print('#Tables: ' + str(countTable))

    driver.close()

def LoadEmbedding(label, property):
    driver = GraphDatabase.driver(url, database=database, auth=(username, password))

    with driver.session() as session:
        # get chunks in document, together with their section titles
        result = session.run(f"MATCH (ch:{label}) -[:HAS_PARENT]-> (s:Section) RETURN id(ch) AS id, s.title + ' >> ' + ch.{property} AS text")
        # call OpenAI embedding API to generate embeddings for each proporty of node
        # for each node, update the embedding property
        count = 0
        for record in result:
            id = record["id"]
            text = record["text"]
            
            # For better performance, text can be batched
            embedding = embeddings.embed_query(text)
            
            # key property of Embedding node differentiates different embeddings
            cypher = "CREATE (e:Embedding) SET e.key=$key, e.value=$embedding"
            cypher = cypher + " WITH e MATCH (n) WHERE id(n) = $id CREATE (n) -[:HAS_EMBEDDING]-> (e)"
            session.run(cypher,key=property, embedding=embedding, id=id )
            count = count + 1

        session.close()
        
        print("Processed " + str(count) + " " + label + " nodes for property @" + property + ".")
        return count

initialiseNeo4j()

uploaded_files = st.file_uploader("Choose a file", type=["pdf"], accept_multiple_files=True)
st.write("Upload a single or multiple files in either PDF")

file_location = './uploads'
os.makedirs(file_location, exist_ok=True)

if len(uploaded_files) is not 0:
    if st.button('Generate Graph'):

        for pdf_file in uploaded_files:
            file_type = pdf_file.type
            if file_type == "application/pdf":
                # Save the uploaded file to the uploads directory
                file_path = os.path.join(file_location, pdf_file.name)
                with open(file_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

        pdf_files = glob.glob(file_location + '/*.pdf')

        for pdf_file in pdf_files:
            doc = pdf_reader.read_pdf(pdf_file)

            ingestDocumentNeo4j(doc, pdf_file)

            LoadEmbedding("Chunk", "sentences")
            LoadEmbedding("Table", "name")

# llm_chain: LLM only response
llm_chain = configure_llm_only_chain(llm)

# rag_chain: KG augmented response
rag_chain = configure_qa_structure_rag_chain(
    llm, embeddings, embeddings_store_url=url, username=username, password=password
)

name = mode_select()

if name == "LLM only" or name == "Disabled":
    output_function = llm_chain
elif name == "Vector + Graph" or name == "Enabled":
    output_function = rag_chain

display_chat()
chat_input()
