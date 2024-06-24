## Question / Answer with a local PDF
UI: http://localhost:8503  
DB client: http://localhost:7474

This application lets you load multiple PDF documents to construct a Knowledge Graph and embed it into Neo4j so you can ask questions about its contents and have the LLM answer them using vector similarity search and graph traversal.

This repository used Neo4j's GenAI stack as a starting point.
Learn more about the details in the [technical blog post](https://neo4j.com/developer-blog/genai-app-how-to-build/).

# Configure

Create a `.env` file from the environment template file `env.example`

Available variables:
| Variable Name          | Default value                      | Description                                                             |
|------------------------|------------------------------------|-------------------------------------------------------------------------|
| OLLAMA_BASE_URL        | http://host.docker.internal:11434  | REQUIRED - URL to Ollama LLM API                                        |   
| NEO4J_URI              | neo4j://database:7687              | REQUIRED - URL to Neo4j database                                        |
| NEO4J_USERNAME         | neo4j                              | REQUIRED - Username for Neo4j database                                  |
| NEO4J_PASSWORD         | password                           | REQUIRED - Password for Neo4j database                                  |
| LLM                    | llama2                             | REQUIRED - Can be any Ollama model tag, or gpt-4 or gpt-3.5 or claudev2 |
| EMBEDDING_MODEL        | sentence_transformer               | REQUIRED - Can be sentence_transformer, openai, aws, ollama or google-genai-embedding-001|
| AWS_ACCESS_KEY_ID      |                                    | REQUIRED - Only if LLM=claudev2 or embedding_model=aws                  |
| AWS_SECRET_ACCESS_KEY  |                                    | REQUIRED - Only if LLM=claudev2 or embedding_model=aws                  |
| AWS_DEFAULT_REGION     |                                    | REQUIRED - Only if LLM=claudev2 or embedding_model=aws                  |
| OPENAI_API_KEY         |                                    | REQUIRED - Only if LLM=gpt-4 or LLM=gpt-3.5 or embedding_model=openai   |
| GOOGLE_API_KEY         |                                    | REQUIRED - Only required when using GoogleGenai LLM or embedding model google-genai-embedding-001|


## LLM Configuration
MacOS and Linux users can use any LLM that's available via Ollama. Check the "tags" section under the model page you want to use on https://ollama.ai/library and write the tag for the value of the environment variable `LLM=` in the `.env` file.
All platforms can use GPT-3.5-turbo and GPT-4 (bring your own API keys for OpenAI models).

No need to install Ollama manually, it will run in a container as part of the stack when running with the Linux profile: run `docker compose --profile linux up`.
Make sure to set the `OLLAMA_BASE_URL=http://llm:11434` in the `.env` file when using Ollama docker container.
To use the Linux-GPU profile: run `docker compose --profile linux-gpu up`. Also change `OLLAMA_BASE_URL=http://llm-gpu:11434` in the `.env` file.

However, if you would like to run Ollama locally on you machine, you may remove the 'llm' and 'llm-gpu' services from the docker-compose.yml file. Alternatively, you may install [Ollama](https://ollama.ai) and start it before running `docker compose up` using `ollama serve` in a separate terminal.
.
# Develop

**To start everything**
```
docker compose up
```
If changes to build scripts have been made, **rebuild**.
```
docker compose up --build
```

To enter **watch mode** (auto rebuild on file changes).
First start everything, then in new terminal:
```
docker compose watch
```

**Shutdown**
If health check fails or containers don't start up as expected, shutdown
completely to start up again.
```
docker compose down
```