# Beginner Project Documentation for SwarAI Backend

This document explains the project in beginner-friendly language.

The order follows your request:

1. System requirements
2. Environment
3. Models
4. Libraries
5. Other important project parts

The goal is simple: a new person should be able to open this file and understand what the project needs, how it runs, what each main term means, and what each file is doing.

## 1. System Requirements

### 1.1 Quick List

- A computer running Windows, Linux, or a Docker-capable environment
- Python 3.10
- `pip` for installing Python packages
- A Python virtual environment such as `.venv`
- Internet access for first-time model download from Hugging Face
- A Supabase project for database and file storage features
- Enough RAM, CPU, and disk space for local AI model loading
- Optional Docker if you want container-based deployment
- Optional Git and VS Code for development convenience

### 1.2 Detailed Explanation Of Each Term

- Operating System:
  An operating system is the main software that runs your computer, such as Windows or Linux. This project is being worked on in a Windows environment right now, but the `Dockerfile` shows that it can also run inside a Linux container.

- Python 3.10:
  Python is the programming language used by this backend. Version 3.10 matters because the project and Docker setup are written around that version. If you use a very different Python version, some packages may fail to install or behave differently.

- `pip`:
  `pip` is Python's package installer. It downloads and installs external libraries listed in `requirements.txt`.

- Virtual Environment:
  A virtual environment is a private Python package area for one project. In this project, the common folder name is `.venv`. It helps prevent conflicts with packages installed for other projects on the same computer.

- Internet Access:
  The first time the backend loads the local LLM, it uses Hugging Face download settings such as `LLAMA_REPO` and `LLAMA_FILENAME`. That means the machine usually needs internet access at least once to download the model file.

- Supabase:
  Supabase is the cloud backend used here for storing users, chat sessions, chat messages, and uploaded files. If Supabase is not configured, login, registration, chat history, and file upload features will not work correctly.

- RAM:
  RAM means working memory. AI models, embeddings, and the FAISS index are loaded into memory while the app runs. More RAM makes local model inference much more stable.

- CPU:
  CPU means the main processor of the computer. This project uses local inference with `llama-cpp-python`, so CPU speed and number of cores directly affect response speed if GPU offload is not used.

- Disk Space:
  Disk space is permanent storage on your machine. You need space for Python packages, the virtual environment, downloaded model files, the FAISS index, and uploaded/generated files. The selected LLM file may be several gigabytes depending on the model you configure.

- Docker:
  Docker is a tool that packages the app and its dependencies into a container. A container is an isolated runtime environment that helps the app run more consistently across different machines.

- Git:
  Git is a version control system. It is not required to run the app, but it is useful for tracking code changes.

- VS Code:
  VS Code is a code editor. It is optional, but the project already contains a small `.vscode/settings.json` file, so it is clearly part of the current development workflow.

### 1.3 Practical Machine Recommendation

- Minimum practical setup:
  Python 3.10, internet access, Supabase credentials, and enough RAM to load your chosen LLM.

- Better setup for smoother local AI work:
  Multi-core CPU, 16 GB or more RAM, enough free disk space for model files, and optionally GPU support if your selected `llama-cpp-python` build and model settings use it.

## 2. Environment

### 2.1 Quick List

- `.env`
- `.venv`
- `data/`
- `faiss_db/`
- `questions.json`
- Startup command: `uvicorn main:app --host 0.0.0.0 --port 7860 --reload`
- Runtime variables for Supabase, authentication, models, PDF extraction, and streaming

### 2.2 What "Environment" Means In This Project

- Environment:
  In programming, "environment" means the settings, secret values, folders, and runtime rules that decide how the program behaves on a specific machine.

- `.env` file:
  This is the place where secret keys and runtime settings usually live. The code reads many values from environment variables, and `config.py` loads them by calling `load_dotenv()`.

- `.venv` folder:
  This is the local Python package environment for this project.

- `data/` folder:
  This folder stores source knowledge documents used to build the FAISS search index.

- `faiss_db/` folder:
  This folder stores the vector index and metadata that help the chatbot retrieve relevant text before asking the LLM to answer.

- `questions.json`:
  This file acts like a flow definition file. It tells the requirement-collection chatbot what question comes next, what options are valid, and when to move to summary or submission.

### 2.3 Beginner Terms You Need Before Reading The Variables

- CORS:
  CORS stands for Cross-Origin Resource Sharing. It is a browser security rule. `CORS_ORIGINS` decides which frontend websites are allowed to call this backend.

- JWT:
  JWT stands for JSON Web Token. It is a signed token used to prove a user is logged in.

- Token:
  In AI settings, a token is a small piece of text the model processes internally. It is not the same thing as a login token.

- Context Window:
  This is how much text the model can "see" at one time during generation.

- Batch:
  A batch setting controls how much model work is processed at a time. Bigger values can improve speed but use more memory.

- Thread:
  A thread is a unit of CPU work. More threads can help use multiple CPU cores.

- GPU Layers:
  This setting controls how much of the model is moved from CPU work to GPU work when supported.

- Warmup:
  Warmup means sending a small test request at startup so the first real user request feels faster.

- Model Artifact:
  A model artifact is the actual downloadable model file, usually a `.gguf` file for `llama-cpp-python`.

### 2.4 Supported Environment Variables

#### App And Security Variables

### Quick List

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `SECRET_KEY`
- `ALGORITHM`
- `ACCESS_TOKEN_EXPIRE_MINUTES`
- `CORS_ORIGINS`

### Detailed Explanation

- `SUPABASE_URL`:
  The web address of your Supabase project. The backend uses it to connect to the database and storage bucket.

- `SUPABASE_KEY`:
  The API key used to access Supabase. Without it, the code cannot create the Supabase client.

- `SECRET_KEY`:
  The secret string used to sign JWT login tokens. This should be long, private, and different for each real deployment.

- `ALGORITHM`:
  The signing algorithm used by JWT. The default in this project is `HS256`.

- `ACCESS_TOKEN_EXPIRE_MINUTES`:
  How long a login token stays valid before expiring.

- `CORS_ORIGINS`:
  Which frontend origins are allowed to call this backend from a browser. If this is set too loosely, security can be weaker. If it is set too strictly, the frontend may fail to connect.

#### Retrieval And Main LLM Variables

### Quick List

- `FAISS_DIR`
- `MAX_CONTEXT_TOKENS`
- `LLM_CONTEXT_WINDOW`
- `LLAMA_REPO`
- `LLAMA_FILENAME`
- `LLM_N_CTX`
- `LLM_N_BATCH`
- `LLM_N_THREADS`
- `LLM_N_THREADS_BATCH`
- `LLM_N_GPU_LAYERS`
- `LLM_CHAT_FORMAT`
- `LLM_OFFLOAD_KQV`
- `LLM_FLASH_ATTN`
- `LLM_WARMUP_ON_STARTUP`

### Detailed Explanation

- `FAISS_DIR`:
  The folder where the FAISS vector index and its metadata are stored.

- `MAX_CONTEXT_TOKENS`:
  A project-wide token limit setting loaded in `config.py`. It is part of the configuration layer even though the main answer-generation code currently trims retrieved context by text length before prompting.

- `LLM_CONTEXT_WINDOW`:
  A general context-window setting loaded in `config.py`. The actual llama runtime context is controlled more directly by `LLM_N_CTX` in `backend.py`.

- `LLAMA_REPO`:
  The Hugging Face repository that contains the main chat model file.

- `LLAMA_FILENAME`:
  The exact filename to download from that repository.

- `LLM_N_CTX`:
  The number of tokens the main LLM can keep in memory while generating.

- `LLM_N_BATCH`:
  The inference batch size for the main LLM.

- `LLM_N_THREADS`:
  How many CPU threads the model can use during normal inference.

- `LLM_N_THREADS_BATCH`:
  How many CPU threads are used for batch processing work.

- `LLM_N_GPU_LAYERS`:
  How many model layers should be offloaded to the GPU, if supported.

- `LLM_CHAT_FORMAT`:
  The chat formatting style expected by the model. Some GGUF models expect formats such as `chatml`.

- `LLM_OFFLOAD_KQV`:
  An advanced llama runtime flag related to moving some internal attention work off the CPU path.

- `LLM_FLASH_ATTN`:
  Another advanced optimization flag that may improve performance depending on the build and hardware.

- `LLM_WARMUP_ON_STARTUP`:
  If enabled, the server sends a tiny test prompt during startup to reduce the delay of the first user request.

#### PDF And Requirement Flow Variables

### Quick List

- `FLOW_FILE`
- `PDF_MAX_SIZE_MB`
- `PDF_MAX_PAGES`
- `PDF_PRELOAD_ON_STARTUP`
- `PDF_EXTRACTOR_REPO`
- `PDF_EXTRACTOR_FILENAME`
- `PDF_EXTRACTOR_CHAT_FORMAT`
- `PDF_EXTRACTOR_N_CTX`
- `PDF_EXTRACTOR_N_BATCH`
- `PDF_EXTRACTOR_N_THREADS`
- `PDF_EXTRACTOR_N_THREADS_BATCH`
- `PDF_EXTRACTOR_N_GPU_LAYERS`
- `PDF_EXTRACTOR_OFFLOAD_KQV`
- `PDF_EXTRACTOR_FLASH_ATTN`
- `PDF_QUESTION_BATCH_SIZE`
- `PDF_EXTRACT_CHUNK_LEN`
- `PDF_EXTRACT_MAX_TOKENS`
- `PDF_EXTRACT_JSON_RETRIES`

### Detailed Explanation

- `FLOW_FILE`:
  The path to the JSON file that defines the requirement-question flow. In this repository it points to `questions.json` by default.

- `PDF_MAX_SIZE_MB`:
  A configured maximum file size setting for PDFs. It is part of the config layer.

- `PDF_MAX_PAGES`:
  The maximum number of PDF pages the text extraction helper will read.

- `PDF_PRELOAD_ON_STARTUP`:
  If true, the app also loads the PDF extractor model during startup.

- `PDF_EXTRACTOR_REPO`:
  Hugging Face repository for the optional dedicated PDF extraction model.

- `PDF_EXTRACTOR_FILENAME`:
  Exact PDF extractor model filename.

- `PDF_EXTRACTOR_CHAT_FORMAT`:
  Chat template format for the PDF extractor model.

- `PDF_EXTRACTOR_N_CTX`:
  Context size for the PDF extractor model.

- `PDF_EXTRACTOR_N_BATCH`:
  Batch size for the PDF extractor model.

- `PDF_EXTRACTOR_N_THREADS`:
  CPU thread count for the PDF extractor model.

- `PDF_EXTRACTOR_N_THREADS_BATCH`:
  Batch thread count for the PDF extractor model.

- `PDF_EXTRACTOR_N_GPU_LAYERS`:
  GPU offload layer count for the PDF extractor model.

- `PDF_EXTRACTOR_OFFLOAD_KQV`:
  Advanced runtime offload setting for the PDF extractor model.

- `PDF_EXTRACTOR_FLASH_ATTN`:
  Advanced attention optimization flag for the PDF extractor model.

- `PDF_QUESTION_BATCH_SIZE`:
  The PDF extractor does not ask the model about every requirement field at once. This variable decides how many question IDs it sends in one batch.

- `PDF_EXTRACT_CHUNK_LEN`:
  Long documents are split into smaller text chunks before extraction. This value controls chunk size.

- `PDF_EXTRACT_MAX_TOKENS`:
  The token limit for the model's extraction response.

- `PDF_EXTRACT_JSON_RETRIES`:
  If the model returns invalid JSON, the extractor tries again. This variable controls how many retries are allowed.

#### Streaming And User Experience Variables

### Quick List

- `STREAM_CHAR_DELAY_SECONDS`

### Detailed Explanation

- `STREAM_CHAR_DELAY_SECONDS`:
  The chat response is streamed character by character to create a typing effect. This variable controls the delay between characters.

### 2.5 Recommended Setup Steps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

If you use Docker instead:

```powershell
docker build -t swarai-backend .
docker run -p 7860:7860 swarai-backend
```

## 3. Models

This section explains both meanings of the word "model" in this project:

- AI models
- Data models used for request validation

### 3.1 AI Models Quick List

- Embedding model: `all-MiniLM-L6-v2`
- Main local chat model: downloaded from `LLAMA_REPO` and `LLAMA_FILENAME`
- Optional PDF extraction model: downloaded from `PDF_EXTRACTOR_REPO` and `PDF_EXTRACTOR_FILENAME`
- FAISS index for retrieval support

### 3.2 AI Model Terms Explained

- Embedding Model:
  An embedding model converts text into numbers called vectors. These vectors make it possible to compare text by meaning instead of exact wording.

- Vector:
  A vector is a list of numbers representing text in a mathematical form.

- Retrieval:
  Retrieval means searching stored text chunks to find the most relevant context before asking the LLM to answer.

- LLM:
  LLM means Large Language Model. It is the part that writes the final answer in natural language.

- Inference:
  Inference means actually running the model to generate an output.

- FAISS:
  FAISS is a similarity-search library. It stores vectors and quickly finds the nearest matches.

### 3.3 AI Models Used In This Repository

- `SentenceTransformer("all-MiniLM-L6-v2")`:
  This model is loaded in `backend.py` and `build_db.py`. It creates embeddings for user questions and for the knowledge chunks stored in FAISS.

- Main Llama model:
  This model is loaded through `llama-cpp-python` in `backend.py`. It answers normal chat questions and streams responses.

- PDF extractor Llama model:
  This is optional. If configured, `pdf_question_extractor.py` uses it to read uploaded requirement documents and return structured JSON answers.

- FAISS index:
  This is not a text-generating model, but it is a core retrieval component. It stores the vectorized knowledge base and later returns the most relevant chunks.

### 3.4 API And Request Models Quick List

- `LoginRequest`
- `RegisterRequest`
- `LoginResponse`
- `UpdateUserInfo`
- `ChatRequest`
- `CreateChatRequest`

### 3.5 What A Pydantic Model Means

- Pydantic:
  Pydantic is a Python library that checks incoming data and makes sure it has the expected fields and data types.

- `BaseModel`:
  This is the parent class used to create data-validation models.

- Request Model:
  A request model describes what the client must send to an API endpoint.

- Response Model:
  A response model describes what the API returns.

### 3.6 Detailed Explanation Of Each Model In `models.py`

- `LoginRequest`:
  Used when a user tries to log in.
  Fields:
  `user_id` is the login name.
  `password` is the plain password sent by the frontend before hashing comparison.

- `RegisterRequest`:
  Used when a new user signs up.
  Fields:
  `firstName`, `lastName`, and `email` store identity details.
  `user_id` is the chosen login ID.
  `password` is the new password that will later be hashed.

- `LoginResponse`:
  Used as the shape of the login result.
  Fields:
  `success` tells whether login worked.
  `token` stores the JWT when login succeeds.
  `message` gives a human-readable status message.

- `UpdateUserInfo`:
  Used when the frontend updates profile details.
  Fields:
  `user_id`, `first_name`, `last_name`, and `email` are the main identity values.
  `bio`, `location`, and `website` are optional profile fields.

- `ChatRequest`:
  This is the main input model for chatting.
  Fields:
  `message` is the user text.
  `user_id` identifies the user and defaults to `guest`.
  `session_id` connects the message to a stored chat session.
  `attachment` is an optional dictionary for uploaded file data such as name, type, and base64 bytes.

- `CreateChatRequest`:
  Used when the frontend creates a new chat session.
  Fields:
  `user_id` identifies the owner.
  `title` starts as `New Chat` unless the client provides another one.

### 3.7 Runtime Data Shapes Used Across The Project

- Requirement answer object:
  The code often stores answers in a dictionary like this:
  `{"value": ..., "confidence": ..., "source": ..., "evidence": ...}`

- `value`:
  The actual answer text or list of answers.

- `confidence`:
  A score from 0.0 to 1.0 that says how sure the extraction logic is.

- `source`:
  Where the answer came from, such as `user` or `pdf`.

- `evidence`:
  A supporting text snippet taken from the uploaded document.

- Session object:
  `chat_handlers.py` keeps a large in-memory session dictionary with fields such as `in_flow`, `node_id`, `context`, `product_contexts`, `active_product`, `customer_name`, `pdf_mode`, and `pdf_processed`.

- `requirement_schema.py` dictionary:
  This file contains `REQUIREMENT_FIELDS`, a helper mapping of requirement IDs to simple descriptions and examples. It is useful as documentation support, although the current runtime flow mainly depends on `questions.json` and `ALLOWED_QUESTION_IDS`.

## 4. Libraries

### 4.1 Quick List From `requirements.txt`

- `fastapi`
- `uvicorn`
- `sentence-transformers`
- `python-dotenv`
- `faiss-cpu`
- `passlib[bcrypt]`
- `pyjwt`
- `python-multipart`
- `llama-cpp-python`
- `huggingface-hub`
- `langchain-core`
- `supabase`
- `pypdf`
- `requests`
- `bcrypt`
- `reportlab`

### 4.2 What "Library" Means

- Library:
  A library is pre-written code made by someone else that your project can use instead of building everything from scratch.

- Dependency:
  A dependency is a library your project needs in order to run.

### 4.3 Detailed Explanation Of Each Installed Library

- `fastapi`:
  The main web framework. A framework is a larger structure that helps organize your whole app. FastAPI handles routes, request parsing, validation, and HTTP responses.

- `uvicorn`:
  The ASGI server that actually runs the FastAPI app. Think of FastAPI as the app logic and Uvicorn as the engine that serves it over HTTP.

- `sentence-transformers`:
  Used to create embeddings for semantic search. It helps the app understand meaning-based similarity.

- `python-dotenv`:
  Loads environment variables from `.env` into the Python process.

- `faiss-cpu`:
  Provides fast vector similarity search on CPU. This is the heart of the local retrieval system.

- `passlib[bcrypt]`:
  Used for password hashing and password verification. Hashing means storing a secure transformed version of the password instead of the raw password itself.

- `pyjwt`:
  Used to create and decode JWT access tokens for login.

- `python-multipart`:
  Commonly used in FastAPI projects for file uploads and form data. In this codebase, attachments are currently passed as structured request data rather than browser multipart forms, but the package is still installed.

- `llama-cpp-python`:
  Lets Python run GGUF Llama-style models locally.

- `huggingface-hub`:
  Downloads model files from Hugging Face repositories.

- `langchain-core`:
  Used here only for `PromptTemplate`. A prompt template is a reusable text pattern for building LLM prompts.

- `supabase`:
  Python client used to connect to Supabase tables and storage buckets.

- `pypdf`:
  Reads PDF files and extracts text from them.

- `requests`:
  A common HTTP library. It is installed but not directly used in the main visible source files right now.

- `bcrypt`:
  Low-level password hashing support used under the passlib setup.

- `reportlab`:
  Generates the final requirement PDF file that users can download.

### 4.4 Important Built-In Python Modules Used Often

- `asyncio`:
  Handles asynchronous work such as streaming, background model calls, and locks.

- `json`:
  Reads and writes JSON data such as `questions.json` and extractor output.

- `re`:
  Regular expression support. Regular expressions are search patterns used to match or clean text.

- `logging`:
  Prints useful runtime messages and errors.

- `datetime`:
  Stores timestamps such as `created_at` and `updated_at`.

- `uuid`:
  Generates unique IDs for chat sessions and guest users.

- `base64`:
  Decodes uploaded file bytes sent as text.

- `zipfile`:
  Reads `.docx` files because a DOCX file is really a ZIP archive containing XML files.

- `pickle`:
  Saves and loads Python objects such as FAISS metadata.

- `os`:
  Reads environment variables, paths, and system-level details.

## 5. Other Important Project Parts

### 5.1 Quick Big-Picture List

- The app starts in `main.py`
- Authentication lives in `auth.py`
- Chat routes live in `chats.py`
- Most chat decision-making lives in `chat_handlers.py`
- LLM and retrieval logic lives in `backend.py`
- Supabase access lives in `db.py` and `storage_utils.py`
- Requirement flow structure lives in `questions.json`
- PDF extraction lives in `extract_pdf_text.py` and `pdf_question_extractor.py`
- PDF output lives in `pdf_generator.py`
- Chat persistence helpers live in `chat_persistence.py`
- Session memory lives in `chat_state.py`

### 5.2 End-To-End Flow In Plain Language

- Startup flow:
  The app starts in `main.py`, creates a FastAPI app, enables CORS, includes the auth and chat routers, and preloads AI models on startup.

- Authentication flow:
  A user can register, log in, or create a temporary guest session.

- Normal chat flow:
  The backend receives a message, checks whether it is a greeting or a requirement-flow trigger, retrieves relevant knowledge from FAISS, asks the local LLM to answer, and streams the answer back.

- Requirement collection flow:
  If the user asks about services such as audio porting or optimization, the system switches into a structured question flow defined in `questions.json`.

- File upload flow:
  If the user uploads a PDF, DOC, DOCX, or TXT file, the backend extracts text, asks the extractor model to map document content to requirement fields, fills in whatever it can, and asks only the missing follow-up questions.

- Submission flow:
  When enough information is collected, the backend generates a formatted requirement PDF and uploads it to Supabase storage.

### 5.3 File-By-File Documentation

#### `main.py`

Quick list:
- Creates the FastAPI app
- Configures CORS
- Includes routers
- Preloads models on startup

Detailed explanation:
- `app = FastAPI(...)` creates the web application object.
- `app.add_middleware(CORSMiddleware, ...)` allows configured frontend origins to call the backend.
- `app.include_router(auth_router)` connects auth endpoints.
- `app.include_router(chats_router)` connects chat endpoints.
- `root()` is a health-check route at `/`.
- `startup_event()` preloads the models so the first user request is faster.
- The `uvicorn.run(...)` block is a local direct-run entry, although the normal startup command for this project is `uvicorn main:app`.

#### `auth.py`

Quick list:
- `/auth/login`
- `/auth/register`
- `/auth/guest`

Detailed explanation:
- `login(req)` looks up the user in Supabase, verifies the password, and returns a JWT token.
- `register(req)` checks whether the user already exists, hashes the password, and saves the new user in the `users` table.
- `guest_login()` creates a temporary guest ID and short-lived token without storing a full account.

#### `db.py`

Quick list:
- Creates the Supabase client
- Provides reusable CRUD helpers

Detailed explanation:
- At import time, the file tries to create a Supabase client using `SUPABASE_URL` and `SUPABASE_KEY`.
- `insert_row(table, payload)` inserts one row.
- `select_rows(table, filters, order, limit)` fetches rows with optional filters and sorting.
- `delete_rows(table, where)` deletes matching rows.
- `update_row(table, where, payload)` updates matching rows.

#### `utils.py`

Quick list:
- Password hashing
- Password verification
- JWT creation
- JWT decoding
- Text normalization

Detailed explanation:
- `hash_password(password)` creates a secure hash before storing passwords.
- `verify_password(plain, hashed)` checks whether the entered password matches the stored hash.
- `create_access_token(subject, expires_delta)` builds a signed JWT token.
- `decode_token(token)` verifies and decodes that token.
- `normalize_text(s)` lowercases and cleans text so fuzzy matching becomes easier.

#### `config.py`

Quick list:
- Loads `.env`
- Centralizes base configuration constants

Detailed explanation:
- `load_dotenv()` loads environment values from `.env`.
- The rest of the file turns environment variables into Python constants used across the project.
- This file is the main place a beginner should read to understand which settings the app expects.

#### `models.py`

Quick list:
- `LoginRequest`
- `RegisterRequest`
- `LoginResponse`
- `UpdateUserInfo`
- `ChatRequest`
- `CreateChatRequest`

Detailed explanation:
- This file does not contain database models.
- It contains Pydantic request and response models that validate API payloads.

#### `backend.py`

Quick list of responsibilities:
- Loads embedding model
- Loads main Llama model
- Optionally loads PDF extractor model
- Loads or creates the FAISS index
- Builds prompts
- Retrieves similar knowledge chunks
- Generates answers
- Streams answers token by token

Quick list of important functions:
- `_env_int`
- `_env_bool`
- `_resolve_chat_format`
- `_is_pdf_extractor_configured`
- `_build_llama_instance`
- `_load_models_and_index_sync`
- `_load_pdf_extractor_model_sync`
- `_async_preload_models`
- `_async_warmup_models`
- `preload_models_for_startup`
- `normalize_text`
- `matches_services_trigger`
- `split_text_safely`
- `_build_prompt`
- `estimate_tokens`
- `_async_load_models_and_index`
- `_async_load_pdf_extractor_model`
- `load_models_if_needed`
- `get_embedding`
- `get_similar_chunks`
- `_ensure_index_loaded`
- `add_chat_to_faiss`
- `sync_supabase_history_to_faiss`
- `_call_model_async`
- `_call_pdf_extractor_async`
- `_extract_stream_token`
- `generate_answer_stream_async`
- `sanitize_output`
- `generate_answer_async`
- `generate_answer`

Detailed explanation:
- `_env_int(name, default)` safely reads integer settings from the environment.
- `_env_bool(name, default)` safely reads boolean settings.
- `_resolve_chat_format(name, default)` decides which chat template format the model should use.
- `_is_pdf_extractor_configured()` checks whether a dedicated extractor model has been configured.
- `_build_llama_instance(...)` creates a Llama runtime object with only the parameters supported by the installed library version.
- `_load_models_and_index_sync()` loads the embedding model, the main chat model, and the FAISS index.
- `_load_pdf_extractor_model_sync()` loads the optional PDF extractor model.
- `_async_preload_models(...)` wraps loading in async-friendly form.
- `_async_warmup_models(...)` sends tiny warmup prompts.
- `preload_models_for_startup()` is the startup entry used by `main.py`.
- `normalize_text(s)` is a local text cleaner used for service-trigger matching.
- `matches_services_trigger(text)` decides whether a user message should start the requirement flow.
- `split_text_safely(text, max_len)` breaks long text into safer chunks.
- `_build_prompt(context_text, user_input)` combines system instructions, retrieved context, and the user question into one prompt.
- `estimate_tokens(text)` estimates token count using a simple character-based rule.
- `_async_load_models_and_index()` ensures loading happens only once with an async lock.
- `_async_load_pdf_extractor_model()` async wrapper for PDF extractor loading.
- `load_models_if_needed()` loads models from either sync or async contexts.
- `get_embedding(text)` creates and caches an embedding vector.
- `get_similar_chunks(query, top_k)` searches FAISS for the most similar stored chunks.
- `_ensure_index_loaded()` reloads the FAISS index if the on-disk file changed.
- `add_chat_to_faiss(query, response)` adds new chat content to the vector index.
- `sync_supabase_history_to_faiss()` pushes old stored messages into FAISS.
- `_call_model_async(prompt_text, max_tokens)` runs the main LLM for non-streamed answers.
- `_call_pdf_extractor_async(prompt_text, max_tokens)` runs the PDF extractor model and expects JSON-style output.
- `_extract_stream_token(chunk)` extracts the text token from a llama streaming chunk.
- `generate_answer_stream_async(query)` performs retrieval, builds the prompt, and yields answer tokens one by one.
- `sanitize_output(text)` removes forbidden tags such as `<think>` and cleans duplicated output.
- `generate_answer_async(query)` performs full non-streaming answer generation.
- `generate_answer(query)` is a sync wrapper around the async answer function.

#### `llm_manager.py`

Quick list:
- Async wrapper for full answer generation
- Async wrapper for streamed answer generation
- Import fallback handling

Detailed explanation:
- This file acts as a compatibility layer around `backend.py`.
- `generate_answer_async(prompt)` calls the backend answer function and returns a safe fallback string if generation fails.
- `generate_answer_stream_async(prompt)` forwards streamed tokens from the backend and provides a fallback message if needed.

#### `faiss_manager.py`

Quick list:
- Search wrapper
- Add wrapper

Detailed explanation:
- This file is a thin compatibility layer around the FAISS helpers in `backend.py`.
- `search(query, top_k)` forwards to `get_similar_chunks`.
- `add(query, response)` forwards to `add_chat_to_faiss`.

#### `greetings.py`

Quick list:
- Greeting keyword set
- Greeting reply template
- Greeting detection function

Detailed explanation:
- `GREETING_KEYWORDS` stores canonical greeting phrases.
- `GREETING_REPLY` is the default greeting answer shown by the bot.
- `is_greeting(text)` normalizes the input and checks exact match, prefix match, and fuzzy match.

#### `flows.py`

Quick list:
- Loads `questions.json`
- Creates `FlowManager`
- Finds the next matching option

Detailed explanation:
- `conversation_flow` is the loaded JSON object from the flow file.
- `FlowManager.__init__(flow)` stores the flow map.
- `get_node(node_id)` returns one flow node.
- `start_flow_for_user(session)` resets the session and starts from `start`.
- `restart_flow(session)` clears flow state.
- `start_from_node(session, node_id)` starts from a specific node, useful when PDF extraction already answered earlier questions.
- `find_best_option(options, user_text, cutoff)` performs exact match, contains match, fuzzy match, and token match to map a user's text to one of the valid flow options.
- `flow_manager` is the shared FlowManager instance used by the chat handlers.

#### `chat_state.py`

Quick list:
- `user_sessions`

Detailed explanation:
- `user_sessions` is a simple in-memory dictionary that stores live session state while the server process is running.
- Important warning for beginners:
  this is memory-only state, so it resets if the backend restarts.

#### `chat_persistence.py`

Quick list:
- Immediate user message save
- Full chat-pair persistence
- Smart title generation
- Session title update

Detailed explanation:
- `save_user_message_immediately(session_id, user_id, message)` writes the user message to Supabase before the assistant reply is ready.
- `persist_chat_pair(...)` writes the user message, assistant reply, optional attachments, and session timestamps in a consistent order.
- `generate_instant_smart_title(msg, in_flow)` creates a readable chat title based on the first meaningful user message.
- `update_session_title_if_needed(session, session_id, message, in_flow)` writes the title only once, so later messages do not constantly rename the session.

#### `chats.py`

Quick list of responsibilities:
- Chat endpoints
- User info endpoints
- Chat history endpoints
- SSE streaming
- Routing requests into greeting, normal QA, requirement flow, or file-upload flow

Quick list of important functions:
- `_get_stream_char_delay_seconds`
- `reset_model_context`
- `update_model_context`
- `estimate_tokens`
- `_infer_attachment_kind`
- `save_user_message_immediately`
- `get_user_info`
- `update_user_info`
- `get_chats`
- `get_chat`
- `create_chat`
- `delete_chat`
- `_sse_data_chunk`
- `_sse_event_chunk`
- `_normalize_option_labels`
- `_extract_stream_meta`
- `_validate_stream_prompt`
- `_extract_reply_text_for_stream`
- `_stream_text_char_by_char`
- `_token_stream_generator`
- `_stream_response`
- `_process_chat_request`
- `chat_endpoint`
- `chat_stream_endpoint`
- `search_chats`
- `clear_chat`
- `logout`

Detailed explanation:
- `_get_stream_char_delay_seconds()` reads the typing-effect delay from the environment.
- `reset_model_context(session_id)` resets token-tracking info for a session.
- `update_model_context(session_id, tokens)` increments the tracked token count.
- `estimate_tokens(text)` estimates token usage.
- `_infer_attachment_kind(attachment)` decides whether an upload is PDF, DOC, DOCX, or TXT.
- `save_user_message_immediately(...)` writes a user message and updates the chat session timestamp.
- `get_user_info(user_id)` returns selected user profile fields from Supabase.
- `update_user_info(req)` updates user profile details.
- `get_chats(user_id)` returns all chat sessions for a user.
- `get_chat(chat_id)` returns all messages inside one chat session.
- `create_chat(payload)` creates a new chat session row and links it to the in-memory session.
- `delete_chat(chat_id)` removes a chat session and its messages.
- `_sse_data_chunk(token)` converts text into the `data: ...` format expected by Server-Sent Events.
- `_sse_event_chunk(event_name, payload)` sends named SSE events such as metadata.
- `_normalize_option_labels(raw_options)` turns option objects into plain label strings.
- `_extract_stream_meta(result)` builds metadata payloads such as follow-up options for the frontend.
- `_validate_stream_prompt(req)` blocks empty requests and invalid attachments.
- `_extract_reply_text_for_stream(result)` extracts the plain text reply from handler results.
- `_stream_text_char_by_char(text, request, stream_name)` creates the typing effect and stops if the client disconnects.
- `_token_stream_generator(req, request, stream_name)` is the main streaming engine. It decides whether the request is greeting, guest chat, flow work, or normal streaming QA.
- `_stream_response(...)` is a helper for building a `StreamingResponse`.
- `_process_chat_request(req)` is the non-SSE route decision engine that sends the request to guest chat, greeting handling, service trigger handling, normal QA, PDF extraction, or the flow engine.
- `chat_endpoint(req, request)` exposes `POST /chats/chat`.
- `chat_stream_endpoint(req, request)` exposes `POST /chats/chat_stream`.
- `search_chats(user_id, q)` searches chat titles.
- `clear_chat(chat_id)` removes messages but keeps the session row and resets its title.
- `logout(user_id)` clears in-memory session state for that user.

#### `chat_handlers.py`

Quick list of responsibilities:
- Guest chat handling
- Session creation and retrieval
- Greeting handling
- Requirement flow start
- Normal QA handling
- Document upload and extraction
- Summary building and editing
- Flow navigation and submission

Quick list of important functions:
- `_infer_attachment_kind`
- `_normalize_extracted_text`
- `_extract_docx_text`
- `_extract_txt_text`
- `_extract_doc_text`
- `_extract_text_from_uploaded_document`
- `_option_labels`
- `_node_option_labels`
- `_extract_customer_name_from_text`
- `_customer_name_from_session`
- `_build_pdf_context_for_export`
- `_has_value`
- `_service_key_from_context`
- `_entry_value_to_text`
- `_extract_query_values_from_context`
- `_collect_queries_for_summary`
- `_build_numbered_summary`
- `_set_query_by_index`
- `_relevant_requirement_ids`
- `_next_unanswered_requirement_id`
- `_set_active_product`
- `_active_product_context`
- `_next_missing_product_and_qid`
- `_requirement_prompt_prefix`
- `handle_guest_chat`
- `init_or_get_session`
- `get_llm_reply`
- `handle_greeting`
- `handle_service_trigger`
- `handle_normal_qa`
- `handle_pdf_upload_and_extraction`
- `_next_requirement_or_final`
- `handle_flow_engine`

Detailed explanation:
- `_infer_attachment_kind(attachment)` classifies uploaded file type.
- `_normalize_extracted_text(raw_text)` cleans extracted text so it is easier to process.
- `_extract_docx_text(docx_bytes)` reads DOCX XML content and converts it to plain text.
- `_extract_txt_text(txt_bytes)` tries multiple encodings to read text files safely.
- `_extract_doc_text(doc_bytes)` uses best-effort text recovery for old binary `.doc` files.
- `_extract_text_from_uploaded_document(file_bytes, kind)` picks the correct extraction helper based on file type.
- `_option_labels(options)` converts flow options into plain label text.
- `_node_option_labels(node)` gets option labels from a flow node.
- `_extract_customer_name_from_text(pdf_text)` uses regex patterns to find a customer name inside uploaded document text.
- `_customer_name_from_session(session)` returns an already stored customer name from session memory.
- `_build_pdf_context_for_export(session)` prepares context in the format expected by the PDF generator.
- `_has_value(entry)` checks whether an answer entry contains a real value.
- `_service_key_from_context(context)` maps a chosen service to one of the internal service categories.
- `_entry_value_to_text(entry)` converts stored answer objects into human-readable text.
- `_extract_query_values_from_context(ctx)` reads stored user queries from context.
- `_collect_queries_for_summary(session)` gathers queries from the main context and product-specific contexts.
- `_build_numbered_summary(session)` creates the numbered summary shown before final submission and builds an index map for later editing.
- `_set_query_by_index(session, query_index, new_value)` updates a selected query after the user chooses a summary number to edit.
- `_relevant_requirement_ids(context)` returns only the requirement IDs relevant to the chosen service type.
- `_next_unanswered_requirement_id(context)` finds the next missing requirement.
- `_set_active_product(session, product_name)` switches the session to the product currently being discussed.
- `_active_product_context(session)` returns the context for the currently active product.
- `_next_missing_product_and_qid(session)` searches across products and finds the next unanswered requirement.
- `_requirement_prompt_prefix(session, next_qid)` adds a small prefix like "Asking remaining requirements for Product X" when multiple products exist.
- `handle_guest_chat(user_id, message)` handles chatting for guest users without full account persistence behavior.
- `init_or_get_session(req, user_id)` creates or retrieves the in-memory session object and ensures default keys exist.
- `get_llm_reply(text)` returns one full non-streamed LLM reply.
- `handle_greeting(...)` sends the greeting reply and stores it.
- `handle_service_trigger(...)` starts the requirement flow and sends the first question from `start`.
- `handle_normal_qa(...)` persists the user message, gets the answer, then persists the assistant reply.
- `handle_pdf_upload_and_extraction(...)` is one of the most important functions in the whole project. It validates the attachment, decodes base64, extracts text, uploads the original file, extracts structured requirements, merges those answers into product contexts, generates a partial PDF, and either completes the flow or asks the next missing requirement question.
- `_next_requirement_or_final(session)` chooses the next missing requirement or jumps to `final_step`.
- `handle_flow_engine(...)` is the main requirement-flow controller. It handles customer-name capture, mistake prompts, summary editing, start-node branching, skip-without-PDF logic, requirement-node answers, query collection, summary display, and final PDF submission.

#### `extract_pdf_text.py`

Quick list:
- PDF format check
- PDF parsing
- Read PDF pages safely

Detailed explanation:
- `PDFParseError` is a custom error class for PDF reading problems.
- `_looks_like_pdf(pdf_bytes)` checks whether the uploaded bytes look like a real PDF.
- `extract_pdf_text(pdf_bytes)` validates the file and returns extracted text.
- `read_pdf_text_from_bytes(pdf_bytes, max_pages)` reads PDF pages one by one and joins their text.

#### `pdf_question_extractor.py`

Quick list of responsibilities:
- Breaks long document text into chunks
- Batches requirement questions
- Prompts the extractor model
- Repairs invalid JSON when possible
- Merges extracted answers with confidence and evidence checks

Quick list of important functions:
- `_to_float`
- `_merge_answer`
- `_normalize_model_output`
- `_iter_question_batches`
- `_build_question_block`
- `_compact_options`
- `_is_evidence_relevant`
- `_contains_evidence_snippet`
- `_load_model_json`
- `_extract_batch_json`
- `extract_answers_from_pdf`

Detailed explanation:
- `_to_float(value, default)` safely converts a value to float.
- `_merge_answer(existing, incoming)` keeps the stronger answer based on confidence, and merges lists if confidence ties.
- `_normalize_model_output(data)` accepts both supported JSON shapes and normalizes them into a product-to-question mapping.
- `_iter_question_batches()` splits all allowed question IDs into smaller groups.
- `_build_question_block(questions, qids)` creates the question instructions sent to the model.
- `_compact_options(raw_options, ...)` keeps the options list short enough for prompts.
- `_is_evidence_relevant(qid, value, evidence)` checks whether the evidence actually supports that value.
- `_contains_evidence_snippet(chunk, evidence)` verifies that the evidence really came from the current document chunk.
- `_load_model_json(raw)` tries to parse model output, including some repair attempts for truncated JSON.
- `_extract_batch_json(prompt)` calls the extractor model and retries if JSON output is invalid.
- `extract_answers_from_pdf(pdf_text, questions)` is the main extraction function. It splits the document into chunks, asks the model about batches of requirement IDs, validates evidence, and returns a complete answer map for every product and every allowed question ID.

#### `pdf_generator.py`

Quick list of responsibilities:
- Labels requirement fields
- Filters visible fields by service type
- Builds requirement tables
- Builds analysis sections
- Generates the final PDF bytes

Quick list of important functions:
- `_has_value`
- `_format_value`
- `_read_text_value`
- `_first_non_na`
- `_read_queries`
- `_build_project_requirement_analysis_lines`
- `_extract_max_frequency_khz`
- `_extract_memory_mb`
- `build_summary_sections`
- `_read_customer_name`
- `_normalize_product_contexts`
- `_service_key_from_product_context`
- `_visible_requirement_ids`
- `build_pdf_rows`
- `generate_final_requirements_pdf`

Detailed explanation:
- `_has_value(entry)` checks whether a stored answer contains usable data.
- `_format_value(entry)` converts answer objects into display text such as `N/A`, a string, or a comma-joined list.
- `_read_text_value(product_context, qid)` reads a formatted value for one field.
- `_first_non_na(*values)` returns the first meaningful value instead of `N/A`.
- `_read_queries(context)` extracts user queries to include in the PDF.
- `_build_project_requirement_analysis_lines(product_items, customer_name)` creates the text of the analysis section on the last PDF page.
- `_extract_max_frequency_khz(text)` reads the biggest frequency value from text like `48 kHz` or `192 kHz`.
- `_extract_memory_mb(text)` converts memory text like `512 MB` or `1 GB` into MB for simple rule checks.
- `build_summary_sections(product_name, product_context, customer_name)` creates a structured summary including key requirements and technical observations.
- `_read_customer_name(context)` looks for customer name fields in the context.
- `_normalize_product_contexts(context)` reshapes context so the PDF generator can treat single-product and multi-product cases the same way.
- `_service_key_from_product_context(product_context)` maps the selected service to `porting`, `optimization`, or `audio_app`.
- `_visible_requirement_ids(product_context)` hides irrelevant requirement rows for the selected service type.
- `build_pdf_rows(product_name, product_context, questions, customer_name)` creates the row list that becomes the PDF table.
- `generate_final_requirements_pdf(context, questions)` builds the complete styled PDF, including requirement tables, query section, and analysis page.

#### `build_db.py`

Quick list:
- Reads source documents from `data/`
- Chunks text
- Embeds chunks
- Builds FAISS index
- Saves FAISS files

Detailed explanation:
- `load_text(path)` loads text from a plain text file or a PDF.
- `chunk(text, size)` splits long text into chunk-sized word groups.
- `read_pdf_text_from_bytes(pdf_bytes, max_pages)` is a helper for PDF text extraction in this script.
- The bottom part of the file is script-style code, not wrapped in a function. It loops through `data/`, creates embeddings, builds a FAISS index, and writes `faiss_db/index.faiss` and `faiss_db/metadata.pkl`.

#### `storage_utils.py`

Quick list:
- Upload any supported file to Supabase storage
- Upload generated PDF to Supabase storage

Detailed explanation:
- `_CONTENT_TYPES_BY_EXT` maps file extensions to MIME types.
- `upload_file_to_supabase(file_bytes, filename, content_type)` uploads a file to the `chat-files` bucket and returns its public URL.
- `upload_pdf_to_supabase(pdf_bytes, filename)` is a small wrapper specialized for PDFs.

#### `requirement_schema.py`

Quick list:
- `REQUIREMENT_FIELDS`

Detailed explanation:
- This file contains human-readable descriptions and examples for some requirement fields.
- It is useful for understanding what fields like `DSP_Processor` or `Audio_Params_2` mean.
- In the current codebase, this file appears to be a helper/reference file rather than a core runtime dependency.

#### `questions.json`

Quick list of important nodes:
- `start`
- `pdf_upload`
- `service_select`
- `Optimization_type`
- `App_type`
- `Porting_type`
- `Porting_question_1`
- `Porting_question_2`
- `DSP_Processor`
- `Application`
- `Audio_Interface`
- `Audio_Params_1`
- `Audio_Params_2`
- `Audio_Params_3`
- `Audio_Tech_1`
- `Audio_Tech_2`
- `CodeBase_1`
- `CodeBase_2`
- `CodeBase_3`
- `CodeBase_4`
- `CodeBase_5`
- `CodeBase_6`
- `TargetPlatform_1`
- `TargetPlatform_2`
- `final_step`
- `show_summary`
- `make_changes`
- `await_new_answer`
- `show_updated_summary`
- `query_check`
- `query_input`
- `query_more`
- `submit_response`
- `mistake_prompt`
- `end`

Detailed explanation:
- This file is the conversation map for requirement collection.
- Each node contains some combination of:
  `id`, `text`, `options`, `type`, `expect_user_input`, and `next`.
- Nodes with `options` act like multiple-choice questions.
- Nodes with `type: "input"` or `expect_user_input: true` expect free text.
- `next` tells the chatbot which node to visit after the current answer.

#### `README.md`

Quick list:
- Short project intro
- Endpoint list
- SSE note

Detailed explanation:
- This is the short top-level readme for the repository.
- It is useful for a quick glance, but it is much less detailed than this documentation file.

#### `PROJECT_FLOW_DOCUMENTATION.md`

Quick list:
- Beginner-oriented project guide

Detailed explanation:
- This file is now the main onboarding document for the repository.
- It explains the system from setup to file-by-file responsibilities.

#### `requirements.txt`

Quick list:
- Python dependency list

Detailed explanation:
- This file tells `pip` which libraries need to be installed.
- It is the main dependency source for local setup and for the Docker image.

#### `Dockerfile`

Quick list:
- Uses Python 3.10 slim image
- Installs system packages
- Copies app code
- Installs Python dependencies
- Starts Uvicorn

Detailed explanation:
- `FROM python:3.10-slim` picks a lightweight Python 3.10 base image.
- The `apt-get install` line installs system-level build tools and math libraries needed by some AI packages.
- `WORKDIR /app` sets the working directory inside the container.
- `COPY . .` copies the repository into the container.
- `RUN pip install -r requirements.txt` installs Python dependencies.
- `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]` starts the backend server.

#### `.gitignore`

Quick list:
- Ignores virtual environment files
- Ignores Python cache files
- Ignores FAISS index files

Detailed explanation:
- This file tells Git which generated files should usually not be tracked.
- In this project it ignores `.venv`, `__pycache__`, `.pyc` files, and FAISS artifacts.

#### `.env`

Quick list:
- Local secrets and runtime settings

Detailed explanation:
- This file stores local environment values.
- Beginners should understand that this file often contains secrets and should be handled carefully.

#### `__init__.py`

Quick list:
- Package marker
- Version string

Detailed explanation:
- This file marks the folder as a Python package and defines `__version__ = "0.2.0"`.

#### `.vscode/settings.json`

Quick list:
- Editor convenience setting

Detailed explanation:
- This file hides a Postman dotenv-notification inside VS Code.
- It does not affect backend runtime behavior.

### 5.4 Data Folder Documentation

Quick list:
- `apo_architecture.txt`
- `audio_engine_pipeline.txt`
- `developing_audio_apps.txt`
- `iaudioclient3_details.txt`
- `mmdevice_api.txt`
- `wasapi_overview.txt`

Detailed explanation:
- `apo_architecture.txt` explains Audio Processing Objects and their role in Windows audio pipelines.
- `audio_engine_pipeline.txt` explains how the Windows audio engine moves and processes sound.
- `developing_audio_apps.txt` covers background knowledge for building Windows audio applications.
- `iaudioclient3_details.txt` explains low-latency shared-mode audio concepts around `IAudioClient3`.
- `mmdevice_api.txt` explains Windows multimedia device enumeration concepts.
- `wasapi_overview.txt` explains WASAPI, which is the domain focus of the chat assistant.

These files are the source knowledge used when the FAISS database is built.

### 5.5 FAISS Database Folder Documentation

Quick list:
- `faiss_db/index.faiss`
- `faiss_db/metadata.pkl`

Detailed explanation:
- `index.faiss` is the actual vector index used for nearest-neighbor search.
- `metadata.pkl` stores the original chunk text and source filenames that match those vectors.

### 5.6 Supabase Tables And Storage Used By The Project

Quick list:
- `users`
- `chat_sessions`
- `chat_messages`
- Storage bucket: `chat-files`

Detailed explanation:
- `users` stores login and profile information.
- `chat_sessions` stores one row per chat session, including title and timestamps.
- `chat_messages` stores the actual user and assistant messages, plus optional attachments.
- `chat-files` stores uploaded original documents and generated PDF outputs.

### 5.7 Best Reading Order For A Beginner

If someone is completely new to the project, this is the easiest order:

1. Read `README.md`
2. Read `main.py`
3. Read `config.py`
4. Read `models.py`
5. Read `auth.py`
6. Read `chats.py`
7. Read `chat_handlers.py`
8. Read `backend.py`
9. Read `questions.json`
10. Read `pdf_question_extractor.py`
11. Read `pdf_generator.py`
12. Read `db.py`, `chat_persistence.py`, and `storage_utils.py`

### 5.8 One-Sentence Summary Of The Entire Repository

This repository is a FastAPI backend that combines local LLM answering, FAISS-based retrieval, Supabase-based persistence, and a guided requirement-collection flow that can also extract requirements from uploaded PDF, DOC, DOCX, and TXT files and then generate a final PDF report.
