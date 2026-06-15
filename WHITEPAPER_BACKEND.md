# SwarAI Backend Whitepaper

## AI-Assisted Requirement Collection and SRS Generation for Audio Engineering Workflows

### Backend-Focused Technical and Business Overview

**Document type:** Backend whitepaper  
**Product:** SwarAI Chat Bot Backend  
**Primary focus:** Requirement collection, technical chat assistance, retrieval-augmented AI, document ingestion, and automated SRS generation  
**Intended audience:** Company website visitors, engineering leaders, product teams, solution architects, delivery managers, and prospective customers  

---

## Executive Summary

SwarAI is an AI-assisted chat backend designed to accelerate requirement discovery and Software Requirement Specification (SRS) generation for audio engineering services. The backend combines a guided conversational flow, document extraction, retrieval-augmented generation, local/open-source AI models, FAISS vector search, Supabase-backed persistence, and automated document generation into one practical workflow.

The system helps teams move from scattered customer inputs, uploaded requirement documents, and informal conversations to structured requirement outputs. It can guide users through service-specific questions, extract requirement information from uploaded documents, generate requirements PDFs, produce SRS DOCX files from an engineering template, and optionally compare a baseline SRS with a local AI-generated version.

The core value is speed and cost control. Instead of relying only on manual requirement analysis or paid AI APIs, the backend uses free and open-source components such as FastAPI, FAISS, SentenceTransformers, Ollama-compatible local models, Qwen-based generation, PDF/DOCX parsing libraries, and an optional Google Colab GPU workflow. This approach reduces dependency on paid per-token services during development and prototyping while helping engineers generate requirement artifacts faster.

---

## Problem Statement

Requirement generation is often slowed down by fragmented inputs and repeated manual effort. Engineering teams may receive requirement details through calls, documents, emails, sample PDFs, and partial technical notes. Converting this information into a clear requirement set or SRS document can take significant time, especially when the team must:

- Ask repetitive discovery questions.
- Review long technical documents manually.
- Identify missing requirement fields.
- Convert informal answers into structured engineering language.
- Prepare customer-ready PDF or DOCX outputs.
- Maintain chat history and traceability across sessions.
- Avoid rework caused by incomplete or inconsistent requirement data.

For audio engineering services, the challenge becomes more specific. Teams often need to capture details about porting, optimization, audio applications, DSP processors, audio interfaces, sampling rates, PCM sample sizes, platforms, codebase properties, chipset details, memory constraints, and customer queries. A generic chatbot is not enough. The system needs a domain-aware backend workflow that can structure the conversation and generate useful artifacts.

---

## Solution Overview

SwarAI addresses this challenge with a backend that supports both conversational assistance and requirement automation.

At a high level, the backend provides:

- A FastAPI application for chat, authentication, SRS, and document APIs.
- Guided requirement collection through a configurable JSON flow.
- Retrieval-augmented AI responses using FAISS and an audio-domain knowledge base.
- Local or Ollama-compatible model integration for cost-conscious AI inference.
- Optional llama.cpp model support with CPU/GPU fallback.
- PDF, DOC, DOCX, and TXT document ingestion.
- Structured extraction of answers from uploaded requirement documents.
- Automated requirement summary and PDF generation.
- Automated SRS JSON extraction and DOCX rendering.
- Optional local AI SRS regeneration and comparison.
- Supabase-backed user, session, message, and file persistence.
- Guest-session support for temporary user access.
- Server-Sent Events streaming for responsive chat experiences.

This backend is designed to act as the intelligence and workflow layer behind a customer-facing chatbot interface.

---

## Backend Scope

This whitepaper focuses only on the backend codebase. It is intentionally complementary to a separate frontend whitepaper.

The backend is responsible for:

- API orchestration.
- User authentication and guest sessions.
- Chat state management.
- Requirement flow execution.
- LLM and retrieval integration.
- Document parsing and extraction.
- SRS generation.
- File upload and storage integration.
- Chat history persistence.
- Streaming response delivery.

The frontend can present these capabilities through chat screens, upload components, option selectors, progress states, document download buttons, and comparison views. The backend provides the underlying services and structured responses needed by that frontend.

---

## Key Capabilities

### 1. Guided Requirement Collection

The backend uses a configurable questions.json flow to guide users through service-specific requirement capture. The flow starts by asking whether the user has an existing requirement document. If not, the user is guided through structured questions.

Supported service paths include:

- Porting.
- Optimization.
- Audio application development.

The flow captures technical inputs such as:

- Target audio algorithm platform.
- Cross-compliance requirements.
- DSP processor selection.
- Audio interface.
- PCM sample size.
- Sampling frequency.
- Audio format.
- Required audio processing modules.
- Current supported platforms.
- Programming language.
- Sample app type.
- Approximate code size.
- Memory requirement.
- Source/library availability.
- Fixed-point or floating-point implementation.
- Target platform and chipset details.

This approach reduces the time spent asking repetitive discovery questions and helps ensure that important engineering details are not missed.

### 2. Document-First Requirement Extraction

If the user already has a requirements document, the backend can accept PDF, DOC, DOCX, and TXT uploads. It extracts readable content, stores the file, and attempts to convert the document into structured requirement data.

The document-first path supports two outcomes:

- Full SRS generation when the document contains enough structured requirement data.
- Partial requirement extraction when the document contains useful information but still needs user clarification.

This makes the system flexible for real customer scenarios where inputs may be complete, partial, or inconsistently formatted.

### 3. Automated SRS Generation

The backend includes an SRS intelligence pipeline that converts engineering documents into structured SRS data and renders it into a DOCX template.

The SRS pipeline can:

- Parse source documents into ordered text, table, and heading blocks.
- Preserve document hierarchy and table-based requirement sections.
- Detect requirement IDs and requirement-like content.
- Extract functional, non-functional, safety, cybersecurity, interface, diagnostic, and acceptance-related information.
- Populate a structured SRSProject model.
- Validate missing or weak fields.
- Fill missing fields using source context and engineering inference.
- Render the final output into a master SRS DOCX template.
- Save structured JSON for traceability.

The generated SRS output includes requirement tables, document metadata, references, definitions, revision information, and requirement management information where available.

### 4. AI-Assisted SRS Regeneration and Comparison

After generating the baseline SRS, the backend can optionally create a second SRS using a local AI setup. The current implementation supports:

- Google Colab FastAPI integration for Qwen2.5 7B SRS generation.
- Ollama-compatible local model fallback.
- BM25-based source excerpt selection before generation.
- Structured JSON normalization.
- Missing-field completion.
- DOCX rendering of the AI-generated version.
- Comparison between the baseline and AI-generated SRS.

The comparison includes metrics such as:

- Number of requirements.
- Confidence score.
- Auto-filled fields.
- Validation warnings.
- Functional requirement count.
- Non-functional requirement count.

This gives users a practical way to compare two generated SRS versions and select the final one.

### 5. Retrieval-Augmented Chat

For general technical chat, the backend uses a FAISS vector index built from an audio-domain knowledge base. The current data folder includes audio engineering reference material such as:

- WASAPI overview.
- Audio app development notes.
- MMDevice API details.
- IAudioClient3 details.
- APO architecture.
- Audio engine pipeline information.

The backend embeds user queries, retrieves relevant chunks, injects retrieved context into the prompt, and asks the configured model to generate a concise answer. This improves answer relevance for audio-domain questions compared with a plain chatbot.

### 6. Streaming Chat Responses

The chat routes support Server-Sent Events (SSE) streaming. Responses can be streamed character by character or token by token, depending on the route and model behavior. This improves perceived responsiveness in the frontend and allows the user to see progress while the backend is generating an answer.

### 7. Persistence and User Sessions

The backend integrates with Supabase for:

- User registration.
- Login.
- JWT-based sessions.
- Remember-me token expiry.
- Chat sessions.
- Chat messages.
- File storage.
- Uploaded and generated document URLs.

The backend also supports guest sessions for temporary access. Guest users can interact with the chatbot without creating a permanent account, while logged-in users can maintain persistent chat history.

---

## Technical Architecture

The backend is implemented as a Python FastAPI application.

### Main Application Layer

The FastAPI app registers three main route groups:

- /auth for login, registration, session validation, and guest login.
- /chats for chat, streaming, chat history, session creation, search, clear, delete, and logout workflows.
- /srs for direct SRS extraction, SRS generation, and generated DOCX download.

The root endpoint provides a simple health check indicating that the backend is running.

### AI and Retrieval Layer

The AI layer supports multiple inference paths:

- Ollama-compatible chat and extractor models.
- Optional llama.cpp backend using locally downloaded Hugging Face model artifacts.
- Optional dedicated PDF extractor model.
- Optional Google Colab FastAPI service for Qwen2.5 7B SRS generation.

The retrieval layer uses:

- sentence-transformers for embeddings.
- all-MiniLM-L6-v2 as the embedding model.
- FAISS for vector search.
- A local faiss_db directory for persisted index and metadata.
- Audio-domain source files in the data directory.

### Requirement Flow Layer

The requirement workflow is driven by questions.json and FlowManager. This keeps the guided conversation configurable and separates flow data from the API code.

The flow engine supports:

- Option matching.
- Typo-tolerant service triggers.
- PDF/document-first handling.
- Service-aware required fields.
- Product-specific contexts.
- Numbered summaries.
- User edits to captured answers.
- Query collection.
- Final PDF export.

### SRS Intelligence Layer

The SRS generator is organized around:

- DocumentParser for PDF, DOCX, DOC, and TXT parsing.
- RequirementExtractor for section, table, text, and metadata extraction.
- SRSValidator for missing-field and placeholder validation.
- fill_missing_srs_fields for contextual enrichment.
- SrsTemplateRenderer for DOCX rendering.
- LocalAISRSGenerator for optional AI-regenerated SRS output.
- build_srs_comparison for comparing generated versions.

This modular design helps separate parsing, extraction, validation, enrichment, rendering, and comparison.

### Storage and Persistence Layer

Supabase is used for persistent application data and uploaded/generated file storage. The backend can also maintain in-memory guest sessions with timeout pruning.

For generated documents, the backend stores:

- Requirement PDFs.
- Uploaded source documents.
- Generated SRS DOCX files.
- SRS JSON extraction results.
- Supabase file URLs.
- Local output paths for server-side downloads.

---

## End-to-End Workflow

### Workflow A: Guided Requirement Collection Without Upload

1. User starts the chat and asks for services or requirement help.
2. Backend detects the service intent and starts the requirement flow.
3. User answers structured questions based on service type.
4. Backend stores each answer in session context.
5. Backend shows a numbered summary.
6. User can edit any captured answer.
7. Backend requests customer name if needed.
8. Backend generates and uploads a final requirements PDF.
9. Chat history and attachment metadata are persisted.

### Workflow B: Requirement Document Upload to SRS

1. User uploads a PDF, DOC, DOCX, or TXT requirement document.
2. Backend validates and extracts document text.
3. Uploaded file is saved to Supabase storage.
4. SRS intelligence pipeline parses the source document.
5. Structured requirements are extracted and validated.
6. Missing fields are completed using contextual inference.
7. Backend renders the SRS into the master DOCX template.
8. Generated DOCX and structured JSON are saved.
9. User is offered an optional local AI SRS regeneration.
10. User can compare baseline and AI-generated SRS versions.
11. User selects the final SRS version.

---

## Use of Free and Open-Source Resources

One of the most important design decisions in SwarAI is the use of free and open-source resources wherever practical. This helps control cost during development, experimentation, and deployment.

The backend uses or supports:

- FastAPI for API development.
- Uvicorn for ASGI serving.
- FAISS CPU for local vector search.
- SentenceTransformers for embeddings.
- Ollama-compatible models for local AI inference.
- Qwen2.5 for optional SRS generation.
- llama.cpp through llama-cpp-python for local model execution.
- Hugging Face model artifacts for configurable local model loading.
- Google Colab T4 GPU runtime as an optional free/low-cost acceleration path for SRS regeneration.
- pypdf, pdfplumber, PyMuPDF, python-docx, docxtpl, and ReportLab for document extraction and generation.
- Supabase for backend data and storage integration.

This approach helps reduce paid API dependency and enables the team to prototype AI-powered requirement workflows without committing early to expensive commercial AI infrastructure.

For a public company website, the value can be stated as:

> SwarAI was engineered with a cost-conscious AI architecture, using local models, open-source retrieval, and optional free GPU resources to reduce dependency on paid AI services while accelerating requirement generation.

---

## Time-Saving Impact

The backend is designed to save engineering and business-analysis time in several ways.

### Reduced Discovery Time

The guided flow standardizes the requirement intake process. Instead of manually remembering every question for porting, optimization, or audio application projects, the chatbot asks the relevant questions in sequence.

### Reduced Document Review Effort

Uploaded requirement documents can be parsed automatically. The backend attempts to identify requirement IDs, requirement tables, section hierarchy, metadata, assumptions, references, definitions, and missing fields.

### Faster SRS Drafting

The system can generate an SRS DOCX directly from extracted requirement data. This reduces the manual effort needed to prepare a first SRS draft from customer inputs.

### Faster Review Cycles

Generated summaries and numbered answer maps allow users to review and modify captured information before final submission. The baseline-vs-AI SRS comparison further helps teams evaluate output quality before choosing a final version.

### Reusable Knowledge

The FAISS index lets the backend reuse existing audio-domain knowledge while answering technical questions. This makes the chatbot more useful for repeated engineering discussions.

If exact metrics are available later, the following website-ready statement can be quantified:

> By automating guided requirement collection, document extraction, and first-draft SRS generation, SwarAI can reduce the time required to move from raw customer inputs to a structured requirement artifact.

Recommended measurable KPIs:

- Average time to collect requirements before and after SwarAI.
- Average time to generate first SRS draft.
- Number of manual clarification cycles reduced.
- Percentage of uploaded documents converted into structured SRS output.
- Number of paid AI API calls avoided during prototyping.
- Cost saved by using local/open-source models and free Colab resources.

---

## Cost-Saving Impact

SwarAI reduces cost exposure primarily through architecture choices.

### Lower AI Inference Cost

The backend can use local or Ollama-compatible models instead of paid hosted LLM APIs. For development and internal use cases, this can avoid recurring per-token charges.

### Free GPU Experimentation

The optional Google Colab FastAPI workflow enables Qwen2.5 7B SRS generation on a Colab T4 GPU. This provides a practical way to test stronger local-style generation without buying dedicated GPU infrastructure at the prototype stage.

### Open-Source Retrieval Stack

FAISS and SentenceTransformers provide retrieval-augmented generation without a paid vector database requirement.

### Automated Document Generation

Automating PDF and DOCX generation reduces manual documentation effort and helps teams produce consistent artifacts faster.

### Reusable Backend Workflow

Because the requirement flow is configurable and the SRS generator is modular, the same backend can be adapted for new service lines or document templates with less rework.

A professional website-safe statement is:

> SwarAI lowers prototyping and operational cost by combining open-source AI infrastructure, local model execution, free-resource experimentation, and automated documentation workflows.

---

## Security and Data Management

The backend includes several practical security and data-management features:

- JWT-based authentication.
- Password hashing with bcrypt/passlib.
- Configurable token expiry.
- Remember-me support.
- Guest login with temporary session behavior.
- CORS configuration.
- Supabase-backed user and chat data.
- File upload validation.
- Generated document download protection against path traversal.
- Environment-variable configuration for credentials and model endpoints.

For production deployment, the following practices are recommended:

- Use strong SECRET_KEY values.
- Restrict CORS_ORIGINS to approved frontend domains.
- Keep Supabase keys and model API keys outside source control.
- Add request size and upload scanning policies.
- Enable HTTPS at the deployment layer.
- Add centralized logging and monitoring.
- Define data-retention rules for uploaded customer documents.
- Review generated SRS output before customer delivery.

---

## Deployment Model

The backend can run locally, in a virtual environment, or inside a Docker-capable environment. The application is served through Uvicorn and can preload models at startup unless model preload is disabled through environment configuration.

The backend supports configurable runtime behavior through environment variables, including:

- Supabase connection settings.
- JWT and token expiry settings.
- CORS settings.
- FAISS and data directories.
- Ollama endpoint and model names.
- llama.cpp model configuration.
- PDF extractor model configuration.
- Colab SRS API endpoint and API key.
- SRS generation timeout and context settings.

This makes the same backend adaptable for local development, internal demos, customer pilots, and production hardening.

---

## Business Value

SwarAI provides value to both engineering teams and business stakeholders.

For engineering teams:

- Reduces repetitive requirement questioning.
- Creates a structured intake process.
- Supports document-first extraction.
- Provides audio-domain retrieval assistance.
- Generates SRS artifacts faster.
- Preserves requirement evidence and metadata.

For business teams:

- Speeds up pre-sales and discovery workflows.
- Creates professional customer-facing requirement outputs.
- Reduces dependency on paid AI services during prototyping.
- Helps standardize requirement collection across projects.
- Provides a scalable foundation for additional engineering domains.

For customers:

- Provides a smoother requirement intake experience.
- Reduces repeated clarification requests.
- Produces clearer documentation earlier in the engagement.
- Enables faster movement from discussion to proposal or implementation.

---

## Differentiators

SwarAI is differentiated by combining several capabilities in one backend:

- Domain-specific audio engineering requirement flow.
- Retrieval-augmented technical chat.
- Document upload and extraction.
- SRS-specific parsing and validation.
- DOCX rendering into a master template.
- Optional local AI SRS regeneration.
- Baseline-vs-AI comparison.
- Cost-conscious use of free and open-source resources.
- Supabase-backed persistence and file storage.
- Streaming-ready API responses for frontend integration.

The result is not just a chatbot. It is a backend workflow engine for requirement generation and documentation automation.

---

## Suggested Website Positioning

The following text can be reused on a company website:

> SwarAI is an AI-assisted requirement generation platform built for audio engineering workflows. It guides users through structured requirement discovery, extracts requirements from uploaded documents, generates professional requirement PDFs and SRS DOCX files, and uses retrieval-augmented AI to answer technical questions. Built with open-source tools, local model support, FAISS retrieval, and optional free GPU resources, SwarAI helps reduce requirement-generation time while controlling AI infrastructure cost.

Short version:

> SwarAI accelerates audio requirement collection and SRS generation using guided chat, document extraction, local AI, and open-source retrieval.

---

## Future Enhancements

Potential future improvements include:

- Admin UI for editing requirement flows.
- Multi-template SRS generation.
- Human approval workflow before final document release.
- Requirement traceability dashboard.
- Role-based access control.
- Analytics for time saved and document-generation success rate.
- Automated test-case generation from SRS requirements.
- Integration with ticketing, CRM, or project-management tools.
- Deployment monitoring and model performance dashboards.

---

## Conclusion

The SwarAI backend demonstrates how an engineering chatbot can go beyond simple question answering. By combining guided requirement intake, document extraction, retrieval-augmented generation, local/open-source AI models, automated PDF/DOCX generation, and SRS comparison, it creates a practical foundation for faster and more cost-effective requirement generation.

The backend is especially valuable because it is designed around real engineering workflows: collecting the right information, structuring it, validating it, generating reusable artifacts, and reducing manual documentation effort. Its use of free and open-source resources also makes it cost-conscious, allowing teams to experiment and deliver AI-assisted requirement workflows without depending entirely on paid AI services.

---

## Metric Placeholders for Final Publication

Before publishing, replace the placeholders below with measured internal numbers if available:

- Requirement collection time before SwarAI: [insert baseline]
- Requirement collection time after SwarAI: [insert measured result]
- First SRS draft preparation time before SwarAI: [insert baseline]
- First SRS draft preparation time after SwarAI: [insert measured result]
- Estimated monthly paid AI API cost avoided during prototype: [insert amount]
- Free/open-source resources used: FastAPI, FAISS, SentenceTransformers, Ollama-compatible models, Qwen2.5, Google Colab, Python document libraries
- Overall impact statement: [insert verified percentage or qualitative statement]

