from langchain_community.vectorstores import FAISS
from manual_embedder import ManualEmbedder
from vectorstore_utils import save_vectorstore, load_vectorstore, vectorstore_exists, get_documents_metadata
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

import os
import time
import collections
import subprocess
import json

# Função para formatar o tempo em minutos e segundos
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} segundos"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    if remaining_seconds == 0:
        return f"{minutes} minuto(s)"
    return f"{minutes} minuto(s) e {remaining_seconds:.2f} segundos"

# Etapa 1 – Carregar documentos
def load_documents():
    docs = []
    file_processing_summary = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "documents")
    print(f"  📂 Lendo arquivos de: {path}")
    for file_name in os.listdir(path):
        full_path = os.path.join(path, file_name)
        file_size_kb = os.path.getsize(full_path) / 1024
        loaded_file_docs = []
        file_type = None
        items_loaded = 0

        if file_name.endswith(".txt"):
            loader = TextLoader(full_path, autodetect_encoding=True)
            loaded_file_docs = loader.load()
            file_type = 'TXT'
            items_loaded = len(loaded_file_docs)
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(full_path)
            loaded_file_docs = loader.load()
            file_type = 'PDF'
            items_loaded = len(loaded_file_docs) # PyPDFLoader loads one doc per page
        
        if file_type:
            docs.extend(loaded_file_docs)
            file_processing_summary.append({
                'filename': file_name,
                'type': file_type,
                'pages_or_items': items_loaded,
                'size_kb': file_size_kb
            })
            print(f"    ➕ Arquivo '{file_name}' ({file_type}, {items_loaded} {'páginas' if file_type == 'PDF' else 'itens'}, {file_size_kb:.2f} KB) adicionado.")
        else:
            print(f"    ⚠️ Arquivo '{file_name}' ignorado (tipo não suportado).")
            
    return docs, file_processing_summary

# Etapa 2 – Separar textos em pedaços
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Etapa 3 – Criar embeddings e FAISS
def create_vectorstore(docs):
    import faiss
    import numpy as np
    embedder = ManualEmbedder("nomic-ai/nomic-embed-text-v1")
    meta = get_documents_metadata(docs)
    # Se já existe e os metadados não mudaram, carrega do disco
    if vectorstore_exists():
        try:
            index, docs_saved, meta_saved = load_vectorstore()
            if meta_saved == meta:
                print("✔️ Vetorstore carregado do disco.")
                return {"index": index, "docs": docs_saved, "embedder": embedder}
            else:
                print("🔄 Mudanças detectadas nos arquivos. Recriando vetorstore...")
        except Exception as e:
            print(f"⚠️ Erro ao carregar vetorstore: {e}. Recriando.")
    # Extrai o texto de cada documento LangChain
    texts = [doc.page_content for doc in docs]
    vectors = embedder.embed_documents(texts)
    dim = vectors.shape[1] if len(vectors.shape) > 1 else len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype=np.float32))
    save_vectorstore(index, docs, meta)
    print("💾 Vetorstore salvo em disco.")
    return {"index": index, "docs": docs, "embedder": embedder}


# Etapa 4 – Criar cadeia RAG
def search_similar_docs(vectorstore, query, k=4):
    embedder = vectorstore["embedder"]
    index = vectorstore["index"]
    docs = vectorstore["docs"]
    query_vec = embedder.embed_query(query)
    import numpy as np
    D, I = index.search(np.array([query_vec], dtype=np.float32), k)
    return [docs[i] for i in I[0]]

from langchain_ollama import OllamaLLM

def manual_qa(llm, vectorstore, query):
    # Busca os documentos mais similares
    docs = search_similar_docs(vectorstore, query, k=4)
    # Concatena o conteúdo dos documentos
    context = "\n".join([doc.page_content for doc in docs])
    # Monta o prompt para o LLM
    prompt = f"Contexto:\n{context}\n\nPergunta: {query}\nResposta:"
    return llm.invoke(prompt)


# Etapa 5 – Interface principal
def main():
    print("🚀 Iniciando o sistema RAG...")

    # Carregar Documentos
    print("\n مرحله ۱: Carregando documentos...")
    start_time = time.time()
    docs, file_summary = load_documents()
    load_time = time.time() - start_time
    
    if not docs:
        print("❌ Nenhum documento foi carregado. Verifique a pasta 'documents'. Encerrando.")
        return

    print(f"\n📊 Métricas de Carregamento de Documentos:")
    print(f"  ⏱️ Tempo de carregamento: {format_time(load_time)}")
    total_files = len(file_summary)
    total_pages_items = sum(f['pages_or_items'] for f in file_summary)
    total_size_kb = sum(f['size_kb'] for f in file_summary)
    print(f"  📄 Total de arquivos processados: {total_files}")
    file_types_count = collections.Counter(f['type'] for f in file_summary)
    for f_type, count in file_types_count.items():
        print(f"     - {f_type}: {count} arquivo(s)")
    print(f"  📖 Total de itens (páginas para PDF, documentos para TXT): {total_pages_items}")
    print(f"  💾 Tamanho total dos arquivos: {total_size_kb:.2f} KB")

    # Fragmentar Textos
    print("\n مرحله ۲: Fragmentando textos...")
    start_time = time.time()
    split_docs = split_documents(docs)
    split_time = time.time() - start_time
    print(f"\n📊 Métricas de Fragmentação de Textos:")
    print(f"  ⏱️ Tempo de fragmentação: {format_time(split_time)}")
    print(f"  🧩 Total de fragmentos criados: {len(split_docs)}")

    # Criar Base Vetorial
    print("\n مرحله ۳: Criando/Carregando base vetorial FAISS...")
    start_time = time.time()
    db = create_vectorstore(split_docs)
    vectorstore_time = time.time() - start_time
    print(f"\n📊 Métricas da Base Vetorial:")
    print(f"  ⏱️ Tempo de criação/carregamento: {format_time(vectorstore_time)}")

    # Função para verificar e baixar o modelo Ollama
def ensure_ollama_model_available(model_name):
    print(f"  🔍 Verificando disponibilidade do modelo Ollama: {model_name}...")
    try:
        # Verificar se o Ollama CLI está acessível
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"  ⚠️ Erro ao acessar o Ollama CLI. Certifique-se de que o Ollama está instalado e no PATH.")
            print(f"     Detalhe do erro: {e}")
            print(f"  ↪️  Tentando prosseguir sem verificação/download automático do modelo.")
            return False # Indica que não pôde verificar/baixar

        # Listar modelos instalados
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True, timeout=30)
        installed_models = []
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1: # A primeira linha é o cabeçalho
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        installed_models.append(parts[0]) # O nome do modelo é a primeira parte
        
        # Adicionar 'latest' se o nome do modelo não tiver tag explícita
        # Ex: 'llama3' é o mesmo que 'llama3:latest'
        model_name_with_tag = model_name if ':' in model_name else f"{model_name}:latest"

        if model_name_with_tag in installed_models:
            print(f"  👍 Modelo '{model_name_with_tag}' já está disponível localmente.")
            return True
        else:
            print(f"  ⚠️ Modelo '{model_name_with_tag}' não encontrado localmente.")
            print(f"  📥 Tentando baixar o modelo '{model_name}'... Isso pode levar alguns minutos.")
            try:
                process = subprocess.Popen(["ollama", "pull", model_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                # Imprimir output em tempo real
                if process.stdout:
                    for line in iter(process.stdout.readline, ''):
                        print(f"     {line.strip()}")
                
                process.wait(timeout=600) # Timeout de 10 minutos para o download
                
                if process.returncode == 0:
                    print(f"  ✅ Modelo '{model_name}' baixado com sucesso!")
                    return True
                else:
                    print(f"  ❌ Falha ao baixar o modelo '{model_name}'.")
                    if process.stderr:
                        for line in iter(process.stderr.readline, ''):
                            print(f"     Erro: {line.strip()}")
                    print(f"     Verifique o nome do modelo e sua conexão com a internet.")
                    print(f"     Você pode tentar baixar manualmente com: ollama pull {model_name}")
                    return False
            except FileNotFoundError:
                print("  ❌ Comando 'ollama' não encontrado. Certifique-se de que Ollama está instalado e no PATH.")
                return False
            except subprocess.TimeoutExpired:
                print(f"  ❌ Timeout ao tentar baixar o modelo '{model_name}'. O download demorou mais de 10 minutos.")
                return False
            except Exception as e:
                print(f"  ❌ Erro inesperado ao tentar baixar o modelo '{model_name}': {e}")
                return False
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Erro ao executar comando Ollama: {e}")
        print(f"     Saída: {e.stdout}")
        print(f"     Erro: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  ❌ Comando 'ollama' não encontrado. Certifique-se de que Ollama está instalado e no PATH.")
        return False
    except Exception as e:
        print(f"  ❌ Erro inesperado durante a verificação do modelo Ollama: {e}")
        return False

# Iniciar RAG com Ollama
    print("\n مرحله ۴: Configurando o modelo LLM...")
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    
    print(f"  ℹ️ Modelo Ollama selecionado: {model_name}.")
    if not ensure_ollama_model_available(model_name):
        print("  ⚠️ Não foi possível confirmar ou baixar o modelo. O sistema tentará usar o modelo assim mesmo.")
        print("     Se ocorrerem erros, verifique a instalação do Ollama e a disponibilidade do modelo.")
    
    llm = OllamaLLM(model=model_name)
    print("✔️ Sistema RAG pronto para receber perguntas.")

    while True:
        query = input("\n❓ Pergunta (ou 'sair' para terminar): ")
        if query.lower() in ['sair', 'exit', 'quit']:
            print("👋 Até logo!")
            break
        if not query.strip():
            print("⚠️ Por favor, digite uma pergunta.")
            continue
            
        print("⏳ Processando sua pergunta...")
        qa_start_time = time.time()
        answer = manual_qa(llm, db, query)
        qa_time = time.time() - qa_start_time
        print(f"💡 Resposta:")
        print(answer)
        print(f"  ⏱️ Tempo para gerar resposta: {format_time(qa_time)}")


if __name__ == "__main__":
    main()
