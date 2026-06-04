import os # Biblioteca para manipulação de caminhos e variáveis de sistema operacionais.
import time # Biblioteca essencial para capturar as métricas de latência (TTFT, ITL, Total).
import pandas as pd # Usada para manipulação de DataFrames e exportação dos resultados.
import csv # Necessária para ler o arquivo de entrada com o quoting correto (QUOTE_NONNUMERIC).
import logging # Usada para registrar eventos no terminal em vez de usar 'print', padrão ouro em pipelines.
import requests # Usada no TutorInteligenteRunner para fazer requisições HTTP POST para a sua API local.
from typing import Dict, Any, List, Optional # Tipagem estática para tornar o código mais legível e seguro.
from dotenv import load_dotenv # Importa a função para carregar credenciais do arquivo .env sem expô-las no código.
from openai import OpenAI # SDK oficial da OpenAI, usado também para o DeepSeek e vLLM (pois são compatíveis).

# Carrega as variáveis do arquivo .env da raiz do projeto (onde estão as API Keys).
load_dotenv(dotenv_path="../src/.env")

# Configuração global de log: define o nível mínimo (INFO) e o formato (Data/Hora - Nível - Mensagem).
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# PAINEL DE CONTROLE CENTRAL DO EXPERIMENTO
# Altere as configurações apenas neste bloco!
# ==============================================================================

# 1. TIPO DE TESTE
# Define qual classe Runner será instanciada: "bruto" (BenchmarkRunner) ou "tutor" (TutorInteligenteRunner).
TIPO_TESTE = "bruto"

# 2. MODELO ATIVO (Descomente apenas 1 por vez)
# Variável que dita o modelo que será testado. As lógicas de provider e temperatura dependerão dessa string.
#MODELO_ATIVO = "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-8bit"
#MODELO_ATIVO = "gpt-5"
#MODELO_ATIVO = "gpt-4.1-nano"
MODELO_ATIVO = "deepseek-v4-pro"

# 3. ARQUIVO DE ENTRADA (Descomente apenas 1 por vez)
# Caminho para o dataset de perguntas. O sufixo "test" no nome será usado depois para nomear o CSV de saída.
ARQUIVO_ENTRADA = "dados_entrada/questions_PDS_final.txt"        # Teste Completo (120 perguntas totais)
#ARQUIVO_ENTRADA = "dados_entrada/questions_PDS_final_test.txt"   # Teste Rápido de Validação (4 perguntas)

# 4. IDIOMAS A PROCESSAR
# Lista de tuplas indicando a sigla do idioma e a respectiva coluna no arquivo de texto.
COLUNAS_PROCESSAR = [("PT", "pergunta_pt"), ("EN", "pergunta_en")]

# 5. PARÂMETROS GLOBAIS DE INFERÊNCIA
SEED_GLOBAL = 42 # Semente determinística para tentar manter as respostas reprodutíveis.
TEMP_GLOBAL = 0.15 # Temperatura baixa (0.15) para reduzir a aleatoriedade, favorecendo testes de RAG/Exatos.
# ==============================================================================

class BenchmarkRunner:
    """Gerencia a execução dos testes com o Modelo Bruto (Direto na API/vLLM)."""
    
    def __init__(self, model_name: str):
        # Construtor: Inicializa a classe guardando o nome do modelo.
        self.model_name = model_name
        # Descobre automaticamente qual API chamar (OpenAI, DeepSeek ou vLLM) baseando-se no nome.
        self.provider = self._discover_provider(model_name)
        # Cria a conexão com a API escolhida.
        self.client = self._initialize_client()

    def _discover_provider(self, model: str) -> str:
        # Põe o nome em minúsculas para não falhar na string matching.
        model_lower = model.lower()
        # Se for GPT ou a família de raciocínio 'o1'/'o3', o provedor é openai.
        if "gpt" in model_lower or "o1" in model_lower: return "openai"
        # Se contiver deepseek, roteia para a API do DeepSeek.
        if "deepseek" in model_lower: return "deepseek"
        # Caso contrário (ex: Ministral), assume que é o modelo local rodando no servidor vLLM.
        return "vllm"

    def _initialize_client(self) -> OpenAI:
        # Configura o cliente da API de acordo com o provedor descoberto.
        if self.provider == "vllm":
            # Conecta na porta 8006 do servidor local (a key é dummy pois é local).
            return OpenAI(api_key="vllm-dummy-key", base_url="http://10.10.80.238:8006/v1")
        elif self.provider == "openai":
            # Puxa a chave da OpenAI do .env (base_url padrão implícita).
            api_key = os.getenv("OPENAI_API_KEY")
            return OpenAI(api_key=api_key)
        elif self.provider == "deepseek":
            # Puxa a chave do DeepSeek do .env e aponta para a URL customizada deles.
            api_key = os.getenv("DEEPSEEK_API_KEY")
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
        else:
            # Proteção caso caia em um provedor não mapeado.
            raise ValueError(f"Provedor {self.provider} não suportado.")

    def _build_streaming_kwargs(self, prompt: str) -> Dict[str, Any]:
        # Dicionário dinâmico com os parâmetros da requisição para a API (kwargs).
        kwargs = {
            "model": self.model_name, # O nome exato do modelo exigido pelo provedor.
            "messages": [{"role": "user", "content": prompt}], # Estrutura padrão de chat.
            "seed": SEED_GLOBAL, # Puxando a semente determinística do Painel de Controle.
            "stream": True, # Ativa o streaming (receber token a token em vez de esperar tudo de uma vez).
            "stream_options": {"include_usage": True} # FATO: Essencial para receber os metadados (tokens) no último chunk da rede.
        }
        model_lower = self.model_name.lower()
    
        # Lista de modelos de RACIOCÍNIO que NÃO suportam manipulação de temperature.
        modelos_sem_temperature = ["o1", "o3", "gpt-5"]
    
        # FATO (Trava de Segurança): Se o modelo não for de raciocínio bloqueado, adiciona a temperatura.
        # Se você mandasse temperature para o GPT-5, a requisição daria erro (Crash).
        if not any(modelo in model_lower for modelo in modelos_sem_temperature):
            kwargs["temperature"] = TEMP_GLOBAL # Adiciona a temperatura de 0.15
            print(f"[DEBUG] Enviando para API: seed={kwargs.get('seed')}, temp={kwargs.get('temperature')}")
        else:
            # Omite a temperatura para respeitar a arquitetura do modelo de raciocínio.
            print(f"[DEBUG] Enviando para API: seed={kwargs.get('seed')}, temp=NÃO ENVIADO (modelo não suporta)")
    
        return kwargs # Retorna o dicionário de parâmetros montado.

    def run_warmup(self):
        # Envia uma requisição boba ("ok") para acordar o modelo, carregar VRAM e fazer handshake TLS.
        logging.info(f"Aquecendo o modelo {self.model_name} (Cold Start)...")
        try:
            kwargs = self._build_streaming_kwargs("Responda apenas 'ok'.")
            kwargs["stream"] = False # Desliga o stream só para esse aquecimento ser mais rápido.
            kwargs.pop("stream_options", None) # Remove opção de uso pois não faremos streaming.
            self.client.chat.completions.create(**kwargs) # Dispara a requisição.
            logging.info("Aquecimento concluído. Prontos para as requisições.")
        except Exception as e:
            logging.error(f"Falha no aquecimento: {e}")

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        kwargs = self._build_streaming_kwargs(prompt) # Prepara os parâmetros.
        first_token_time, usage_data, full_text = None, None, "" # Inicializa as variáveis métricas.
        start_time = time.time() # ⏱️ Dispara o cronômetro oficial do início da requisição.
        
        try:
            response_stream = self.client.chat.completions.create(**kwargs) # Abre a conexão de rede em modo Stream.
            for chunk in response_stream: # Itera sobre cada pedaço (token) que chega da internet.
                
                # FATO CRÍTICO (TTFT): 'chunk.choices[0].delta.content' isola a string visível.
                # Como a API do GPT-5 e do DeepSeek não mandam raciocínio no 'content', este IF é ignorado enquanto eles pensam.
                # O cronômetro do first_token_time só vai bater quando a resposta final visível começar.
                if first_token_time is None and chunk.choices and chunk.choices[0].delta.content:
                    first_token_time = time.time() # ⏱️ Marca o tempo do primeiro token visual.
                
                # Se houver texto no chunk, vai concatenando na string full_text (Resposta Gerada).
                if chunk.choices and chunk.choices[0].delta.content:
                    full_text += chunk.choices[0].delta.content
                
                # FATO (Metadados da API): Captura os tokens do último chunk. 
                # Modelos de raciocínio INCLUEM os tokens invisíveis aqui, INFLANDO a variável!
                if chunk.usage is not None:
                    usage_data = chunk.usage
            
            total_time = time.time() - start_time # ⏱️ Latência Total (Tempo de silêncio + Tempo escrevendo).
            
            # FATO (Cálculo do TTFT): Tempo do 1º Token Visível menos o tempo de envio.
            # Aqui fica embutido todo o tempo de Raciocínio (Chain-of-Thought), podendo dar 60s ou mais.
            ttft = first_token_time - start_time if first_token_time else 0.0
            
            # Extrai os tokens de metadados reportados pela provedora.
            prompt_tokens = usage_data.prompt_tokens if usage_data else 0
            # FATO (A Inversão): Para o Modo Bruto, isso traz Texto Visível + Tokens de Raciocínio Ocultos.
            completion_tokens = usage_data.completion_tokens if usage_data else 0
            total_tokens = usage_data.total_tokens if usage_data else 0
            
            # FATO (Cálculo do ITL): Subtrai o TTFT do Tempo Total.
            # Como o TTFT engoliu o tempo de pensar, o ITL_s fica restrito APENAS ao tempo físico da digitação do texto visual.
            itl = (total_time - ttft) if completion_tokens > 1 else 0.0  
            
            # FATO (A Ilusão Matemática): Divide Tokens INFLADOS (Texto+Raciocínio) pelo ITL (Tempo Curto da escrita visual).
            # Isso gerou valores absurdos como 344 Tokens por segundo, distorcendo os gráficos brutos!
            throughput = completion_tokens / (total_time - ttft) if (total_time - ttft) > 0 else 0.0  
            
        except Exception as e:
            logging.error(f"Erro na inferência bruta: {e}") # Trata quedas de conexão ou recusas da API.
            return self._error_dict(e) # Retorna dicionário zerado para não quebrar o loop.

        # Retorna todas as métricas formatadas para virar uma linha no CSV.
        return self._format_dict(full_text, total_time, ttft, itl, throughput, prompt_tokens, completion_tokens, total_tokens)

    def _error_dict(self, error):
        # Helper para padronizar linhas com falha na rede (evita crash do DataFrame).
        return self._format_dict(f"ERRO: {error}", 0, 0, 0, 0, 0, 0, 0)
        
    def _format_dict(self, text, total_t, ttft, itl, tps, t_in, t_out, t_tot):
        # Monta o dicionário que vira as colunas finais. O round() limpa as casas decimais.
        return {
            "Resposta_Gerada": text, "Latencia_Total_s": round(total_t, 4),
            "TTFT_s": round(ttft, 8), "ITL_s": round(itl, 8),
            "Throughput_tps": round(tps, 2), "Tokens_In": t_in,
            "Tokens_Out": t_out, "Tokens_Total": t_tot
        }


class TutorInteligenteRunner(BenchmarkRunner):
    """Gerencia a execução dos testes no Backend do Tutor via Streaming API (RAG)."""
    
    def __init__(self, model_name: str):
        # Construtor do RAG: Herda do BenchmarkRunner, mas aponta para o seu servidor local (FastAPI/Django etc).
        self.model_name = model_name
        self.session_id = f"benchmark-{int(time.time())}" # Cria um ID de sessão único para o RAG não confundir os logs.
        self.api_url = f"http://127.0.0.1:8000/api/v1/chat/{self.session_id}?stream=true" # Endpoint local do seu RAG.

    def run_warmup(self):
        # Chama o backend local para aquecer as instâncias (subir vector db, etc).
        logging.info(f"Aquecendo o Tutor Inteligente...")
        try:
            payload = {"query": "ok", "model": self.model_name}
            requests.post(self.api_url, json=payload, timeout=60) # Timeout de 60s por precaução.
            logging.info("Aquecimento do Tutor concluído.")
        except Exception as e:
            logging.warning(f"Aviso no warmup do Tutor (API está no ar?): {e}")

    def generate_response(self, prompt: str) -> Dict[str, Any]:
        start_time = time.time() # ⏱️ Dispara cronômetro oficial do modo RAG.
        first_token_time = None
        full_text = "" # Vai guardar o texto limpo retornado pela sua API local.
        
        # Puxando o SEED e TEMP diretamente do Painel de Controle Global para padronizar.
        payload = {
            "query": prompt,
            "model": self.model_name,
            "seed": SEED_GLOBAL
        }

        # Trava de segurança idêntica ao Modo Bruto: remove a temperature se for modelo de raciocínio.
        modelos_sem_temperature = ["o1", "o3", "gpt-5"]
        if not any(modelo in self.model_name.lower() for modelo in modelos_sem_temperature):
            payload["temperature"] = TEMP_GLOBAL

        try:
            # Faz a chamada POST nativa do python pedindo stream=True. Timeout de 10 min.
            response = requests.post(self.api_url, json=payload, stream=True, timeout=600)
            response.raise_for_status() # Verifica se deu erro HTTP 500/400.
            
            # FATO (Recepção do Stream no Backend): Como o seu RAG limpa as tags de raciocínio, 
            # ele fica mudo segurando a conexão HTTP, e só despacha pedaços (chunks) de letras visíveis.
            for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                if chunk:
                    # ⏱️ Marca o momento em que a 1ª letra final caiu na rede (TTFT alto devido ao raciocínio oculto).
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_text += chunk # Concatena a resposta legível pelo aluno.
            
            total_time = time.time() - start_time # Latência end-to-end do RAG.
            ttft = first_token_time - start_time if first_token_time else 0.0
            
            # FATO (A Raiz da Diferença): Como a conexão HTTP via requests.post não fornece metadados,
            # o script estimou os tokens usando (Caracteres Visíveis / 4). 
            # Isso descartou o raciocínio oculto, gerando Tokens_Out super baixos comparados ao Bruto!
            completion_tokens = max(1, len(full_text) // 4)
            prompt_tokens = max(1, len(prompt) // 4)
            
            # O ITL reflete a escrita limpa.
            itl = (total_time - ttft) if completion_tokens > 1 else 0.0  
            
            # FATO (Throughput Realista): Diferente do Bruto, dividiu tokens visíveis limpos pelo tempo de escrita limpo.
            # Resultado: A velocidade fez sentido físico (ex: ~30 TPS no Tutor vs 344 TPS no Bruto).
            throughput = completion_tokens / (total_time - ttft) if (total_time - ttft) > 0 else 0.0  

            return self._format_dict(
                full_text, total_time, ttft, itl, throughput, 
                prompt_tokens, completion_tokens, prompt_tokens + completion_tokens
            )
            
        except Exception as e:
            logging.error(f"Erro na comunicação com o Tutor: {e}")
            return self._error_dict(e)


def process_dataset():
    """Lê o dataset, instancia o Runner correto e exporta os resultados dinamicamente."""
    os.makedirs("resultados_brutos", exist_ok=True) # Cria a pasta se ela não existir.
    
    # Validação do Painel de Controle para evitar erros de digitação do usuário.
    if TIPO_TESTE not in ["bruto", "tutor"]:
        raise ValueError("TIPO_TESTE deve ser 'bruto' ou 'tutor'")
    
    if not COLUNAS_PROCESSAR:
        raise ValueError("COLUNAS_PROCESSAR não pode estar vazio. Defina pelo menos uma coluna.")
    
    # Lógica excelente para validar se a sigla/coluna existe antes de perder horas na API.
    df_check = pd.read_csv(ARQUIVO_ENTRADA, sep='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8', nrows=1)
    for idioma, coluna in COLUNAS_PROCESSAR:
        if coluna not in df_check.columns:
            raise ValueError(f"Coluna '{coluna}' não encontrada no arquivo. Colunas disponíveis: {list(df_check.columns)}")
    
    # Higieniza o nome do modelo (trocando / por _) para evitar quebra de rotas no Linux/Windows.
    safe_model_name = MODELO_ATIVO.replace("/", "_")
    colunas_sufixo = "_".join([col for _, col in COLUNAS_PROCESSAR])
    
    # Dinamicamente nomeia o CSV para refletir se foi o teste Rápido ou Completo.
    tipo_arquivo = "COMPLETO" if "test" not in ARQUIVO_ENTRADA else "TESTE"
    output_csv = f"resultados_brutos/geracao_{TIPO_TESTE}_{safe_model_name}_{colunas_sufixo}_{tipo_arquivo}.csv"
    
    logging.info(f"TESTE: {TIPO_TESTE.upper()} | MODELO: {MODELO_ATIVO} | DATASET: {tipo_arquivo}")
    logging.info(f"COLUNAS A PROCESSAR: {COLUNAS_PROCESSAR}")
    
    # Carrega todo o dataset do TXT. O quotechar evita quebra de linha maluca no meio da pergunta.
    df = pd.read_csv(ARQUIVO_ENTRADA, sep='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8')
    
    # Instancia a classe condicionalmente baseada no TIPO_TESTE.
    runner = BenchmarkRunner(MODELO_ATIVO) if TIPO_TESTE == "bruto" else TutorInteligenteRunner(MODELO_ATIVO)
    runner.run_warmup() # Aquece a rede.
    
    # Define as colunas do CSV final que vai salvar as respostas e as métricas.
    colunas = ["Idioma", "Interferencia_Tutor", "Capitulo", "Num_Questao", "Pergunta", "Modelo", 
               "Resposta_Gerada", "Latencia_Total_s", "TTFT_s", "ITL_s", 
               "Throughput_tps", "Tokens_In", "Tokens_Out", "Tokens_Total"]
    pd.DataFrame(columns=colunas).to_csv(output_csv, index=False, encoding='utf-8') # Cria o arquivo zerado com cabeçalhos.
    
    # Traduz para o CSV se o RAG agiu ou não na requisição.
    status_tutor = "NÃO" if TIPO_TESTE == "bruto" else "SIM"
    
    # Laço Duplo: Itera primeiro pelas colunas de Idioma (PT, EN) e depois pelas linhas do Dataset.
    for idioma, coluna_pergunta in COLUNAS_PROCESSAR:
        logging.info(f"=== INICIANDO BLOCO: {idioma} (Coluna: {coluna_pergunta}) ===")
        
        for index, row in df.iterrows():
            pergunta = row[coluna_pergunta]
            
            # Pula linhas vazias geradas por artefatos no TXT.
            if pd.isna(pergunta) or str(pergunta).strip() == "":
                logging.warning(f"[{idioma}] Q{index+1} - Pergunta vazia ou inválida. Pulando...")
                continue
                
            logging.info(f"[{idioma}] Processando Q{index+1}/{len(df)} | Cap: {row['capitulo']}")
            
            # Manda a string para a classe correspondente processar na rede e devolver as métricas (m).
            m = runner.generate_response(pergunta)
            linha = {
                "Idioma": idioma,
                "Interferencia_Tutor": status_tutor,
                "Capitulo": row['capitulo'],
                "Num_Questao": row['numero_questao'],
                "Pergunta": pergunta,
                "Modelo": MODELO_ATIVO,
                "Resposta_Gerada": m["Resposta_Gerada"],
                "Latencia_Total_s": m["Latencia_Total_s"],
                "TTFT_s": m["TTFT_s"],
                "ITL_s": m["ITL_s"],
                "Throughput_tps": m["Throughput_tps"],
                "Tokens_In": m["Tokens_In"],
                "Tokens_Out": m["Tokens_Out"],
                "Tokens_Total": m["Tokens_Total"]
            }
            
            # SALVAMENTO INCREMENTAL: mode='a' e header=False anexam a linha no final do CSV sem reescrever.
            pd.DataFrame([linha]).to_csv(output_csv, mode='a', header=False, index=False, encoding='utf-8')
            time.sleep(0.5) # Respiro para não tomar Rate Limit 429 da provedora de API.
    
    logging.info(f"Benchmark concluído com sucesso! Salvo em: {output_csv}")

def listar_colunas_disponiveis():
    """Helper function para ver quais colunas existem no dataset. Útil para debug."""
    df = pd.read_csv(ARQUIVO_ENTRADA, sep='\t', quotechar='|', quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8', nrows=1)
    print("Colunas disponíveis no dataset:")
    for col in df.columns:
        print(f"  - {col}")

# Ponto de entrada padrão do Python. Só executa process_dataset se for chamado diretamente.
if __name__ == "__main__":
    process_dataset()