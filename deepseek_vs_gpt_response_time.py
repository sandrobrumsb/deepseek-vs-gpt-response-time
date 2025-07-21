from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
import os
import time

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Inicialização dos clientes das APIs com as chaves de ambiente
client_deepseek = OpenAIClient(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)

client_openai = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))


def to_load(file):
    """
    Lê o conteúdo de um arquivo e retorna como string.

    Parâmetros:
    file (str): caminho para o arquivo

    Retorno:
    str: conteúdo do arquivo
    """
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def create_prompt_chat(prompt_system, prompt_user):
    """
    Cria a estrutura da mensagem para o chat, com roles de system e user.

    Parâmetros:
    prompt_system (str): texto para o papel do sistema (instruções)
    prompt_user (str): texto para o papel do usuário

    Retorno:
    list: lista de mensagens formatadas para a API de chat
    """
    return [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_user},
    ]


def response_chat_with_timing(client, model, message):
    """
    Envia uma requisição de chat para a API, medindo o tempo de resposta.

    Parâmetros:
    client (OpenAIClient): cliente da API
    model (str): modelo a ser utilizado
    message (list): mensagens formatadas para a conversa

    Retorno:
    tuple: (resposta do modelo, tempo gasto em segundos)
    """
    start = time.time()  # Início da medição do tempo
    response = client.chat.completions.create(model=model, messages=message)
    end = time.time()  # Fim da medição do tempo
    duration = end - start
    return response.choices[0].message.content, duration


# Configuração dos prompts
prompt_system = """
Identifique o perfil de compra para cada cliente a seguir.
O formato de saída deve ser:
Cliente - descreva o perfil do cliente em 3 palavras.
"""

# Carrega o conteúdo do arquivo CSV que será usado como input do usuário
prompt_user = to_load(r"dados\lista_de_compras_20_clientes.csv")

# Cria a mensagem para enviar ao modelo
message = create_prompt_chat(prompt_system, prompt_user)

# Realiza a chamada para a API DeepSeek e mede o tempo de resposta
response_deepseek, time_deepseek = response_chat_with_timing(
    client_deepseek, "deepseek-chat", message
)

# Realiza a chamada para a API OpenAI GPT-3.5 e mede o tempo de resposta
response_openai, time_openai = response_chat_with_timing(
    client_openai, "gpt-3.5-turbo", message
)

# Exibe os resultados e os tempos
print(f"\n DeepSeek - Tempo de response: {time_deepseek:.2f} segundos\n")
print(response_deepseek)

print(f"\n OpenAI GPT - Tempo de response: {time_openai:.2f} segundos\n")
print(response_openai)
