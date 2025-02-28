import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def obter_anos_disponiveis():
    """Obtém a lista de anos disponíveis na pasta enem-essays"""
    try:
        base_dir = "database/enem-essays"
        anos = []
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)) and item.isdigit():
                if os.path.exists(os.path.join(base_dir, item, "subject.txt")):
                    anos.append(int(item))
        return sorted(anos)
    except Exception as e:
        logging.error(f"Erro ao obter anos disponíveis: {str(e)}")
        return []

def ler_tema(ano):
    """Lê o tema do arquivo subject.txt do ano especificado"""
    try:
        caminho_arquivo = f"database/enem-essays/{ano}/subject.txt"
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logging.error(f"Erro ao ler tema do ano {ano}: {str(e)}")
        return None

def gerar_prompts(tema):
    """Gera as variações de prompts para um tema"""
    aviso = "\n\nIMPORTANTE: Responda APENAS com o texto da redação, sem comentários iniciais ou finais, sem explicações e sem qualquer mensagem adicional."
    
    return [
        # Prompts Complexos
        # Variação 1 - Foco em repertório sociocultural
        f'''Me ajuda a fazer uma redação do ENEM sobre "{tema}"?
Queria uma redação com:
- Introdução com dados ou fatos históricos
- Desenvolvimento bem argumentado
- Conclusão com proposta de intervenção detalhada
- Linguagem formal, sem erros{aviso}''',

        # Variação 2 - Foco em argumentação
        f'''Pode escrever uma redação dissertativa argumentativa sobre o tema do ENEM: "{tema}"
Preciso que tenha:
- Contextualização na introdução
- Argumentos com exemplos
- Proposta de intervenção com agente, ação, meio, finalidade
- Conectivos e repertório{aviso}''',

        # Variação 3 - Foco em estrutura
        f'''Escreva uma redação estilo ENEM sobre "{tema}" seguindo essa estrutura:
- 1º parágrafo: introdução com contextualização
- 2º e 3º parágrafos: desenvolvimento com argumentos
- 4º parágrafo: conclusão com proposta de intervenção{aviso}''',

        # Variação 4 - Foco em atualidades
        f'''Preciso de uma redação do ENEM sobre "{tema}".
Use:
- Dados atuais na introdução
- Argumentos bem desenvolvidos
- Exemplos da realidade brasileira
- Conclusão com solução viável{aviso}''',

        # Variação 5 - Foco em citações
        f'''Pode fazer uma redação dissertativa sobre "{tema}"?
Queria que tivesse:
- Citação de algum autor ou pensador na introdução
- Desenvolvimento com dados e exemplos
- Conclusão com proposta detalhada
- Linguagem formal do ENEM{aviso}''',

        # Variação 6 - Foco em problematização
        f'''Me ajuda com essa redação do ENEM? O tema é "{tema}"
Preciso que:
- Problematize bem o tema na introdução
- Use argumentos consistentes
- Faça relações com a realidade
- Proponha soluções práticas na conclusão{aviso}''',

        # Prompts Simples
        # Variação 7
        f'''Escreva uma redação do ENEM sobre "{tema}". Preciso que tenha introdução, desenvolvimento e conclusão com proposta de intervenção.{aviso}''',
        
        # Variação 8
        f'''Me ajuda a fazer uma redação para o ENEM? O tema é "{tema}". Quero uma redação completa com uma boa proposta de intervenção no final.{aviso}''',
        
        # Variação 9
        f'''Preciso treinar para o ENEM, pode fazer uma redação modelo sobre "{tema}"? Quero tirar uma nota boa, então capricha na argumentação e na proposta de intervenção.{aviso}''',
        
        # Variação 10
        f'''Tô com dificuldade nesse tema do ENEM: "{tema}". Pode fazer uma redação mostrando como argumentar bem e fazer uma boa conclusão?{aviso}''',
        
        # Variação 11
        f'''Faz uma redação nota 1000 sobre "{tema}"? Preciso que tenha introdução com contextualização, 2 parágrafos de desenvolvimento e conclusão com proposta de intervenção.{aviso}'''
    ]

def salvar_prompts(ano, prompts):
    """Salva os prompts em arquivos"""
    try:
        diretorio = f"database/prompts/{ano}"
        os.makedirs(diretorio, exist_ok=True)
        
        for i, prompt in enumerate(prompts, 1):
            arquivo = f"{diretorio}/prompt_{i:02d}.txt"
            with open(arquivo, "w", encoding="utf-8") as f:
                f.write(prompt)
            logging.info(f"Prompt {i} salvo em {arquivo}")
            
    except Exception as e:
        logging.error(f"Erro ao salvar prompts do ano {ano}: {str(e)}")

if __name__ == "__main__":
    anos = obter_anos_disponiveis()
    if not anos:
        logging.error("Nenhum ano encontrado na pasta enem-essays")
        exit(1)
    
    logging.info(f"Anos encontrados: {anos}")
    
    for ano in anos:
        logging.info(f"Processando ano {ano}")
        tema = ler_tema(ano)
        if tema:
            prompts = gerar_prompts(tema)
            salvar_prompts(ano, prompts)
            logging.info(f"Prompts do ano {ano} gerados com sucesso")
        else:
            logging.error(f"Não foi possível processar o ano {ano}") 