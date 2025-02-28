import os
import shutil

# Diretório base onde estão as redações do ENEM
enem_dir = "database/enem-essays"
# Diretório onde serão criados os arquivos de IA
ai_dir = "database/ai-essays"

def criar_estrutura():
    # Lista todos os anos na pasta enem-essays
    anos = [nome for nome in os.listdir(enem_dir) if os.path.isdir(os.path.join(enem_dir, nome))]
    
    # Para cada ano, cria a pasta correspondente em ai-essays
    for ano in anos:
        pasta_destino = os.path.join(ai_dir, ano)
        
        # Cria a pasta se não existir
        os.makedirs(pasta_destino, exist_ok=True)
        
        # Cria 11 arquivos com o padrão de nomenclatura
        for i in range(1, 12):
            nome_arquivo = f"redacao_chat_gpt_{i:02d}.txt"
            caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)
            
            # Cria arquivo vazio se não existir
            if not os.path.exists(caminho_arquivo):
                with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                    pass

if __name__ == "__main__":
    criar_estrutura()
