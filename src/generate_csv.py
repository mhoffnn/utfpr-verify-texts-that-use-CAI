import os
import csv
from pathlib import Path
import random

# Configurações
database_dir = "utfpr-verify-texts-that-use-CAI/database"
train_csv = "train_essays.csv"
validate_csv = "validate_essays.csv"
validation_ratio = 0.1  # 10% para validação

# Estrutura para armazenar os dados
data = {
    "human": {},
    "ai": {}
}

# Função para limpar o texto
def clean_text(text):
    return text.replace('\n', ' ').strip()

# Processar arquivos humanos (enem-essays)
for year_dir in Path(database_dir, "enem-essays").iterdir():
    if not year_dir.is_dir() or year_dir.name == "subject.txt":
        continue
    
    year = year_dir.name
    data["human"][year] = []
    
    for essay_file in year_dir.iterdir():
        if essay_file.name == "subject.txt" or not essay_file.is_file():
            continue
        
        # Extrair conteúdo do arquivo
        with open(essay_file, 'r', encoding='utf-8') as f:
            content = clean_text(f.read())
        
        # Extrair nome do autor (se existir)
        author = essay_file.stem.split('-', 1)[-1].strip()
        
        data["human"][year].append({
            "author": author,
            "content": content,
            "source": "human",
            "prompt": "N/A"
        })

# Processar arquivos de IA (ai-essays)
for year_dir in Path(database_dir, "ai-essays").iterdir():
    if not year_dir.is_dir():
        continue
    
    year = year_dir.name
    data["ai"][year] = []
    
    for essay_file in year_dir.iterdir():
        if not essay_file.is_file():
            continue
        
        # Extrair número do prompt
        try:
            prompt_num = int(essay_file.stem.split('_')[-1].split('.')[0])
        except:
            prompt_num = 0
        
        # Extrair conteúdo do arquivo
        with open(essay_file, 'r', encoding='utf-8') as f:
            content = clean_text(f.read())
        
        data["ai"][year].append({
            "author": "ChatGPT",
            "content": content,
            "source": "ai",
            "prompt": f"Prompt {prompt_num}"
        })

# Função para separar dados em treinamento e validação
def split_data(data_dict, ratio):
    train_data = []
    validate_data = []
    
    for year, essays in data_dict.items():
        # Embaralhar os ensaios
        random.shuffle(essays)
        
        # Calcular o ponto de corte
        split_index = int(len(essays) * (1 - ratio))
        
        # Separar os dados
        train_data.extend([{**essay, 'year': year} for essay in essays[:split_index]])
        validate_data.extend([{**essay, 'year': year} for essay in essays[split_index:]])
    
    return train_data, validate_data

# Separar dados humanos e de IA
human_train, human_validate = split_data(data["human"], validation_ratio)
ai_train, ai_validate = split_data(data["ai"], validation_ratio)

# Função para escrever dados no CSV
def write_to_csv(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['year', 'source', 'prompt', 'author', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Escrever arquivos de treinamento e validação
write_to_csv(train_csv, human_train + ai_train)
write_to_csv(validate_csv, human_validate + ai_validate)

print(f"Arquivos CSV gerados com sucesso:")
print(f"- Arquivo de treinamento: {train_csv}")
print(f"- Arquivo de validação: {validate_csv}")
print(f"Total de textos no treinamento: {len(human_train) + len(ai_train)}")
print(f"Total de textos na validação: {len(human_validate) + len(ai_validate)}")
