import nltk
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Union
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.exceptions import NotFittedError

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(self, max_features: int = 5000, n_estimators: int = 100, test_size: float = 0.2):
        """
        Inicializa o classificador de texto.
        
        Args:
            max_features (int): Número máximo de features para o TF-IDF
            n_estimators (int): Número de árvores no Random Forest
            test_size (float): Proporção do conjunto de teste
        """
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.test_size = test_size
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.lemmatizer = WordNetLemmatizer()
        self._is_fitted = False
        
        # Baixar recursos necessários do NLTK
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')

    def preprocess_text(self, text: str) -> str:
        """
        Pré-processa um texto removendo stopwords, lematizando e normalizando.
        
        Args:
            text (str): Texto a ser pré-processado
            
        Returns:
            str: Texto pré-processado
        """
        try:
            # Converter para minúsculas
            text = text.lower()
            # Remover pontuação e caracteres especiais
            text = ''.join([c for c in text if c.isalpha() or c == ' '])
            # Tokenizar
            words = text.split()
            # Remover stopwords
            words = [w for w in words if w not in stopwords.words('portuguese')]
            # Lematização
            words = [self.lemmatizer.lemmatize(w) for w in words]
            return ' '.join(words)
        except Exception as e:
            logger.error(f"Erro no pré-processamento do texto: {str(e)}")
            raise

    def preprocess_data(self, texts: List[str]) -> List[str]:
        """
        Pré-processa uma lista de textos.
        
        Args:
            texts (List[str]): Lista de textos a serem pré-processados
            
        Returns:
            List[str]: Lista de textos pré-processados
        """
        return [self.preprocess_text(text) for text in texts]

    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extrai features dos textos usando TF-IDF.
        
        Args:
            texts (List[str]): Lista de textos pré-processados
            
        Returns:
            np.ndarray: Matriz de features
        """
        try:
            return self.vectorizer.fit_transform(texts).toarray()
        except Exception as e:
            logger.error(f"Erro na extração de features: {str(e)}")
            raise

    def train(self, X: List[str], y: List[int]) -> Tuple[float, float]:
        """
        Treina o modelo com os dados fornecidos.
        
        Args:
            X (List[str]): Lista de textos
            y (List[int]): Lista de labels
            
        Returns:
            Tuple[float, float]: Acurácia no treino e teste
        """
        try:
            # Pré-processar dados
            X_processed = self.preprocess_data(X)
            
            # Extrair features
            X_features = self.extract_features(X_processed)
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y, test_size=self.test_size, random_state=42
            )
            
            # Treinar modelo
            self.model.fit(X_train, y_train)
            self._is_fitted = True
            
            # Calcular acurácia
            train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
            test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
            
            logger.info(f"Acurácia no treino: {train_accuracy:.4f}")
            logger.info(f"Acurácia no teste: {test_accuracy:.4f}")
            
            return train_accuracy, test_accuracy
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {str(e)}")
            raise

    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Faz predições para novos textos.
        
        Args:
            texts (List[str]): Lista de textos para predição
            
        Returns:
            np.ndarray: Predições do modelo
        """
        if not self._is_fitted:
            raise NotFittedError("O modelo precisa ser treinado antes de fazer predições")
            
        try:
            # Pré-processar dados
            X_processed = self.preprocess_data(texts)
            
            # Extrair features
            X_features = self.vectorizer.transform(X_processed).toarray()
            
            # Fazer predições
            return self.model.predict(X_features)
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            raise

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Salva o modelo e o vetorizador em arquivos.
        
        Args:
            path (Union[str, Path]): Caminho para salvar os arquivos
        """
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(self.model, path / 'model.joblib')
            joblib.dump(self.vectorizer, path / 'vectorizer.joblib')
            
            logger.info(f"Modelo salvo em {path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar o modelo: {str(e)}")
            raise

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Carrega o modelo e o vetorizador de arquivos.
        
        Args:
            path (Union[str, Path]): Caminho dos arquivos do modelo
        """
        try:
            path = Path(path)
            
            self.model = joblib.load(path / 'model.joblib')
            self.vectorizer = joblib.load(path / 'vectorizer.joblib')
            self._is_fitted = True
            
            logger.info(f"Modelo carregado de {path}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar o modelo: {str(e)}")
            raise

def main(csv_path: str = 'train_essays.csv'):
    """
    Função principal para treinar o modelo com dados do CSV.
    
    Args:
        csv_path (str): Caminho para o arquivo CSV contendo os dados de treinamento.
                       Por padrão, usa 'train_essays.csv'
    """
    try:
        # Carregar dados do CSV
        logger.info(f"Carregando dados do CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Preparar dados para treinamento
        X = df['content'].tolist()  # Textos para classificação
        y = df['source'].map({'human': 0, 'ai': 1}).tolist()  # Convertendo labels para 0 e 1
        logger.info(f"Dados carregados com sucesso: {len(X)} textos e {len(y)} labels")
        
        # Inicializar e treinar o classificador
        logger.info("Inicializando classificador...")
        classifier = TextClassifier()
        
        # Treinar modelo
        logger.info("Iniciando treinamento...")
        train_acc, test_acc = classifier.train(X, y)
        
        # Salvar modelo treinado
        logger.info("Salvando modelo...")
        classifier.save_model("model")
        
        logger.info(f"Treinamento concluído com sucesso!")
        logger.info(f"Acurácia no treino: {train_acc:.4f}")
        logger.info(f"Acurácia no teste: {test_acc:.4f}")
        
        # Exemplo de predição
        logger.info("\nExemplo de predição:")
        sample_texts = [
            "Este é um texto de exemplo para teste",
            "Outro texto para verificar a classificação"
        ]
        predictions = classifier.predict(sample_texts)
        for text, pred in zip(sample_texts, predictions):
            label = "humano" if pred == 0 else "IA"
            logger.info(f"Texto: {text[:50]}... -> Classificado como: {label}")
        
    except Exception as e:
        logger.error(f"Erro na execução principal: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    # Se um argumento foi fornecido, use-o como caminho do CSV
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'train_essays.csv'
    main(csv_path)
