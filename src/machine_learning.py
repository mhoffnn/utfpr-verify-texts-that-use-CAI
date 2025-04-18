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
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextClassifier:
    def __init__(self, max_features: int = 1000, test_size: float = 0.2):
        """
        Inicializa o classificador de texto.
        
        Args:
            max_features (int): Número máximo de features para o TF-IDF
            test_size (float): Proporção do conjunto de teste
        """
        self.max_features = max_features
        self.test_size = test_size
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.model = Pipeline([
            ('clf', MultinomialNB(alpha=0.1))
        ])
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
            processed_text = ' '.join(words)
            
            # Log do texto pré-processado
            logger.debug(f"Texto original: {text[:100]}...")
            logger.debug(f"Texto processado: {processed_text[:100]}...")
            
            return processed_text
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
            logger.info("Pré-processando textos...")
            X_processed = self.preprocess_data(X)
            
            # Extrair features
            logger.info("Extraindo features...")
            X_features = self.extract_features(X_processed)
            logger.info(f"Número de features extraídas: {X_features.shape[1]}")
            
            # Dividir dados
            logger.info("Dividindo dados em treino e teste...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y, test_size=self.test_size, random_state=42, stratify=y
            )
            
            # Treinar modelo
            logger.info("Treinando modelo...")
            self.model.fit(X_train, y_train)
            self._is_fitted = True
            
            # Calcular métricas
            logger.info("Calculando métricas...")
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Validação cruzada estratificada
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_features, y, cv=cv)
            
            # Métricas detalhadas
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')
            
            logger.info("\nMétricas de Performance:")
            logger.info(f"Acurácia no treino: {train_accuracy:.4f}")
            logger.info(f"Acurácia no teste: {test_accuracy:.4f}")
            logger.info(f"Acurácia média na validação cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            logger.info(f"Precisão: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            
            # Matriz de confusão
            cm = confusion_matrix(y_test, y_test_pred)
            logger.info("\nMatriz de Confusão:")
            logger.info(f"Verdadeiros Negativos: {cm[0,0]}")
            logger.info(f"Falsos Positivos: {cm[0,1]}")
            logger.info(f"Falsos Negativos: {cm[1,0]}")
            logger.info(f"Verdadeiros Positivos: {cm[1,1]}")
            
            # Relatório de classificação
            logger.info("\nRelatório de Classificação:")
            logger.info(classification_report(y_test, y_test_pred, target_names=['humano', 'IA']))
            
            # Verificar overfitting
            if train_accuracy > 0.95 and test_accuracy < 0.7:
                logger.warning("Possível overfitting detectado!")
            
            # Mostrar exemplos de erros
            logger.info("\nExemplos de erros de classificação:")
            for i, (true, pred) in enumerate(zip(y_test, y_test_pred)):
                if true != pred:
                    logger.info(f"Exemplo {i}:")
                    logger.info(f"Texto: {X_test[i][:100]}...")
                    logger.info(f"Verdadeiro: {'humano' if true == 0 else 'IA'}")
                    logger.info(f"Predito: {'humano' if pred == 0 else 'IA'}")
            
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
        
        # Verificar dados
        logger.info(f"\nInformações do DataFrame:")
        logger.info(f"Total de linhas: {len(df)}")
        logger.info(f"Colunas: {df.columns.tolist()}")
        logger.info(f"\nDistribuição por ano:")
        logger.info(df['year'].value_counts())
        logger.info(f"\nDistribuição por fonte:")
        logger.info(df['source'].value_counts())
        
        # Preparar dados para treinamento
        X = df['content'].tolist()  # Textos para classificação
        y = df['source'].map({'human': 0, 'ai': 1}).tolist()  # Convertendo labels para 0 e 1
        
        # Verificar balanceamento dos dados
        logger.info(f"\nDistribuição das classes:")
        logger.info(f"Humanos: {y.count(0)}")
        logger.info(f"IA: {y.count(1)}")
        
        # Verificar exemplos de texto
        logger.info("\nExemplos de textos:")
        logger.info(f"Primeiro texto humano: {X[0][:100]}...")
        logger.info(f"Primeiro texto IA: {X[y.index(1)][:100]}...")
        
        logger.info(f"Dados carregados com sucesso")
        
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
        
        # Exemplo de predição
        logger.info("\nExemplo de predição:")
        sample_texts = [
            """“Soldados da França! Do alto dessas pirâmides, quarenta séculos vos contemplam!”. Essa frase, dita por Napoleão Bonaparte aos seus soldados durante a Batalha das Pirâmides, no Egito, em junho de 1798, é uma das muitas frases de efeito que o líder francês proferiu ao longo de sua vida. Napoleão foi uma das figuras mais emblemáticas da história humana.
            Aqueles que foram seus contemporâneos, tanto os entusiastas quanto os detratores, comparavam-no a grandes conquistadores, como Alexandre Magno, da Macedônia, e Otávio Augusto, de Roma. Sua genialidade como estrategista de guerra e suas grandes habilidades como político são, hoje, algo consensual entre os especialistas em sua biografia.""",
            """Napoleão Bonaparte foi um dos maiores líderes militares e políticos da história. Nascido na Córsega em 1769, se destacou no exército francês durante a Revolução Francesa, aproveitando o caos político para subir ao poder. Em 1799, após um golpe de Estado, tornou-se Primeiro Cônsul e, em 1804, se coroou Imperador dos Franceses. Durante seu reinado, expandiu o império francês por grande parte da Europa, implementando reformas significativas, como o Código Napoleônico, que ainda influencia sistemas jurídicos até hoje. Sua ambição militar o levou a várias vitórias, mas também a grandes derrotas, como a fracassada invasão da Rússia em 1812, que começou sua queda.
            Após uma série de derrotas, Napoleão foi forçado a abdicar em 1814 e exilado para a ilha de Elba. Porém, ele retornou à França em 1815, retomando o poder por um breve período conhecido como os "Cem Dias", até ser derrotado de forma definitiva na Batalha de Waterloo. Exilado novamente, agora para a remota ilha de Santa Helena, Napoleão passou seus últimos anos até sua morte em 1821. Seu legado permanece complexo: para uns, foi um tirano implacável, enquanto para outros, um líder visionário que moldou a Europa moderna."""
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
