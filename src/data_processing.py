# src/data_processing.py
import pandas as pd
import os
from src.config import AppConfig
from src.logger_config import get_logger

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, config: AppConfig):
        self.config = config
        self.raw_data_path = config.raw_data_path
        self.target_data_path = config.raw_data_with_target_path
        self.processed_data_path = config.processed_data_path

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"A carregar dados brutos de: {self.raw_data_path}")
        df_raw = pd.read_csv(self.raw_data_path)
        logger.info(f"A carregar dados com o alvo de: {self.target_data_path}")
        df_target = pd.read_csv(self.target_data_path)
        return df_raw, df_target

    def merge_data_with_target(self, df_raw: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
        logger.info("A juntar dados brutos com a coluna-alvo.")
        target_cols = ['country_id', 'year', self.config.model.TARGET_VARIABLE]
        df_target_subset = df_target[target_cols]
        df_merged = pd.merge(df_raw, df_target_subset, on=['country_id', 'year'], how='left')
        df_merged.dropna(subset=[self.config.model.TARGET_VARIABLE], inplace=True)
        df_merged[self.config.model.TARGET_VARIABLE] = df_merged[self.config.model.TARGET_VARIABLE].astype(int)
        return df_merged

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("A iniciar a limpeza de dados...")
        if self.config.model.DROP_COLUMNS:
            df = df.drop(columns=self.config.model.DROP_COLUMNS, errors='ignore')
        for col in df.columns:
            if df[col].dtype == 'object' and col not in self.config.model.CATEGORICAL_FEATURES:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def save_data(self, df: pd.DataFrame):
        os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
        logger.info(f"A salvar dados processados em: {self.processed_data_path}")
        df.to_csv(self.processed_data_path, index=False)

    def run(self):
        logger.info("--- INICIANDO PIPELINE DE PROCESSAMENTO DE DADOS ---")
        df_raw, df_target = self.load_datasets()
        df_merged = self.merge_data_with_target(df_raw, df_target)
        cleaned_df = self.clean_data(df_merged)
        self.save_data(cleaned_df)
        logger.info("--- PIPELINE DE PROCESSAMENTO DE DADOS CONCLUÍDO ---")

if __name__ == "__main__":
    config = AppConfig()
    processor = DataProcessor(config)
    processor.run()
