from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from pydantic.alias_generators import to_camel
from typing import Dict, Optional
from datetime import datetime


# --- Configuração Base ---
class CamelCaseModel(BaseModel):
    """Um modelo base que converte snake_case para camelCase para a API."""

    class Config:
        # Pydantic v1: alias_generator = to_camel, allow_population_by_field_name = True
        # Pydantic v2:
        alias_generator = to_camel
        populate_by_name = True
        from_attributes = True


# --- Schemas para a API de Predição ---

class PredictionFeatures(CamelCaseModel):
    """Schema para os dados de entrada da predição, com validação e descrições."""
    taxa_de_juros: float = Field(
        ..., ge=-10, le=100,
        description="A taxa de juros básica da economia (em %).",
        json_schema_extra={"example": 5.0}
    )
    inflacao_anual: float = Field(
        ..., ge=-20, le=100,
        description="A taxa de inflação acumulada nos últimos 12 meses (em %).",
        json_schema_extra={"example": 2.5}
    )
    crescimento_pib: float = Field(
        ..., ge=-50, le=50,
        description="A taxa de crescimento do Produto Interno Bruto (em %).",
        json_schema_extra={"example": 3.0}
    )
    divida_publica_pib: float = Field(
        ..., ge=0,
        description="A razão entre a dívida pública e o PIB (em %).",
        json_schema_extra={"example": 60.0}
    )
    balanca_comercial: float = Field(
        ...,
        description="O saldo da balança comercial (exportações - importações, em bilhões de USD).",
        json_schema_extra={"example": 10.0}
    )
    investimento_estrangeiro: float = Field(
        ...,
        description="O investimento estrangeiro direto (em bilhões de USD).",
        json_schema_extra={"example": 20.0}
    )


class PredictionResultBase(CamelCaseModel):
    """Schema base contendo os resultados essenciais de uma predição."""
    prediction: str = Field(..., description="A predição textual: 'Estável' ou 'Instável'.", examples=["Estável"])
    prediction_code: int = Field(..., description="O código numérico da predição (0 para Estável, 1 para Instável).",
                                 examples=[0])
    probability_stable: float = Field(..., ge=0, le=1, description="A probabilidade calculada da classe 'Estável'.",
                                      examples=[0.95])
    probability_unstable: float = Field(..., ge=0, le=1, description="A probabilidade calculada da classe 'Instável'.",
                                        examples=[0.05])

    @model_validator(mode='after')
    def check_probabilities_sum_to_one(self) -> 'PredictionResultBase':
        """Valida se a soma das probabilidades é aproximadamente 1."""
        prob_sum = self.probability_stable + self.probability_unstable
        if not (0.99 < prob_sum < 1.01):
            raise ValueError('A soma das probabilidades deve ser igual a 1')
        return self


class PredictionResponse(PredictionResultBase):
    """Schema completo para a resposta da API, incluindo SHAP values e ID do registro."""
    prediction_id: int = Field(..., description="O ID do registro da predição salvo no banco de dados.", examples=[123])
    shap_values: Dict[str, float] = Field(..., description="Valores SHAP que explicam a contribuição de cada feature.")


class PredictionRecord(PredictionResultBase):
    """Schema para exibir um registro histórico de predição do banco de dados."""
    id: int
    created_at: datetime
    features: PredictionFeatures  # Tipagem forte usando o schema já definido


# --- Schemas para Autenticação e Usuários ---

class Token(CamelCaseModel):
    access_token: str
    token_type: str


class TokenData(CamelCaseModel):
    subject: str = Field(..., description="O 'subject' do token, geralmente o email ou ID do usuário.")


class UserBase(CamelCaseModel):
    email: str = Field(..., description="Email do usuário.", examples=["user@example.com"])


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="Senha do usuário (mínimo 8 caracteres).")


class User(UserBase):
    id: int
    is_active: bool = Field(..., description="Indica se a conta do usuário está ativa.")