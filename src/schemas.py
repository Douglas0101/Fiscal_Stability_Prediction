# src/schemas.py (FINAL)

from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """
    Esquema para a requisição de previsão.
    Os nomes dos campos aqui são os que o cliente da API deve enviar.
    Usamos 'alias' para mapear para os nomes de colunas originais se necessário.
    """
    year: int = Field(..., example=2024)
    inflation_cpi: float = Field(..., alias='Inflation (CPI %)', example=3.1)
    gdp_current_usd: float = Field(..., alias='GDP (Current USD)', example=1800000000000)
    gdp_per_capita: float = Field(..., alias='GDP per Capita (Current USD)', example=45000)
    unemployment_rate: float = Field(..., alias='Unemployment Rate (%)', example=5.8)
    interest_rate_real: float = Field(..., alias='Interest Rate (Real, %)', example=1.5)
    inflation_gdp_deflator: float = Field(alias='Inflation (GDP Deflator, %)', example=2.9)
    gdp_growth_annual: float = Field(alias='GDP Growth (% Annual)', example=2.5)
    current_account_balance_gdp: float = Field(alias='Current Account Balance (% GDP)', example=-2.1)
    government_expense_gdp: float = Field(alias='Government Expense (% of GDP)', example=21.0)
    tax_revenue_gdp: float = Field(alias='Tax Revenue (% of GDP)', example=18.5)
    gross_national_income_usd: float = Field(alias='Gross National Income (USD)', example=1790000000000)
    # Adicione aqui as colunas de country_id se o seu modelo as usar como input direto.
    # Exemplo: country_id_br: float = 0.0 ... etc.

    class Config:
        from_attributes = True
        populate_by_name = True # Permite usar o alias na entrada


class PredictionResponse(BaseModel):
    prediction: int = Field(..., example=0)
    probability: float = Field(..., example=0.95)


class HealthStatus(BaseModel):
    api_status: str = "ok"
    model_loaded: bool
    database_status: str