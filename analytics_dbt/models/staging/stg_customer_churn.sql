select
    customerID as customer_id,
    Contract as contract_type,
    InternetService as internet_service,
    MonthlyCharges as monthly_charges,
    TotalCharges as total_charges,
    tenure,
    Churn as churn_label
from read_csv_auto('{{ var("silver_csv_path", "../data/silver/customer_churn_silver.csv") }}')
