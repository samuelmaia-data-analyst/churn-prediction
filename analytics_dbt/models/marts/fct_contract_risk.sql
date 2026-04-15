select
    contract_type,
    internet_service,
    count(*) as customers,
    avg(case when churn_label = 1 then 1.0 else 0.0 end) as churn_rate,
    avg(monthly_charges) as avg_monthly_charges,
    sum(monthly_charges) as monthly_revenue
from {{ ref('stg_customer_churn') }}
group by 1, 2
