import pandas as pd
import numpy as np

np.random.seed(42)
num_samples = 5000

data = {
    "age": np.random.randint(21, 70, num_samples),
    "income": np.random.randint(20000, 200000, num_samples),
    "loan_amount": np.random.randint(5000, 500000, num_samples),
    "num_loans": np.random.randint(1, 10, num_samples),
    "credit_card_usage": np.random.randint(0, 100, num_samples),
    "delinquency_rate": np.random.uniform(0, 1, num_samples),
    "cibil_score": np.random.randint(300, 900, num_samples)
}

df = pd.DataFrame(data)
df.to_csv("cibil_dataset.csv", index=False)
print("✅ CIBIL dataset generated: cibil_dataset.csv")
