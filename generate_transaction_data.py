import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate 10,000 samples
n_samples = 10000

# Generate SNo (1 to 10000)
sno = list(range(1, n_samples + 1))

# Generate Time values in HH:MM format
# Distribute transactions throughout the day (24 hours)
time_values = []
for i in range(n_samples):
    # Generate random time throughout the day
    # Distribute more transactions during business hours (9 AM - 6 PM)
    if random.random() < 0.6:  # 60% during business hours
        hour = random.randint(9, 17)
    else:  # 40% during other hours
        hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    # Format as HH:MM
    time_str = f"{hour:02d}:{minute:02d}"
    time_values.append(time_str)

# Generate Transaction_ID (16-digit numbers like 3597980945245810)
transaction_ids = []
for i in range(n_samples):
    # Generate 16-digit number
    # Start with a base and add variation
    base = 3000000000000000 + random.randint(0, 999999999999999)
    transaction_id = str(base).zfill(16)
    transaction_ids.append(transaction_id)

# Generate Amount values (realistic transaction amounts)
# Most transactions are small, some are larger (following a log-normal distribution)
amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_samples)
amounts = np.round(amounts, 2)
# Cap at reasonable maximum
amounts = np.clip(amounts, 0.01, 50000.0)

# Generate Fraud or not (0 = Not Fraud, 1 = Fraud)
fraud_count = 935
fraud_labels = np.zeros(n_samples, dtype=int)
fraud_indices = np.random.choice(n_samples, size=fraud_count, replace=False)
fraud_labels[fraud_indices] = 1

# Adjust amounts for fraud transactions (fraud tends to have different amount patterns)
for i in range(n_samples):
    if fraud_labels[i] == 1:
        # Fraud transactions often have unusual amounts
        if random.random() < 0.5:
            # Some fraud transactions are very small (testing cards)
            amounts[i] = round(random.uniform(0.01, 10.0), 2)
        else:
            # Some fraud transactions are large
            amounts[i] = round(random.uniform(1000.0, 50000.0), 2)

# Create DataFrame
df = pd.DataFrame({
    'SNo': sno,
    'Time': time_values,
    'Transaction_ID': transaction_ids,
    'Amount': amounts,
    'Fraud or not': fraud_labels
})

# Save to CSV
output_file = 'transaction_data.csv'
df.to_csv(output_file, index=False)

print(f"Generated {n_samples} samples and saved to {output_file}")
print(f"Fraud transactions: {fraud_labels.sum()} ({fraud_labels.sum()/n_samples*100:.2f}%)")
print(f"Normal transactions: {(fraud_labels == 0).sum()} ({(fraud_labels == 0).sum()/n_samples*100:.2f}%)")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())

