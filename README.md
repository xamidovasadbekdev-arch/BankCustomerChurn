# Bank Customer Churn Prediction

Dataset: [Bank Customer Churn — Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

The steps to do an end-to-end project:
1. Data Loading ✅
2. Data Understanding ✅
3. Data Cleaning ✅
4. Exploratory Data Analysis ✅
5. Feature Engineering ✅
6. Train Model ⏳
7. Save model with joblib ⏳
8. FastAPI Wrapping and Dockerizing ⏳
9. Deploying ⏳

This is my another end-to-end project, and this time the focus was on going deeper into EDA — testing every variable against the target instead of just the obvious ones, and being honest when a variable turned out not to matter.

### 1, 2. Data Loading and Understanding

Downloaded the dataset from Kaggle and got a first look at the structure — 18 columns, just over 10,000 rows, 
a mix of customer profile data (CreditScore, Geography, Age, Gender), 
banking relationship data (Tenure, Balance, NumOfProducts, IsActiveMember), and 
customer experience data (Complain, Satisfaction Score, Card Type, Point Earned). 
The target column is `Exited` — 1 if the customer churned, 0 if they stayed.

### 3. Data Cleaning

Checked for nulls and duplicates using `.isnull().sum()` and `.duplicated().sum()` — this dataset came in clean, no missing values and no duplicate rows, so no filling or dropping was needed at this stage.

### 4. EDA

I went through almost every column one by one, comparing it against `Exited` using `groupby().mean()` for exact numbers and pairing that with the right plot for each — `barplot` for categorical comparisons, `regplot` for trends, `violinplot` and `boxplot` for distribution shape, `kdeplot` for density comparisons, and a final `heatmap` across all numeric columns as a cross-check.

A few findings turned out to matter a lot, and a few I expected to matter turned out not to — both felt worth keeping in the writeup.

**Geography** — Germany churns at 32.4%, more than double France (16.2%) and Spain (16.7%). What made this more interesting: France has roughly double Germany's customer count (5,014 vs 2,509), yet both countries lose almost the same *number* of customers (811 vs 814). Germany's retention problem is proportionally much worse than the raw rate alone suggests.

**IsActiveMember** — inactive members churn at 26.9%, active members at 14.3%. Roughly double, and one of the clearer signals in the dataset.

**NumOfProducts** — this one surprised me. I expected more products to mean a more loyal customer, but the relationship is U-shaped: 1 product → 27.7% churn, 2 products → 7.6% churn (the loyalty sweet spot), 3 products → 82.7% churn, 4 products → 100% churn (60 out of 60 customers). Going from 2 to 3+ products seems to correlate with customers leaving almost entirely — possibly oversold products or a service experience that breaks down past a certain point.

**Gender** — female customers churn at 25.1% vs 16.5% for male customers. The dataset doesn't explain why, but the gap is large enough to be worth flagging rather than ignoring.

**HasZeroBalance** — customers with an empty balance churn at 13.8%, less than half the 24.1% rate for customers who actually keep money in the account. Counterintuitive at first — I'd have guessed an idle, empty account signals someone already checked out. My read: a zero-balance account is often already passive, so there's nothing left to "leave" in a meaningful sense, while customers actively managing a balance are the ones actually comparing the bank against competitors.

**Age** — a clear upward trend, older customers churn more, confirmed with both a `regplot` and a grouped average.

On the other side, several variables I expected to carry signal turned out close to flat:

* **CreditScore** — 645 vs 652 between churned and retained, no real gap.
* **EstimatedSalary** — 99.7k vs 101.5k, no real gap.
* **HasCrCard** — 20.8% vs 20.2%, basically identical.
* **Card Type** — all four tiers (Diamond, Platinum, Silver, Gold) sit within 19–22% churn, no separation by tier.
* **Tenure** — churn stays in a narrow 17–23% band from year 0 to year 10, no trend either way.
* **Satisfaction Score** — somewhat surprising given it's a direct feedback metric, but churn stays flat across all five score levels.
* **Point Earned** — 604 vs 607, no gap.

One more check worth calling out separately: **Complain**. The correlation here was almost perfect — 0.1% churn among customers with no complaint on file, 99.5% churn among those who complained. That's not a useful predictive signal, it's data leakage: a complaint is recorded right around when someone is already on their way out, not an early warning sign. I'll be dropping this column before modeling so the model has to learn from the features that are actually available *before* a customer reaches that point.

A correlation heatmap across all numeric columns at the end backed up everything above — `Complain` lighting up almost perfectly against `Exited`, `Age` showing a moderate positive correlation, `IsActiveMember` negative, and the financial/status columns (CreditScore, EstimatedSalary, Satisfaction Score, Point Earned) all sitting close to zero.

#### Conclusion

The pattern across the whole EDA section is pretty consistent: **engagement and relationship factors predict churn — financial and status factors mostly don't.** IsActiveMember, NumOfProducts, Balance, Geography, Age, and Gender all carry real signal. CreditScore, EstimatedSalary, HasCrCard, Card Type, Satisfaction Score, and Point Earned don't meaningfully separate churned customers from retained ones. Customers don't seem to leave because of their financial profile — they leave because of how engaged they are with the bank and how their product relationship is going.


### 5. Feature Engineering
First work i did was dropping unnecessary columns that were rowid, customerid, surname and complain. I dropped them as they have no effect on model's performance. Next, I encoded categorical columns into numerical ones. I changed gender into 1, 0 using binary encoding. Then encoded geography into 010, 100, 000 using one hot encoding with pd.get_dummies(drop_first=True). Lastly, I used ordinal encoding to encode card types into "SILVER": 0,"GOLD": 1,"PLATINUM": 2,"DIAMOND": 3 using mapping.
