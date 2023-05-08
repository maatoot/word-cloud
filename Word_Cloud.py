import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Convert the dataset into a list of transactions
transactions = []
for i in range(0, len(data)):
    transaction = []
    for j in range(0, len(data.columns)):
        item = str(data.values[i, j])
        if item != 'nan':
            transaction.append(item)
    transactions.append(transaction)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
data = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


print(rules.sort_values('lift', ascending=False).head(10))

# Concatenate all the items in the dataset
items = []
for transaction in transactions:
    items += transaction

# word cloud
wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(' '.join(items))

# viz
plt.figure(figsize=(8,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
