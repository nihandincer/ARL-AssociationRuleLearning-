############################################
# ASSOCIATION_RULE_LEARNING
############################################


############################################
# Data Preparation
###########################################


!pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_excel("/Users/nHn/Desktop/online_retail_II.xlsx",
                       sheet_name="Year 2010-2011")

df = df_.copy()
df.info()
df.head()

#HELPERS
from Desktop.helpers.helpers import check_df
check_df(df)

#HELPERS
from Desktop.helpers.helpers import crm_data_prep

#veri düzenleme
#eksik verileri silme
#gereken düzeltmeleri yaparak veriyi betimsel olarak hazırlama.
df=crm_data_prep(df)
check_df(df)

#Almanya'yı seciyorum.
df_ger=df[df["Country"]=="Germany"]  #9495 rows
check_df(df)


#Her bir ürünün StockCode a göre Quantity sini sum ediyoruz.
df_ger.groupby(["Invoice","StockCode"]).agg({"Quantity":"sum"}).head(10)


#Invoice ları tekillestiriyoruz.
df_ger.groupby(["Invoice","StockCode"]).agg({"Quantity":"sum"}).unstack().iloc[0:5,0:5]


df[(df["StockCode"] == 16235) & (df["Invoice"] == 538174)]


#Satırlarda sadece bir adet fatura adı olsun.
#Sütunlarda ürünler olsun.
#kesişiminde hangi faturalardan kaçar tane olduğu yazsın.
df_ger.groupby(["Invoice","StockCode"]).\
    agg({"Quantity":"sum"}).\
    unstack().fillna(0).iloc[0:5,0:5]


#Veriyi beklenen product forma getirdik.
df_ger.groupby(["Invoice","StockCode"]).\
    agg({"Quantity":"sum"}).\
    unstack().fillna(0).\
    applymap(lambda i:1 if i>0 else 0).iloc[0:5,0:5]


# fonksiyonlaştırdık.
# Stockcode yerine ürün isimlerini aldık.
def create_invoice_product_df(dataframe):
    return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)


ger_inv_pro_df = create_invoice_product_df(df_ger)

ger_inv_pro_df.head()



# Her bir invoice'da kaç eşsiz ürün vardır.
df_ger.groupby("Invoice"). agg({"StockCode":"nunique"})

# Her bir product kaç eşsiz sepettedir.
df_ger.groupby("StockCode"). agg({"Invoice":"nunique"})



############################################
# ASSOCIATION_RULE
############################################

#apriori kullanılıyor.
#min_support değeri 0.01 olanları alsınn.
#use_colnames= kolon isimlerini kullansın.
#item ların tek tek ve kombinasyonlu supportları geldi.

frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)




rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
rules.sort_values("lift", ascending=False).head()

#conviction: Y olmadan X in beklenen frekansı
#leverage: Lifte benzer. Supportu yüksek değere öncelik verir. Yanlıdır.
#lift=daha az sıklığa rağmen güçlü ilişkileri bulabilir. Yansızdır. Support düşük olsa da ilişkiler nettir.


############################################
# Functionalization
############################################


import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
from Desktop.helpers.helpers import crm_data_prep, create_invoice_product_df

df_ = pd.read_excel("/Users/nHn/Desktop/online_retail_II.xlsx",
                       sheet_name="Year 2010-2011")
df = df_.copy()

df = crm_data_prep(df)

def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))

    return rules


rules = create_rules(df)

#support ve lifte göre sıralama.
rules.sort_values(["support","lift"], ascending= [False,False]).head()

