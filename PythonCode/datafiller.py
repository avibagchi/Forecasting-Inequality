from pandas_datareader import wb
import xlsxwriter
import pandas as pd


df=pd.DataFrame ()
def method (data):
    data = data.reset_index(1)
    data.columns = ['year', 'gini']
    data = data.drop('year', 1)
    #print (data.loc[data['gini'] >= 0].iloc[0])
    global df
    df=data.loc[data['gini'] >= 0].iloc[0]
    print (df)
    with pd.ExcelWriter ('gini27.xlsx', engine="openpyxl", mode='a') as writer:
        #writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        df.to_excel(writer, sheet_name="Sheet1")


results= []
with open ('Abbreviations.csv') as read_obj:
    for row in read_obj:
        results.append (row.rstrip("\n"))
print (results)

for x in range (len (results)):
    try:
        data=wb.download (indicator='SI.POV.GINI',country=str (results [x]), start=2000, end=2020)
        print (method (data))
    except:
        print ("No data available for: "+results [x])

df2 = pd.concat(pd.read_excel('gini27.xlsx', sheet_name=None), ignore_index=True)
df2.to_excel ("gini28.xlsx")






