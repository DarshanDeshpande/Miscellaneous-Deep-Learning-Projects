import gzip, json
import pandas as pd

review,summarylist = [],[]
with gzip.open(r'Software.json.gz') as f:
    for l in f:
        full_text = json.loads(l.strip())
        try:
            if full_text['summary'] and full_text['reviewText']:
                review.append(full_text['reviewText'])
                summarylist.append(full_text['summary'])
        except KeyError as k:
            continue


df_dict = {'review': review, 'summary': summarylist}

df = pd.DataFrame(df_dict)
df.to_csv('TextSummarizationDataset.csv')