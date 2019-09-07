import pandas as pd
import random
import yfinance as yf
# from sec_edgar_downloader import Downloader
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import re
import string
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# static stuff
BASE_URL = 'https://finance.yahoo.com/quote/'
END_URL = '/profile?ltr=1'
TICKERS_LIST_SAMPLE = ['MSFT', 'AMZN', 'FB']
TICKER_LIST_URL = 'https://raw.githubusercontent.com/bowenanalytics/penn_apps/master/yahoo_tickers.csv'
D2V_MODEL_NAME = get_tmpfile("d2v_model")


def scrape_business_description(ticker):
    """given a ticker, scrape yahoo! finance for the business description of the company."""
    url = BASE_URL + ticker + END_URL
    bd = ''
    print('Attempting to scrape business description for ticker {}...'
          .format(ticker))
    with requests.Session() as s:
        try:
            r = s.get(url)
            data = r.text
            bs = BeautifulSoup(data, features='html.parser')
            results = bs.find_all('section', class_='quote-sub-section Mt(30px)')
            bd = results[0].find('p').text
            print('Successfully scraped ticker {}'.format(ticker))

        except:
            print('No business description found for {}'.format(ticker))
        time.sleep(0.5)
    return bd

def manage_scrape_process(tickers):
    """scraper manager. Scrapes a ticker list and returns a DataFrame containing the results."""
    data = {'ticker': [], 'business_description': []}
    for t in tickers:
        bd = scrape_business_description(t)
        data['ticker'].append(t)
        data['business_description'].append(bd)
    data_df = pd.DataFrame(data, columns=data.keys())
    return data_df

def scrape_all():
    """reads the saved csv of ticker list from the GitHub repo and filter for NYSE stocks."""
    df = pd.read_csv(TICKER_LIST_URL, index_col=0).reset_index()
    # NMS = NASDAQ
    # NYQ = NYSE
    df = df[((df['Exchange'] == 'NMS') | (df['Exchange'] == 'NYQ'))].sort_values('Ticker').reset_index()
    # tickers = df[:10]['Ticker'].tolist()
    tickers = df['Ticker'].tolist()
    data_df = manage_scrape_process(tickers)
    data_df.to_pickle("./test.pkl")

def generate_tagged_docs(df):
    """perform text cleaning in preparation for Doc2Vec model."""
    bd = df['business_description'].tolist()
    bd_out = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for d in bd:
        # lowercase
        d = d.lower()
        # remove numbers
        d = re.sub(r'\d+', '', d)
        # remove punctuation
        d = d.translate(str.maketrans('', '', string.punctuation))
        # remove leading/trailing whitespace
        d = d.strip()
        # tokenize
        tokens = word_tokenize(d)
        d = [i for i in tokens if not i in stop_words]
        d = [lemmatizer.lemmatize(i) for i in d]
        bd_out.append(d)
    df['cleaned_description'] = bd_out
    df = df[['ticker', 'cleaned_description']]
    df = df[df['cleaned_description'].str.len() > 0]

    tickers = df['ticker'].tolist()
    lists_of_words = df['cleaned_description'].tolist()
    docs = []
    for i in range(len(tickers)):
        doc_tag = [tickers[i]]
        doc_words = lists_of_words[i]
        t = TaggedDocument(doc_words, doc_tag)
        docs.append(t)

    return docs

def print_random_similarity(df, docs, model):
    """for debugging. picks a ticker at random, computes the similarity, and prints."""
    # random.seed(100)
    doc_id = random.randint(0, len(docs) - 1)
    this_doc = docs[doc_id][1]
    inferred_vector = model.infer_vector(this_doc)
    sims = model.docvecs.most_similar([inferred_vector], topn=5)
    target_ticker = this_doc[0]
    print('target ticker: {}'.format(target_ticker))
    this_desc = df[df['ticker'] == target_ticker]['business_description'].tolist()[0]
    print(this_desc)
    print('\nmost similar:')
    for s in sims:
        t = s[0]
        this_desc = df[df['ticker'] == t]['business_description'].tolist()[0]
        print(this_desc)



if __name__ == '__main__':

    # scrape_all()
    df = pd.read_pickle("./test.pkl")
    docs = generate_tagged_docs(df)

    # model = Doc2Vec(docs, vector_size=30, window=5, min_count=2, workers=4)
    # model.save(D2V_MODEL_NAME)

    model = Doc2Vec.load(D2V_MODEL_NAME)

    # print_random_similarity(df, docs, model)

    tickers_list = df['ticker'].dropna().tolist()
    static_dict = {
        'code': [], 'desc': [],
        's1': [], 's2': [], 's3': [], 's4': [], 's5': [],
        'd1': [], 'd2': [], 'd3': [], 'd4': [], 'd5': []
    }
    for t in tickers_list:
        this_desc = df[df['ticker'] == t]['business_description'].tolist()[0]
        inferred_vector = model.infer_vector([t])
        sims = model.docvecs.most_similar([inferred_vector], topn=5)

        static_dict['code'].append(t)
        static_dict['desc'].append(df[df['ticker'] == t]['business_description'].tolist()[0])

        t1 = sims[0][0]
        t2 = sims[1][0]
        t3 = sims[2][0]
        t4 = sims[3][0]
        t5 = sims[4][0]

        static_dict['s1'].append(t1)
        static_dict['d1'].append(df[df['ticker'] == t1]['business_description'].tolist()[0])
        static_dict['s2'].append(t2)
        static_dict['d2'].append(df[df['ticker'] == t2]['business_description'].tolist()[0])
        static_dict['s3'].append(t3)
        static_dict['d3'].append(df[df['ticker'] == t3]['business_description'].tolist()[0])
        static_dict['s4'].append(t4)
        static_dict['d4'].append(df[df['ticker'] == t4]['business_description'].tolist()[0])
        static_dict['s5'].append(t5)
        static_dict['d5'].append(df[df['ticker'] == t5]['business_description'].tolist()[0])

    out_df = pd.DataFrame(static_dict, columns=static_dict.keys())
    out_df.to_csv('/users/votta/code/penn_apps/similarity.csv')
    out_df.to_parquet('/users/votta/code/penn_apps/similarity.parquet.gzip')


    z = 0




    # # Initialize a downloader instance.
    # # If no argument is passed to the constructor, the package
    # # will attempt to locate the user's downloads folder.
    # dl = Downloader('/users/votta/code/penn_apps')
    # # Get all 8-K filings for Apple (ticker: AAPL)
    # dl.get_8k_filings("AAPL")
    # # Get all 8-K filings for Apple, including filing amends (8-K/A)
    # dl.get_8k_filings("AAPL", include_amends=True)
    # # Get all 8-K filings for Apple before March 25, 2017
    # # Note: before_date string must be in the form "YYYYMMDD"
    # dl.get_8k_filings("AAPL", before_date="20170325")
    # # Get the past 5 8-K filings for Apple
    # dl.get_8k_filings("AAPL", 5)
    # # Get all 10-K filings for Microsoft (ticker: MSFT)
    # dl.get_10k_filings("MSFT")
    # # Get the latest 10-K filing for Microsoft
    # dl.get_10k_filings("MSFT", 1)
    # # Get all 10-Q filings for Visa (ticker: V)
    # dl.get_10q_filings("V")
    # # Get all 13F-NT filings for the Vanguard Group (CIK: 0000102909)
    # dl.get_13f_nt_filings("0000102909")
    # # Get all 13F-HR filings for the Vanguard Group
    # dl.get_13f_hr_filings("0000102909")
    # # Get all SC 13G filings for Apple
    # dl.get_sc_13g_filings("AAPL")
    # # Get all SD filings for Apple
    # dl.get_sd_filings("AAPL")



    # msft = yf.Ticker("MSFT")
    # # get stock info
    # stock_info = msft.info
    # # get historical market data
    # hist = msft.history(period="max")
    # # show actions (dividends, splits)
    # stock_actions = msft.actions
    # # show dividends
    # stock_div = msft.dividends
    # # show splits
    # stock_splits = msft.splits
    # # show financials
    # stock_financials = msft.financials
    # # show balance heet
    # stock_bs = msft.balance_sheet
    # # show cashflow
    # stock_cf = msft.cashflow
    # # show options expirations
    # stock_opt = msft.options
    # # get option chain for specific expiration
    # opt = msft.option_chain('2019-09-26')
    # # data available via: opt.calls, opt.puts

    z = 1