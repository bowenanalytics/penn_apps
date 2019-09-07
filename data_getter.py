import pandas as pd
import yfinance as yf
# from sec_edgar_downloader import Downloader
import requests
from bs4 import BeautifulSoup

BASE_URL = """https://finance.yahoo.com/quote/"""
END_URL = """/profile?ltr=1"""
tickers = ['MSFT', 'AMZN', 'FB']

def scrape_business_description(ticker):
    url = BASE_URL + ticker + END_URL
    bd = ''
    print('Attempting to scrape business description for ticker {}...'
          .format(ticker))
    with requests.Session() as s:
        r = s.get(url)
        data = r.text
        bs = BeautifulSoup(data, features='html.parser')
        results = bs.find_all('section', class_='quote-sub-section Mt(30px)')
        bd = results[0].find('p').text

    print('Successfully scraped')
    return bd

def manage_scrape_process(tickers):
    data = {'ticker' : [], 'business_description' : []}
    for t in tickers:
        bd = scrape_business_description(t)
        data['ticker'].append(t)
        data['business_description'].append(bd)
    data_df = pd.DataFrame(data, columns=data.keys())

    z = 0





if __name__ == '__main__':

    manage_scrape_process(tickers)





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