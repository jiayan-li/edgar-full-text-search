import re
from bs4 import BeautifulSoup
import os
import requests
import logging
from transformers import pipeline
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional


headers = {
    "User-Agent": "Contact: jiayanjoanneli@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "efts.sec.gov"
}

headers_sec = {
    "User-Agent": "Contact: jiayanjoanneli@gmail.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}


def transform_filename(filename: str) -> str:
    """
    Transforms a filename from the format 'a-b:c' to 'ab/c' for use as in URL.
    """

    # Find the position of the colon
    colon_pos = filename.index(':')
    
    # Remove hyphens before the colon
    before_colon = filename[:colon_pos].replace('-', '')
    
    # Replace colon with a forward slash
    after_colon = filename[colon_pos:].replace(':', '/')
    
    # Concatenate the parts
    transformed_filename = before_colon + after_colon
    return transformed_filename


def extract_info(display_name):
    """
    Extracts the company name, ticker, and CIK from the display name of a company.
    """

    # Define the regex pattern with flexible whitespace matching
    pattern = r"^(.*)\s+\(([^)]+)\)\s+\(CIK\s+([0-9]+)\)$"
    
    # Match the pattern
    match = re.match(pattern, display_name)
    
    if match:
        company_name = match.group(1).strip()  # Strip any leading/trailing whitespace
        ticker = match.group(2).strip() if match.group(2) else None
        cik = match.group(3).strip()
        return company_name, ticker, cik
    else:
        try:
            # Try to match the pattern without ticker
            pattern = r"^(.*)\s+\(CIK\s+([0-9]+)\)$"
            match = re.match(pattern, display_name)
            if match:
                company_name = match.group(1).strip()
                ticker = None
                cik = match.group(2).strip()
                return company_name, ticker, cik
        except:
            # If no match is found, return None for all fields
            return None, None, None


def parse_content(content: bytes) -> str:
    """
    Parse the content of a form.
    Args:
        content (bytes): The content of the form.
    Returns:
        str: The parsed content of the form.
    """
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(content, 'html.parser')

    # Extract the text from the page
    text = soup.get_text()

    # Clean up the text by removing extra whitespace and newlines
    cleaned_text = ' '.join(text.split())

    return cleaned_text


def zero_shot_classification(text: str) -> Dict:
    """
    Perform zero-shot classification on the relevant text.
    """

    # Set logging level to ERROR to silence warnings
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Load the zero-shot-classification pipeline
    classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")

    class_labels = [
        "CFIUS review announced (without a final determination)",
        "Notices of CFIUS approvals",
        "Notices of CFIUS denials"
    ]

    # Perform zero-shot classification
    result = classifier(text, candidate_labels=class_labels)

    return result


### Functions in main script for easier testing ###
def define_search(end_date: Optional[str] = None,
                  start_date: Optional[str] = None,
                  recent_n_day: Optional[int] = None,
                  keyword: str = 'CFIUS',
                  forms: List[str] = ["8-K"]
                  ) -> dict:
    """
    Define the search parameters for the EDGAR Full Text Search API.
    Args:
        end_date (str): The end date for the search in the format 'YYYY-MM-DD'.
        start_date (str): The start date for the search in the format 'YYYY-MM-DD'.
        recent_n_day (int): The number of recent days to search.
        keyword (str): The keyword to search for in the filings.
        forms (list): A list of form types to search for.
    Returns:
        dict: The search parameters for the API.
    """

    # Check if the start date or recent_n_day parameter is provided
    if start_date is not None and recent_n_day is not None:
        raise ValueError("Both start_date and recent_n_day parameters cannot be provided.")
    
    # Set the end date to today if not provided
    if end_date is None:
        end_date = datetime.now()
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        end_date_str = end_date

    # Calculate the start date based on the recent_n_day parameter
    if recent_n_day is not None:
        start_date = end_date - timedelta(days=recent_n_day)
        start_date_str = start_date.strftime('%Y-%m-%d')
    elif start_date is not None:
        start_date_str = start_date
    else:
        raise ValueError("Invalid parameters provided.")

    # Define the parameters for the search
    params = {
        "dateRange": "custom",
        "startdt": start_date_str,
        "enddt": end_date_str,
        "category": "custom",
        "forms": forms,
        "q": keyword,   # Search keyword
        "from": 0,
        "size": 100
    }

    return params


def fetch_filings(params: Dict[str, any],
                  base_url: str = "https://efts.sec.gov/LATEST/search-index",
                  headers: Dict[str, str] = headers
                  ) -> List:
    """
    Fetch 8-K filings from the SEC EDGAR Full Text Search API.
    Args:
        params (dict): The search parameters for the API.
        base_url (str): The base URL for the API endpoint.
        headers (dict): The headers to be included in the request.
    Returns:
        list: A list of filings retrieved from the API.
    """

    filings = []
    while True:
        # Send a GET request to the API endpoint
        response = requests.get(base_url, headers=headers, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            print(f"Fetching data from {params['from']} to {params['from'] + params['size']}")
            # Parse the JSON response
            data = response.json()
            
            # Extract the filings
            hits = data.get('hits', {}).get('hits', [])
            if not hits:
                break
            filings.extend(hits)
            
            # Check if there are more results to fetch
            if len(hits) < params['size']:
                break
            
            # Update the 'from' parameter to fetch the next set of results
            params['from'] += params['size']
        else:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            break
    
    return filings


def construct_filings_df(all_filings: list) -> pd.DataFrame:
    """
    Construct a DataFrame with the relevant information from the filings in search result.
    Args:
        all_filings (list): A list of filings retrieved from the API.
    Returns:
        pd.DataFrame: A DataFrame containing the relevant information from the filings.
    """
    
    filing_data = []
    for filing in all_filings:
        source = filing.get('_source', {})
        file_name = transform_filename(filing.get('_id', ''))
        if source and file_name:
            # Extract the relevant information
            cik = source.get('ciks', [''])[0] 
            company = source.get('display_names', [''])[0]
            try:
                company_name, ticker, _ = extract_info(company)
            except:
                print(company)
                company_name, ticker = None, None
            filing_date = source.get('file_date', '')
            form = source.get('form', '')
            file_type = source.get('file_type', '')
            file_description = source.get('file_description', '')
            items = source.get('items', [])
            url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{file_name}"
            
            # Construct a dictionary with the data
            filing_info = {
                "cik": cik,
                "company_name": company_name,
                "ticker": ticker,
                "form": form,
                "filing_date": filing_date,
                "file_type": file_type,
                "file_description": file_description,
                "items": items,
                "form_url": url
            }

            # Append the dictionary to the list
            filing_data.append(filing_info)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(filing_data)

    return df


def get_sec_soup(url: str, headers: dict = headers_sec, save: bool = False) -> BeautifulSoup:
    """
    Get the BeautifulSoup object for a given SEC filing URL.
    Args:
        url (str): The URL of the SEC filing.
        headers (dict): The headers to be included in the request.
        save (bool): Whether to save the HTML content to a file.
    Returns:
        BeautifulSoup: The parsed HTML content of the SEC filing.
    """

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        if save:
            with open(f"{url.split('/')[-1]}.html", 'w') as f:
                f.write(response.text)
        return BeautifulSoup(response.text, 'html.parser')
    else:
        raise Exception(f"Error {response.status_code}: Unable to retrieve data from {url}")
    

def extract_context_around_keyword(soup: BeautifulSoup, 
                                   context_tags: int, 
                                   keyword: str = 'CFIUS', 
                                   tags_to_search=['p', 'td', 'div']) -> str:
    """
    Extract the context around a keyword in the text of an HTML document. 
    Note that only the first occurrence of the keyword is considered.
    Args:
        soup (BeautifulSoup): The BeautifulSoup object containing the parsed HTML content.
        context_tags (int): The number of context tags to extract before and after the keyword.
        keyword (str): The keyword to search for in the text.
        tags_to_search (list): A list of HTML tags to search for the keyword.
    Returns:
        str: The extracted context around the keyword. 
    """

    # Find all tags of the specified type
    tags = soup.find_all(tags_to_search)
    
    # Initialize an empty list to store the context
    context = []
    
    # Loop through the tags to find the keyword and extract context
    for i, element in enumerate(tags):
        if keyword in element.get_text():
            # Get the start and end indices for the context
            start_index = max(0, i - context_tags)
            end_index = min(len(tags), i + context_tags + 1)
            
            # Extract the context tags
            context_tags = tags[start_index:end_index]
            
            # Add the text of each context tag to the context list
            for context_tag in context_tags:
                text = context_tag.get_text()
                # Check if there are alphabets in text
                if re.search('[a-zA-Z]', text):
                    context.append(re.sub(r'\s+', ' ', text))

            break
    
    # Join the context into a single string
    context_string = " ".join(context)
    
    return context_string


def label_filing(df_8k: pd.DataFrame,
                 rerun: bool = False,
                 save_8k: bool = False) -> pd.DataFrame:
    """
    Extract relevant text from each form and classify it. This function 
    pulls the newest data from edgar and appends to the existing data.
    Args:
        df_8k (pd.DataFrame): The DataFrame containing the 8-K filings.
        rerun (bool): Whether to rerun the classification for all filings.
        save_8k (bool): Whether to save the DataFrame to a CSV file.
    Returns:
        pd.DataFrame: The DataFrame with the relevant text and classification results.
    """
    
    # Check if the cache exists
    if os.path.exists('data/8k_labeled.csv') and not rerun:
        df_cache = pd.read_csv('data/8k_labeled.csv')
        # Concatenate the two DataFrames vertically
        df_8k = pd.concat([df_8k, df_cache], axis=0)

        # Remove duplicate entries based on 'form_url' and reset the index
        df_8k.drop_duplicates(subset='form_url', keep='last', inplace=True)
        df_8k.reset_index(drop=True, inplace=True)
    else:
        # Initialize a new column to store the relevant text, label, and score
        df_8k['relevant_text'] = ""
        df_8k['label'] = ""
        df_8k['score'] = 0.0

    for i, row in df_8k.iterrows():
        if i % 10 == 0:
            print(f"Processing filing {i+1}/{len(df_8k)}")

        # If the row has already been processed, skip it
        if row['label']:
            continue
        
        url = row['form_url']
        soup = get_sec_soup(url, save=save_8k)  # Assuming this is a function to get the parsed HTML content
        relevant_text = extract_context_around_keyword(soup, context_tags=1)  # Assuming this function extracts relevant text

        # If extraction is not successful
        if not relevant_text:
            df_8k.to_csv('data/8k_labeled.csv', index=False)
            return df_8k.at[i, 'form_url']

        else:
            # Update the 'relevant_text' column using the DataFrame's 'at' method
            df_8k.at[i, 'relevant_text'] = relevant_text
            result = zero_shot_classification(relevant_text)  # Assuming this function returns classification results
            df_8k.at[i, 'label'] = result['labels'][0]
            df_8k.at[i, 'score'] = result['scores'][0]

    df_8k.to_csv('data/8k_labeled.csv', index=False)

    return df_8k


def label_cfius_reviews(end_date: Optional[str] = None,
                        start_date: Optional[str] = None,
                        recent_n_day: Optional[int] = None,
                        rerun: bool = False,
                        save_8k: bool = False
                        ) -> pd.DataFrame:
    """
    Fetch all 8-K forms filed within a timeframe relevant to CFIUS, extract relevant texts, 
    and classify the CIFUS reviews into announcement, approval, and denial.
    Args:
        end_date (str): The end date for the search in the format 'YYYY-MM-DD'.
        start_date (str): The start date for the search in the format 'YYYY-MM-DD'.
        recent_n_day (int): The number of recent days to search.
        rerun (bool): Whether to rerun the classification for all filings.
        save_8k (bool): Whether to save the DataFrame to a CSV file.
    Returns:
        pd.DataFrame: The DataFrame with the relevant text and classification results.
    """

    # Data collection
    params = define_search(end_date=end_date, start_date=start_date, recent_n_day=recent_n_day)
    all_filings = fetch_filings(params)

    # Data engineering
    df = construct_filings_df(all_filings)
    df_8k = df[(df['form'] == '8-K') & (~df['file_type'].str.contains('EX'))].reset_index(drop=True)

    # Data modeling
    df_8k = label_filing(df_8k, rerun=rerun, save_8k=save_8k)

    return df_8k
