#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional
import re

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# Configuration
OUTPUT_DIR = './output'
SCRAPED_DATA_FILE = os.path.join(OUTPUT_DIR, 'scraped_data.csv')
RESULT_FILE = os.path.join(OUTPUT_DIR, 'result.json')
URL = 'https://en.wikipedia.org/wiki/List_of_highest-grossing_films'
MAX_RETRIES = 3
RETRY_DELAY = 3  # seconds
REQUEST_TIMEOUT = 30

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logging.info(f"Output directory ensured: {OUTPUT_DIR}")
    except Exception as e:
        logging.error(f'Failed to create output directory {OUTPUT_DIR}: {e}')
        sys.exit(1)

def fetch_page_with_retry(url: str, max_retries: int = MAX_RETRIES) -> requests.Response:
    """Fetch webpage with retry logic"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            logging.info(f'Fetching page (attempt {attempt}/{max_retries})...')
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            logging.info(f'Successfully fetched page on attempt {attempt}')
            return response
        except requests.exceptions.RequestException as e:
            logging.warning(f'Request failed on attempt {attempt}: {e}')
            if attempt < max_retries:
                logging.info(f'Retrying in {RETRY_DELAY} seconds...')
                time.sleep(RETRY_DELAY)
            else:
                logging.error('Max retries reached for page fetching')
                raise

def find_target_table(soup: BeautifulSoup) -> BeautifulSoup:
    """Find the target table with robust fallback logic"""
    try:
        # Strategy 1: Look for table with specific caption containing "Highest-grossing"
        tables = soup.find_all('table', class_='wikitable')
        logging.info(f'Found {len(tables)} wikitable(s)')
        
        for i, table in enumerate(tables):
            caption = table.find('caption')
            if caption:
                caption_text = caption.get_text(strip=True)
                logging.info(f'Table {i} caption: {caption_text[:100]}...')
                if any(keyword in caption_text.lower() for keyword in ['highest-grossing', 'highest grossing', 'box office']):
                    logging.info(f'Found target table by caption at index {i}')
                    return table
        
        # Strategy 2: Look for table with headers that match expected columns
        for i, table in enumerate(tables):
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True).lower() for th in header_row.find_all(['th', 'td'])]
                logging.info(f'Table {i} headers: {headers[:5]}...')
                if any(keyword in ' '.join(headers) for keyword in ['rank', 'film', 'worldwide', 'gross', 'year']):
                    logging.info(f'Found target table by headers at index {i}')
                    return table
        
        # Strategy 3: Use first table as fallback
        if tables:
            logging.warning('Using first table as fallback')
            return tables[0]
        
        raise ValueError('No tables found on page')
        
    except Exception as e:
        logging.error(f'Error finding target table: {e}')
        raise

def clean_currency_value(value: str) -> float:
    """Clean currency values and convert to float"""
    if not value or value.strip() == '':
        return 0.0
    
    # Remove currency symbols, commas, and whitespace
    cleaned = re.sub(r'[\$,\s]', '', value)
    
    # Handle billion (B) and million (M) suffixes
    if 'billion' in value.lower() or value.endswith('B'):
        multiplier = 1_000_000_000
        cleaned = re.sub(r'[bB].*$', '', cleaned)
    elif 'million' in value.lower() or value.endswith('M'):
        multiplier = 1_000_000
        cleaned = re.sub(r'[mM].*$', '', cleaned)
    else:
        multiplier = 1
    
    try:
        return float(cleaned) * multiplier
    except (ValueError, TypeError):
        logging.warning(f'Could not parse currency value: {value}')
        return 0.0

def clean_year_value(value: str) -> int:
    """Extract year from value"""
    if not value:
        return 0
    
    # Extract 4-digit year
    year_match = re.search(r'\b(19|20)\d{2}\b', value)
    if year_match:
        return int(year_match.group())
    
    return 0

def parse_table_data(table: BeautifulSoup) -> pd.DataFrame:
    """Parse table data into pandas DataFrame with robust error handling"""
    try:
        rows = table.find_all('tr')
        if not rows:
            raise ValueError('No rows found in table')
        
        # Extract headers
        header_row = rows[0]
        headers = []
        for cell in header_row.find_all(['th', 'td']):
            header_text = cell.get_text(strip=True)
            headers.append(header_text if header_text else f'Column_{len(headers)}')
        
        logging.info(f'Table headers: {headers}')
        
        # Extract data rows
        data_rows = []
        for row_idx, row in enumerate(rows[1:], 1):
            cells = row.find_all(['td', 'th'])
            if not cells:
                continue
                
            row_data = []
            for cell_idx, cell in enumerate(cells):
                cell_text = cell.get_text(strip=True)
                # Handle cells with links
                link = cell.find('a')
                if link and not cell_text:
                    cell_text = link.get_text(strip=True)
                row_data.append(cell_text)
            
            # Ensure row has same number of columns as headers
            while len(row_data) < len(headers):
                row_data.append('')
            
            if len(row_data) > len(headers):
                row_data = row_data[:len(headers)]
            
            data_rows.append(row_data)
            
            # Log first few rows for debugging
            if row_idx <= 3:
                logging.info(f'Row {row_idx}: {row_data[:3]}...')
        
        if not data_rows:
            raise ValueError('No data rows found in table')
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        logging.info(f'Created DataFrame with shape: {df.shape}')
        
        return df
        
    except Exception as e:
        logging.error(f'Error parsing table data: {e}')
        raise

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process and clean the DataFrame"""
    try:
        logging.info('Processing DataFrame...')
        
        # Identify key columns (case-insensitive)
        rank_col = None
        gross_col = None
        year_col = None
        film_col = None
        peak_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'rank' in col_lower and rank_col is None:
                rank_col = col
            elif any(word in col_lower for word in ['worldwide', 'gross', 'box office']) and gross_col is None:
                gross_col = col
            elif 'year' in col_lower and year_col is None:
                year_col = col
            elif any(word in col_lower for word in ['film', 'title', 'movie']) and film_col is None:
                film_col = col
            elif 'peak' in col_lower and peak_col is None:
                peak_col = col
        
        logging.info(f'Identified columns - Rank: {rank_col}, Gross: {gross_col}, Year: {year_col}, Film: {film_col}, Peak: {peak_col}')
        
        # Clean and convert data types
        if rank_col:
            df['Rank_Clean'] = pd.to_numeric(df[rank_col].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        
        if gross_col:
            df['Gross_Clean'] = df[gross_col].apply(clean_currency_value)
        
        if year_col:
            df['Year_Clean'] = df[year_col].apply(clean_year_value)
        
        if peak_col:
            df['Peak_Clean'] = pd.to_numeric(df[peak_col].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
        
        # Remove rows with invalid data
        initial_rows = len(df)
        if 'Rank_Clean' in df.columns:
            df = df.dropna(subset=['Rank_Clean'])
        
        logging.info(f'Cleaned DataFrame: {initial_rows} -> {len(df)} rows')
        
        return df
        
    except Exception as e:
        logging.error(f'Error processing DataFrame: {e}')
        return df

def save_scraped_data(df: pd.DataFrame):
    """Save scraped data to CSV file"""
    try:
        df.to_csv(SCRAPED_DATA_FILE, index=False)
        logging.info(f'Scraped data saved to: {SCRAPED_DATA_FILE}')
    except Exception as e:
        logging.error(f'Failed to save scraped data: {e}')
        raise

def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the scraped data and answer questions"""
    try:
        results = {}
        
        # Question 1: How many $2 bn movies were released before 2000?
        if 'Gross_Clean' in df.columns and 'Year_Clean' in df.columns:
            two_bn_before_2000 = df[
                (df['Gross_Clean'] >= 2_000_000_000) & 
                (df['Year_Clean'] < 2000) & 
                (df['Year_Clean'] > 0)
            ]
            results['q1_answer'] = len(two_bn_before_2000)
            logging.info(f'Q1: {results["q1_answer"]} movies grossed $2bn+ before 2000')
        else:
            results['q1_answer'] = "Unable to determine - missing gross or year data"
        
        # Question 2: Which is the earliest film that grossed over $1.5 bn?
        if 'Gross_Clean' in df.columns and 'Year_Clean' in df.columns:
            over_1_5bn = df[
                (df['Gross_Clean'] >= 1_500_000_000) & 
                (df['Year_Clean'] > 0)
            ].sort_values('Year_Clean')
            
            if not over_1_5bn.empty:
                earliest_film = over_1_5bn.iloc[0]
                film_name = earliest_film.get(next((col for col in df.columns if 'film' in col.lower() or 'title' in col.lower()), df.columns[0]), 'Unknown')
                results['q2_answer'] = f"{film_name} ({int(earliest_film.get('Year_Clean', 0))})"
            else:
                results['q2_answer'] = "No films found grossing over $1.5bn"
            logging.info(f'Q2: {results["q2_answer"]}')
        else:
            results['q2_answer'] = "Unable to determine - missing gross or year data"
        
        # Question 3: What's the correlation between Rank and Peak?
        if 'Rank_Clean' in df.columns and 'Peak_Clean' in df.columns:
            clean_data = df[['Rank_Clean', 'Peak_Clean']].dropna()
            if len(clean_data) > 1:
                correlation = clean_data['Rank_Clean'].corr(clean_data['Peak_Clean'])
                results['q3_answer'] = round(correlation, 4)
            else:
                results['q3_answer'] = "Insufficient data for correlation"
            logging.info(f'Q3: Correlation = {results["q3_answer"]}')
        else:
            results['q3_answer'] = "Unable to calculate - missing rank or peak data"
        
        # Question 4: Create scatterplot
        plot_data_uri = create_scatterplot(df)
        if plot_data_uri:
            results['q4_answer'] = plot_data_uri
            logging.info('Q4: Scatterplot created successfully')
        else:
            results['q4_answer'] = "Unable to create plot - insufficient data"
        
        return results
        
    except Exception as e:
        logging.error(f'Error analyzing data: {e}')
        return {"error": str(e)}

def create_scatterplot(df: pd.DataFrame) -> Optional[str]:
    """Create scatterplot of Rank vs Peak with regression line"""
    try:
        if 'Rank_Clean' not in df.columns or 'Peak_Clean' not in df.columns:
            logging.warning('Missing Rank_Clean or Peak_Clean columns for plotting')
            return None
        
        # Prepare data
        plot_data = df[['Rank_Clean', 'Peak_Clean']].dropna()
        if len(plot_data) < 2:
            logging.warning('Insufficient data points for plotting')
            return None
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Scatterplot
        plt.scatter(plot_data['Rank_Clean'], plot_data['Peak_Clean'], alpha=0.6, s=50)
        
        # Regression line
        z = np.polyfit(plot_data['Rank_Clean'], plot_data['Peak_Clean'], 1)
        p = np.poly1d(z)
        plt.plot(plot_data['Rank_Clean'], p(plot_data['Rank_Clean']), "r--", alpha=0.8, linewidth=2)
        
        plt.xlabel('Rank')
        plt.ylabel('Peak')
        plt.title('Rank vs Peak - Highest Grossing Films')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        
        # Check size
        img_data = buffer.getvalue()
        if len(img_data) > 100000:  # 100KB limit
            # Reduce quality
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=50, bbox_inches='tight')
            buffer.seek(0)
            img_data = buffer.getvalue()
        
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
        
    except Exception as e:
        logging.error(f'Error creating scatterplot: {e}')
        plt.close()
        return None

def main():
    """Main execution function"""
    setup_logging()
    ensure_output_dir()
    
    try:
        # Check if scraped data already exists
        if os.path.exists(SCRAPED_DATA_FILE):
            logging.info('Loading existing scraped data...')
            df = pd.read_csv(SCRAPED_DATA_FILE)
        else:
            logging.info('Scraping fresh data from Wikipedia...')
            
            # Fetch and parse webpage
            response = fetch_page_with_retry(URL)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find and parse table
            table = find_target_table(soup)
            df = parse_table_data(table)
            df = process_dataframe(df)
            
            # Save scraped data
            save_scraped_data(df)
        
        # Analyze data and answer questions
        logging.info('Analyzing data...')
        analysis_results = analyze_data(df)
        
        # Prepare final results
        final_results = {
            "status": "success",
            "data_shape": df.shape,
            "questions_and_answers": [
                f"Q1: How many $2 bn movies were released before 2000? A: {analysis_results.get('q1_answer', 'N/A')}",
                f"Q2: Which is the earliest film that grossed over $1.5 bn? A: {analysis_results.get('q2_answer', 'N/A')}",
                f"Q3: What's the correlation between the Rank and Peak? A: {analysis_results.get('q3_answer', 'N/A')}",
                f"Q4: Scatterplot created: {'Yes' if analysis_results.get('q4_answer', '').startswith('data:image') else 'No'}"
            ],
            "scatterplot_data_uri": analysis_results.get('q4_answer', ''),
            "summary": f"Successfully analyzed {len(df)} films from Wikipedia"
        }
        
        # Save results
        with open(RESULT_FILE, 'w') as f:
            json.dump(final_results, f, cls=NumpyEncoder, indent=2)
        
        logging.info(f'Analysis completed successfully. Results saved to: {RESULT_FILE}')
        print("SUCCESS: Analysis completed")
        
    except Exception as e:
        error_results = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }
        
        try:
            with open(RESULT_FILE, 'w') as f:
                json.dump(error_results, f, cls=NumpyEncoder, indent=2)
        except:
            pass
        
        logging.error(f'Analysis failed: {e}')
        print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
