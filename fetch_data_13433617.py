from pymongo import MongoClient
import pandas as pd
import time
import os
from pathlib import Path
from bson import ObjectId
import numpy as np
from typing import Optional
import multiprocessing
from datetime import datetime
import threading
from bson.json_util import dumps, loads
import json

class MongoDataFetcher:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str, batch_id: str):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.batch_id = batch_id
        self.cache_dir = 'data_cache'
        self.chunk_size = 100000
        self.processed_records = 0
        self.total_records = 0
        self._lock = threading.Lock()
        Path(self.cache_dir).mkdir(exist_ok=True)
        
    def log_progress(self, message: str) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")

    def clean_document(self, doc: dict) -> dict:
        """Clean MongoDB document by converting non-serializable types"""
        def convert_value(v):
            if isinstance(v, ObjectId):
                return str(v)
            elif isinstance(v, (datetime, np.datetime64)):
                return v.isoformat()
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [convert_value(item) for item in v]
            elif isinstance(v, (int, float, str, bool, type(None))):
                return v
            else:
                return str(v)  # Convert any other types to string
        
        return {k: convert_value(v) for k, v in doc.items()}

    def update_progress(self, chunk_size: int) -> None:
        with self._lock:
            self.processed_records += chunk_size
            progress = (self.processed_records / self.total_records) * 100
            records_per_sec = self.processed_records / (time.time() - self.start_time)
            self.log_progress(
                f"Processed {self.processed_records:,}/{self.total_records:,} records "
                f"({progress:.1f}%) - Rate: {records_per_sec:.0f} records/sec"
            )

    def process_chunk(self, chunk_data: tuple) -> pd.DataFrame:
        """Process a chunk of MongoDB documents"""
        chunk, chunk_num = chunk_data
        start_time = time.time()
        
        # Clean and convert documents before creating DataFrame
        cleaned_chunk = [self.clean_document(doc) for doc in chunk]
        
        try:
            df_chunk = pd.DataFrame(cleaned_chunk)
            
            # Update progress
            self.update_progress(len(chunk))
            
            processing_time = time.time() - start_time
            self.log_progress(
                f"Chunk {chunk_num}: Processed {len(chunk):,} records in {processing_time:.2f} seconds "
                f"({len(chunk)/processing_time:.0f} records/sec)"
            )
            
            return df_chunk
            
        except Exception as e:
            self.log_progress(f"Error processing chunk {chunk_num}: {str(e)}")
            # Return empty DataFrame with same columns to maintain consistency
            return pd.DataFrame(columns=cleaned_chunk[0].keys() if cleaned_chunk else [])

    def fetch_data(self, use_cache: bool = True) -> pd.DataFrame:
        try:
            self.start_time = time.time()
            client = MongoClient(self.mongo_uri, maxPoolSize=None)
            db = client[self.db_name]
            collection = db[self.collection_name]
            
            self.total_records = collection.count_documents({})
            self.log_progress(f"Total documents in collection {self.collection_name}: {self.total_records:,}")

            if self.total_records == 0:
                return pd.DataFrame()

            # Calculate optimal chunk size and number of workers
            num_cores = multiprocessing.cpu_count()
            self.log_progress(f"Using {num_cores} CPU cores for parallel processing")
            
            num_chunks = min(num_cores * 4, self.total_records // self.chunk_size + 1)
            chunk_size = self.total_records // num_chunks + 1

            # Split data into chunks
            chunks = []
            cursor = collection.find({}, batch_size=chunk_size)
            current_chunk = []
            chunk_num = 1

            self.log_progress("Starting to fetch documents from MongoDB...")
            for doc in cursor:
                current_chunk.append(doc)
                if len(current_chunk) >= chunk_size:
                    chunks.append((current_chunk, chunk_num))
                    current_chunk = []
                    chunk_num += 1
            
            if current_chunk:
                chunks.append((current_chunk, chunk_num))

            # Process chunks using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor
            self.log_progress(f"Starting parallel processing of {len(chunks)} chunks...")
            
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                df_chunks = list(executor.map(self.process_chunk, chunks))

            # Remove any empty DataFrames from failed chunks
            df_chunks = [df for df in df_chunks if not df.empty]
            
            if not df_chunks:
                self.log_progress("No valid data chunks processed")
                return pd.DataFrame()

            # Combine chunks
            self.log_progress("Combining processed chunks...")
            df = pd.concat(df_chunks, ignore_index=True)
            
            total_time = time.time() - self.start_time
            avg_rate = self.total_records / total_time
            self.log_progress(
                f"Completed loading {len(df):,} records in {total_time:.1f} seconds "
                f"(average rate: {avg_rate:.0f} records/sec)"
            )

            # Save to cache if requested
            if use_cache:
                try:
                    cache_path = os.path.join(self.cache_dir, f'batch_{self.batch_id}.parquet')
                    self.log_progress(f"Caching {len(df):,} records to {cache_path}...")
                    df.to_parquet(cache_path, index=False)
                    self.log_progress("Cache save completed successfully")
                except Exception as e:
                    self.log_progress(f"Warning: Failed to cache data: {str(e)}")

            return df

        except Exception as e:
            self.log_progress(f"Error in fetch_data: {str(e)}")
            return pd.DataFrame()
        finally:
            if 'client' in locals():
                client.close()

def main():
    # Configuration
    MONGO_URI = "mongodb://iRecon:AppleiRecon%2314210@rn3-irecont-lmdb08.rno.apple.com:10906/CCiRecon_I001_DEV0?authSource=CCiRecon_I001_DEV0"
    DB_NAME = 'CCiRecon_I166_DEV0'
    COLLECTION_NAME = 'match_batchId_13433617'
    BATCH_ID = '13433617'

    try:
        fetcher = MongoDataFetcher(MONGO_URI, DB_NAME, COLLECTION_NAME, BATCH_ID)
        fetcher.log_progress(f"Starting data fetch for collection: {COLLECTION_NAME}")
        
        df = fetcher.fetch_data(use_cache=True)
        
        if df.empty:
            fetcher.log_progress("No data fetched")
        else:
            fetcher.log_progress(f"Final data shape: {df.shape}")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()