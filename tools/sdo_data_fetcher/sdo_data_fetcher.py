"""
SDO (Solar Dynamic Observatory) Data Fetcher

This script fetches the latest solar images from NASA's Solar Dynamic Observatory
using the Helioviewer API. It supports multiple instruments and wavelengths.
"""

import requests
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Optional, Dict
import argparse
from sdo_provider import SDOProviderClient, SDO_SOURCES


class SDODataFetcher:
    """Fetches latest SDO data from Helioviewer API"""
    
    BASE_URL = "https://api.helioviewer.org/v2/"
    SDO_SOURCES = SDO_SOURCES
    
    def __init__(self, output_dir: str = "sdo_data"):
        """Initialize the fetcher with an output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.provider_client = SDOProviderClient(output_dir=output_dir)
    
    def get_latest_available_date(self, source: str = "AIA_171", provider: str = "auto") -> Optional[str]:
        """Query the API for the latest available SDO observation time"""
        timestamp = self.provider_client.get_latest_timestamp(source=source, provider=provider)
        if timestamp:
            print(f"Latest SDO data available: {timestamp}")
            return timestamp

        fallback = (datetime.now(timezone.utc) - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        print(f"Using fallback date: {fallback}")
        return fallback
    
    def get_latest_image(self, source: str = "AIA_171", image_scale: float = 2.4, provider: str = "auto") -> Optional[Dict]:
        """
        Fetch the latest SDO image
        
        Args:
            source: SDO source identifier (e.g., 'AIA_171', 'HMI_Magnetogram')
            image_scale: Resolution in arcseconds per pixel (lower = higher resolution)
            
        Returns:
            Dictionary with image metadata and file path
        """
        _ = image_scale
        if source not in self.SDO_SOURCES:
            raise ValueError(f"Invalid source. Choose from: {list(self.SDO_SOURCES.keys())}")

        print(f"Fetching latest {source} data...")
        print(f"Description: {self.SDO_SOURCES[source]['description']}")
        return self.provider_client.download_latest_image(source=source, provider=provider)
    
    def get_latest_data_timestamp(self, source: str = "AIA_171", provider: str = "auto") -> Optional[str]:
        """Get the timestamp of the latest available SDO data"""
        try:
            return self.provider_client.get_latest_timestamp(source=source, provider=provider)
        except requests.exceptions.RequestException as e:
            print(f"Error getting latest timestamp: {e}")
            return None
    
    def download_multiple_wavelengths(self, sources: list = None, provider: str = "auto"):
        """
        Download images from multiple SDO sources
        
        Args:
            sources: List of source identifiers (defaults to common wavelengths)
        """
        if sources is None:
            sources = ["AIA_171", "AIA_193", "AIA_304", "HMI_Magnetogram"]
        
        results = []
        for source in sources:
            print(f"\n{'='*60}")
            result = self.get_latest_image(source, provider=provider)
            if result:
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Downloaded {len(results)} images successfully!")
        return results
    
    @staticmethod
    def list_available_sources():
        """Print all available SDO sources"""
        print("\nAvailable SDO Data Sources:")
        print("="*60)
        for key, value in SDODataFetcher.SDO_SOURCES.items():
            print(f"{key:20} - {value['description']}")
        print("\nAvailable providers:")
        SDOProviderClient.list_providers()


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Fetch latest SDO (Solar Dynamic Observatory) data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download latest AIA 171 Å image
  python sdo_data_fetcher.py
  
  # Download specific wavelength
  python sdo_data_fetcher.py --source AIA_304
  
  # Download multiple wavelengths
  python sdo_data_fetcher.py --multiple
  
  # List all available sources
  python sdo_data_fetcher.py --list
  
  # Download to specific directory
  python sdo_data_fetcher.py --output my_sdo_images
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='AIA_171',
        help='SDO source to fetch (default: AIA_171)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='sdo_data',
        help='Output directory for downloaded data (default: sdo_data)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=2.4,
        help='Image scale in arcseconds per pixel (default: 2.4)'
    )
    
    parser.add_argument(
        '--multiple', '-m',
        action='store_true',
        help='Download multiple common wavelengths'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available SDO sources'
    )
    
    parser.add_argument(
        '--timestamp', '-t',
        action='store_true',
        help='Get the timestamp of latest available data'
    )
    
    parser.add_argument(
        '--provider', '-p',
        type=str,
        default='auto',
        help='Data provider: auto, lmsal, jsoc, nasa, helioviewer'
    )
    
    args = parser.parse_args()
    
    # List available sources and exit
    if args.list:
        SDODataFetcher.list_available_sources()
        return
    
    # Initialize fetcher
    fetcher = SDODataFetcher(output_dir=args.output)
    
    # Get latest timestamp
    if args.timestamp:
        timestamp = fetcher.get_latest_data_timestamp(source=args.source, provider=args.provider)
        if timestamp:
            print(f"Latest SDO data available at: {timestamp}")
        return
    
    # Download data
    if args.multiple:
        fetcher.download_multiple_wavelengths(provider=args.provider)
    else:
        fetcher.get_latest_image(source=args.source, image_scale=args.scale, provider=args.provider)


if __name__ == "__main__":
    main()
