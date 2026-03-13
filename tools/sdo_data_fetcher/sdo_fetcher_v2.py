"""
SDO Data Fetcher v2 - Alternative Implementation
Uses NASA's Helioviewer.org latest images API
"""

import requests
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional, Dict
import argparse
from sdo_provider import SDOProviderClient, SDO_SOURCES


class SDOFetcher:
    """Simplified SDO data fetcher using Helioviewer's latest images"""
    
    SDO_SOURCES = SDO_SOURCES
    
    def __init__(self, output_dir: str = "sdo_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.provider_client = SDOProviderClient(output_dir=output_dir)
    
    def get_latest_image_png(self, source: str = "AIA_171", provider: str = "auto") -> Optional[Dict]:
        """
        Fetch latest SDO image as PNG using a simpler method
        
        Args:
            source: SDO source identifier
            
        Returns:
            Dictionary with metadata and filepath
        """
        return self.provider_client.download_latest_image(source=source, provider=provider)
    
    def get_latest_image_direct(self, source: str = "AIA_171", provider: str = "auto") -> Optional[Dict]:
        """
        Alternative method: Fetch from SDO's direct image feed
        Uses helioviewer.org's pre-rendered latest images
        """
        result = self.provider_client.download_latest_image(source=source, provider=provider)
        if result:
            print(f"\n{'='*60}")
            print(f"Success! Downloaded latest SDO {source} image")
            print(f"Provider: {result.get('provider_name', result.get('provider', 'unknown'))}")
            if result.get("observation_time"):
                print(f"Observation time: {result['observation_time']}")
            print(f"{'='*60}\n")
        return result
    
    def download_multiple(self, sources: list = None, provider: str = "auto"):
        """Download multiple wavelengths"""
        if sources is None:
            sources = ["AIA_171", "AIA_193", "AIA_304", "HMI_Magnetogram"]
        
        print(f"\nDownloading {len(sources)} different SDO images...")
        print("="*60)
        
        results = []
        for source in sources:
            result = self.get_latest_image_direct(source, provider=provider)
            if result:
                results.append(result)
        
        print(f"\n{'='*60}")
        print(f"Successfully downloaded {len(results)}/{len(sources)} images")
        print(f"{'='*60}\n")
        
        return results
    
    @staticmethod
    def list_sources():
        """List all available sources"""
        print("\n" + "="*60)
        print("Available SDO Data Sources")
        print("="*60)
        for key, info in SDOFetcher.SDO_SOURCES.items():
            print(f"  {key:20} - {info['name']} ({info['wavelength']})")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SDO Data Fetcher v2 - Fetch latest solar images from NASA's SDO",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--source', '-s', default='AIA_171',
                       help='SDO source (default: AIA_171)')
    parser.add_argument('--output', '-o', default='sdo_data',
                       help='Output directory (default: sdo_data)')
    parser.add_argument('--multiple', '-m', action='store_true',
                       help='Download multiple wavelengths')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available sources')
    parser.add_argument('--provider', '-p', default='auto',
                       help='Data provider: auto, lmsal, jsoc, nasa, helioviewer')
    
    args = parser.parse_args()
    
    if args.list:
        SDOFetcher.list_sources()
        SDOProviderClient.list_providers()
        return
    
    fetcher = SDOFetcher(output_dir=args.output)
    
    if args.multiple:
        fetcher.download_multiple(provider=args.provider)
    else:
        fetcher.get_latest_image_direct(source=args.source, provider=args.provider)


if __name__ == "__main__":
    main()
