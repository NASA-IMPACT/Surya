"""
Advanced SDO Examples - Building on the basic fetcher
Demonstrates monitoring, time-series, and composite image creation
"""

import time
from datetime import datetime, timezone
from pathlib import Path
from sdo_fetcher_v2 import SDOFetcher


def continuous_monitor(interval_seconds=300, sources=None, provider="auto"):
    """
    Continuously monitor and download SDO images at specified intervals
    
    Args:
        interval_seconds: Time between downloads (default: 5 minutes)
        sources: List of sources to monitor (default: AIA_171)
    """
    if sources is None:
        sources = ["AIA_171"]
    
    fetcher = SDOFetcher(output_dir="monitoring")
    
    print(f"Starting continuous monitoring...")
    print(f"Sources: {', '.join(sources)}")
    print(f"Interval: {interval_seconds} seconds")
    print(f"Provider mode: {provider}")
    print(f"Press Ctrl+C to stop\n")
    
    iteration = 0
    try:
        while True:
            iteration += 1
            timestamp = datetime.now(timezone.utc).isoformat()
            
            print(f"\n{'='*60}")
            print(f"Iteration #{iteration} at {timestamp}")
            print(f"{'='*60}")
            
            for source in sources:
                try:
                    result = fetcher.get_latest_image_direct(source, provider=provider)
                    if result:
                        print(f"✓ {source} downloaded successfully via {result.get('provider_name', result.get('provider', 'unknown'))}")
                    else:
                        print(f"✗ {source} failed")
                except Exception as e:
                    print(f"✗ Error downloading {source}: {e}")
            
            print(f"\nWaiting {interval_seconds} seconds until next download...")
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Downloaded {iteration} sets of images.")
        print(f"Images saved in: {fetcher.output_dir}")


def download_comparison_set():
    """
    Download a comparison set of multiple wavelengths
    Useful for multi-wavelength solar analysis
    """
    print("\n" + "="*60)
    print("Downloading Multi-Wavelength Comparison Set")
    print("="*60 + "\n")
    
    # Select complementary wavelengths
    sources = [
        "AIA_171",  # Quiet corona
        "AIA_193",  # Active regions
        "AIA_304",  # Prominences
        "AIA_211",  # Active regions (hotter)
        "HMI_Magnetogram",  # Magnetic field
        "HMI_Continuum",  # Visible surface
    ]
    
    fetcher = SDOFetcher(output_dir="comparison_set")
    results = fetcher.download_multiple(sources, provider="auto")
    
    print("\n" + "="*60)
    print("Comparison Set Complete!")
    print("="*60)
    print(f"Downloaded {len(results)} images")
    print("\nUse these for:")
    print("  - Multi-wavelength composite images")
    print("  - Temperature analysis")
    print("  - Active region identification")
    print("  - Prominence and filament studies")
    print("="*60 + "\n")
    
    return results


def download_active_region_set():
    """
    Download wavelengths optimal for observing active regions and flares
    """
    print("\n" + "="*60)
    print("Downloading Active Region / Flare Observation Set")
    print("="*60 + "\n")
    
    # Wavelengths best for active regions and flares
    sources = [
        "AIA_94",   # Hot flare plasma
        "AIA_131",  # Flaring regions
        "AIA_193",  # Active regions
        "AIA_211",  # Active regions
        "HMI_Magnetogram",  # Magnetic field
    ]
    
    fetcher = SDOFetcher(output_dir="active_regions")
    results = fetcher.download_multiple(sources, provider="auto")
    
    print("\nActive region monitoring complete!")
    print("Check these images for:")
    print("  - Solar flares (bright spots in 94Å and 131Å)")
    print("  - Active region structure (193Å, 211Å)")
    print("  - Sunspot magnetic complexity (HMI Magnetogram)")
    
    return results


def quick_space_weather_check():
    """
    Quick download for space weather assessment
    """
    print("\n" + "="*60)
    print("SPACE WEATHER QUICK CHECK")
    print("="*60 + "\n")
    
    fetcher = SDOFetcher(output_dir="space_weather")
    
    # Get the most relevant images for space weather
    sources = ["AIA_193", "HMI_Magnetogram"]
    
    print("Downloading key space weather indicators...")
    results = fetcher.download_multiple(sources, provider="auto")
    
    if len(results) == 2:
        print("\n" + "="*60)
        print("READY FOR ANALYSIS")
        print("="*60)
        print("\nCheck the images for:")
        print("  📸 AIA 193: Active regions and coronal holes")
        print("  🧲 HMI Magnetogram: Complex magnetic fields (flare potential)")
        print("\nLook for:")
        print("  ⚠️  Dark regions = coronal holes → fast solar wind")
        print("  ⚠️  Bright active regions = potential for flares")
        print("  ⚠️  Complex magnetograms = higher flare risk")
        print("="*60 + "\n")
    
    return results


def download_prominence_monitoring():
    """
    Download wavelengths optimal for prominence/filament observation
    """
    print("\n" + "="*60)
    print("Prominence/Filament Monitoring Set")
    print("="*60 + "\n")
    
    # Best wavelengths for prominences
    sources = [
        "AIA_304",  # Primary prominence wavelength
        "AIA_171",  # Context (corona)
        "HMI_Continuum",  # Visible disk
    ]
    
    fetcher = SDOFetcher(output_dir="prominences")
    results = fetcher.download_multiple(sources, provider="auto")
    
    print("\nProminence monitoring complete!")
    print("304Å is best for seeing prominences on the solar limb")
    
    return results


def create_monitoring_script():
    """
    Generate a standalone monitoring script
    """
    script_content = '''#!/usr/bin/env python3
"""
Automated SDO Monitoring Script
Runs continuously and downloads images every 15 minutes
"""

import time
from datetime import datetime, timezone
from sdo_fetcher_v2 import SDOFetcher
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sdo_monitor.log'),
        logging.StreamHandler()
    ]
)

def main():
    fetcher = SDOFetcher(output_dir="continuous_monitoring")
    sources = ["AIA_171", "AIA_193", "HMI_Magnetogram"]
    interval = 900  # 15 minutes
    
    logging.info("Starting SDO continuous monitoring")
    logging.info(f"Sources: {sources}")
    logging.info(f"Interval: {interval} seconds")
    
    iteration = 0
    while True:
        try:
            iteration += 1
            logging.info(f"=== Iteration {iteration} ===")
            
            for source in sources:
                try:
                    result = fetcher.get_latest_image_direct(source, provider="auto")
                    if result:
                        logging.info(f"✓ Downloaded {source} via {result.get('provider_name', result.get('provider', 'unknown'))}")
                except Exception as e:
                    logging.error(f"✗ Failed to download {source}: {e}")
            
            logging.info(f"Waiting {interval} seconds...")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logging.info("Monitoring stopped by user")
            break
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    main()
'''
    
    with open("monitoring_daemon.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("\n✓ Created 'monitoring_daemon.py'")
    print("Run it with: python monitoring_daemon.py")
    print("It will continuously download SDO images every 15 minutes")


def main():
    """Main menu for advanced examples"""
    print("\n" + "="*60)
    print("SDO Advanced Examples")
    print("="*60)
    print("\n1. Download multi-wavelength comparison set")
    print("2. Download active region/flare observation set")
    print("3. Quick space weather check")
    print("4. Download prominence monitoring set")
    print("5. Start continuous monitoring (Ctrl+C to stop)")
    print("6. Create monitoring daemon script")
    print("7. Exit")
    
    choice = input("\nSelect option (1-7): ").strip()
    
    if choice == "1":
        download_comparison_set()
    elif choice == "2":
        download_active_region_set()
    elif choice == "3":
        quick_space_weather_check()
    elif choice == "4":
        download_prominence_monitoring()
    elif choice == "5":
        sources = input("Enter sources (comma-separated, or press Enter for AIA_171): ").strip()
        if sources:
            sources = [s.strip() for s in sources.split(",")]
        else:
            sources = ["AIA_171"]
        interval = input("Enter interval in seconds (default 300): ").strip()
        interval = int(interval) if interval else 300
        provider = input("Enter provider (auto/lmsal/jsoc/nasa/helioviewer, default auto): ").strip() or "auto"
        continuous_monitor(interval, sources, provider)
    elif choice == "6":
        create_monitoring_script()
    elif choice == "7":
        print("Goodbye!")
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
