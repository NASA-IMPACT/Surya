"""
Shared SDO data provider logic with automatic fallback.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import requests


SDO_SOURCES = {
    "AIA_94": {
        "sourceId": 13,
        "name": "AIA 94",
        "wavelength": "94Å",
        "description": "AIA 94 Å - Hot flare plasma",
        "nasa_code": "0094",
        "lmsal_code": "0094",
    },
    "AIA_131": {
        "sourceId": 14,
        "name": "AIA 131",
        "wavelength": "131Å",
        "description": "AIA 131 Å - Flaring regions",
        "nasa_code": "0131",
        "lmsal_code": "0131",
    },
    "AIA_171": {
        "sourceId": 15,
        "name": "AIA 171",
        "wavelength": "171Å",
        "description": "AIA 171 Å - Quiet corona and coronal loops",
        "nasa_code": "0171",
        "lmsal_code": "0171",
    },
    "AIA_193": {
        "sourceId": 16,
        "name": "AIA 193",
        "wavelength": "193Å",
        "description": "AIA 193 Å - Hot plasma in active regions",
        "nasa_code": "0193",
        "lmsal_code": "0193",
    },
    "AIA_211": {
        "sourceId": 17,
        "name": "AIA 211",
        "wavelength": "211Å",
        "description": "AIA 211 Å - Active regions",
        "nasa_code": "0211",
        "lmsal_code": "0211",
    },
    "AIA_304": {
        "sourceId": 18,
        "name": "AIA 304",
        "wavelength": "304Å",
        "description": "AIA 304 Å - Chromosphere and prominence",
        "nasa_code": "0304",
        "lmsal_code": "0304",
    },
    "AIA_335": {
        "sourceId": 19,
        "name": "AIA 335",
        "wavelength": "335Å",
        "description": "AIA 335 Å - Active regions",
        "nasa_code": "0335",
        "lmsal_code": "0335",
    },
    "AIA_1600": {
        "sourceId": 20,
        "name": "AIA 1600",
        "wavelength": "1600Å",
        "description": "AIA 1600 Å - Upper photosphere",
        "nasa_code": "1600",
        "lmsal_code": "1600",
    },
    "AIA_1700": {
        "sourceId": 21,
        "name": "AIA 1700",
        "wavelength": "1700Å",
        "description": "AIA 1700 Å - Temperature minimum",
        "nasa_code": "1700",
        "lmsal_code": "1700",
    },
    "HMI_Continuum": {
        "sourceId": 22,
        "name": "HMI Continuum",
        "wavelength": "Continuum",
        "description": "HMI Continuum - Solar surface",
        "nasa_code": "HMIIC",
        "lmsal_code": "_HMI_cont_aiascale",
        "jsoc_path": "/data/hmi/images/latest/HMI_latest_Int_1024x1024.gif",
        "jsoc_timestamp_key": "continuum",
    },
    "HMI_Magnetogram": {
        "sourceId": 23,
        "name": "HMI Magnetogram",
        "wavelength": "Magnetogram",
        "description": "HMI Magnetogram - Magnetic field",
        "nasa_code": "HMII",
        "lmsal_code": "_HMImag",
        "jsoc_path": "/data/hmi/images/latest/HMI_latest_Mag_1024x1024.gif",
        "jsoc_timestamp_key": "magnetogram",
    },
}


PROVIDER_LABELS = {
    "lmsal": "LMSAL Sun Today",
    "jsoc": "Stanford JSOC",
    "nasa": "NASA SDO",
    "helioviewer": "Helioviewer API",
}


AUTO_PROVIDER_ORDER = ("lmsal", "jsoc", "nasa", "helioviewer")


class SDOProviderClient:
    """Download latest SDO imagery from multiple redundant providers."""

    def __init__(self, output_dir: str = "sdo_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()

    def download_latest_image(self, source: str = "AIA_171", provider: str = "auto") -> Optional[Dict]:
        """Download the latest image using the requested provider or fallback chain."""
        if source not in SDO_SOURCES:
            raise ValueError(f"Invalid source. Choose from: {list(SDO_SOURCES.keys())}")

        provider_order = self._resolve_provider_order(provider)

        print(f"\nFetching latest {source} image...")
        print(f"Wavelength: {SDO_SOURCES[source]['wavelength']}")
        print(f"Provider order: {', '.join(provider_order)}")

        last_error = None

        for provider_name in provider_order:
            try:
                print(f"Trying provider: {PROVIDER_LABELS[provider_name]}")
                result = getattr(self, f"_download_from_{provider_name}")(source)
                if result:
                    return result
            except requests.exceptions.RequestException as e:
                last_error = e
                print(f"Provider {provider_name} failed: {e}")
            except Exception as e:
                last_error = e
                print(f"Provider {provider_name} failed unexpectedly: {e}")

        if last_error:
            print(f"All providers failed. Last error: {last_error}")
        else:
            print("All providers failed.")
        return None

    def get_latest_timestamp(self, source: str = "AIA_171", provider: str = "auto") -> Optional[str]:
        """Best-effort timestamp lookup using the same provider order."""
        if source not in SDO_SOURCES:
            raise ValueError(f"Invalid source. Choose from: {list(SDO_SOURCES.keys())}")

        for provider_name in self._resolve_provider_order(provider):
            try:
                timestamp = getattr(self, f"_timestamp_from_{provider_name}")(source)
                if timestamp:
                    return timestamp
            except Exception:
                continue
        return None

    @staticmethod
    def list_providers():
        """Print available provider names."""
        print("\nAvailable data providers:")
        print("=" * 60)
        print("auto         - Automatic fallback chain")
        for key, label in PROVIDER_LABELS.items():
            print(f"{key:12} - {label}")

    def _resolve_provider_order(self, provider: str) -> Iterable[str]:
        provider = provider.lower()
        if provider == "auto":
            return AUTO_PROVIDER_ORDER
        if provider not in PROVIDER_LABELS:
            raise ValueError(f"Invalid provider. Choose from: auto, {', '.join(PROVIDER_LABELS.keys())}")
        return (provider,)

    def _download_from_lmsal(self, source: str) -> Optional[Dict]:
        code = SDO_SOURCES[source].get("lmsal_code")
        if not code:
            return None

        for day_offset in range(0, 4):
            candidate_date = datetime.now(timezone.utc) - timedelta(days=day_offset)
            date_path = candidate_date.strftime("%Y/%m/%d")
            urls = [
                f"http://suntoday.lmsal.com/sdomedia/SunInTime/{date_path}/t{code}.jpg",
                f"https://suntoday.lmsal.com/sdomedia/SunInTime/{date_path}/t{code}.jpg",
            ]

            for url in urls:
                try:
                    response = self.session.get(url, timeout=30, stream=True)
                    if response.status_code == 404:
                        response.close()
                        continue

                    response.raise_for_status()
                    return self._save_response(
                        response=response,
                        source=source,
                        provider="lmsal",
                        image_url=url,
                        extension=".jpg",
                        observation_time=response.headers.get("Last-Modified") or candidate_date.strftime("%Y-%m-%d"),
                        extra_metadata={"date_path": date_path},
                    )
                except requests.exceptions.RequestException:
                    continue

        return None

    def _download_from_jsoc(self, source: str) -> Optional[Dict]:
        jsoc_path = SDO_SOURCES[source].get("jsoc_path")
        if not jsoc_path:
            return None

        url = f"https://jsoc1.stanford.edu{jsoc_path}"
        response = self.session.get(url, timeout=30, stream=True)
        response.raise_for_status()

        return self._save_response(
            response=response,
            source=source,
            provider="jsoc",
            image_url=url,
            extension=Path(jsoc_path).suffix or ".img",
            observation_time=self._timestamp_from_jsoc(source),
        )

    def _download_from_nasa(self, source: str) -> Optional[Dict]:
        nasa_code = SDO_SOURCES[source].get("nasa_code")
        if not nasa_code:
            return None

        urls = [
            f"http://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_{nasa_code}.jpg",
            f"https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_{nasa_code}.jpg",
        ]

        for url in urls:
            try:
                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()

                return self._save_response(
                    response=response,
                    source=source,
                    provider="nasa",
                    image_url=url,
                    extension=".jpg",
                    observation_time=response.headers.get("Last-Modified"),
                )
            except requests.exceptions.RequestException:
                continue

        return None

    def _download_from_helioviewer(self, source: str) -> Optional[Dict]:
        source_id = SDO_SOURCES[source]["sourceId"]
        info_url = "https://api.helioviewer.org/v2/getClosestImage/"
        info_params = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sourceId": source_id,
        }

        info_response = self.session.get(info_url, params=info_params, timeout=30)
        info_response.raise_for_status()
        image_info = info_response.json()
        image_id = image_info.get("id")

        if not image_id:
            return None

        tile_url = "https://api.helioviewer.org/v2/getTile/"
        tile_params = {
            "id": image_id,
            "x": 0,
            "y": 0,
            "imageScale": 2.4,
            "display": "true",
        }
        response = self.session.get(tile_url, params=tile_params, timeout=30, stream=True)
        response.raise_for_status()

        return self._save_response(
            response=response,
            source=source,
            provider="helioviewer",
            image_url=response.url,
            extension=".png",
            observation_time=image_info.get("date"),
            extra_metadata={"image_id": image_id},
        )

    def _timestamp_from_lmsal(self, source: str) -> Optional[str]:
        code = SDO_SOURCES[source].get("lmsal_code")
        if not code:
            return None

        for day_offset in range(0, 4):
            candidate_date = datetime.now(timezone.utc) - timedelta(days=day_offset)
            date_path = candidate_date.strftime("%Y/%m/%d")
            urls = [
                f"http://suntoday.lmsal.com/sdomedia/SunInTime/{date_path}/t{code}.jpg",
                f"https://suntoday.lmsal.com/sdomedia/SunInTime/{date_path}/t{code}.jpg",
            ]
            for url in urls:
                try:
                    response = self.session.head(url, timeout=15)
                    if response.status_code == 200:
                        return response.headers.get("Last-Modified") or candidate_date.strftime("%Y-%m-%d")
                except requests.exceptions.RequestException:
                    continue
        return None

    def _timestamp_from_jsoc(self, source: str) -> Optional[str]:
        timestamp_key = SDO_SOURCES[source].get("jsoc_timestamp_key")
        if not timestamp_key:
            return None

        url = "https://jsoc1.stanford.edu/data/hmi/images/latest/image_times_UTC"
        response = self.session.get(url, timeout=15)
        response.raise_for_status()

        for line in response.text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            if key.strip().lower() == timestamp_key:
                return value.strip()
        return None

    def _timestamp_from_nasa(self, source: str) -> Optional[str]:
        nasa_code = SDO_SOURCES[source].get("nasa_code")
        if not nasa_code:
            return None

        urls = [
            f"http://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_{nasa_code}.jpg",
            f"https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_{nasa_code}.jpg",
        ]

        for url in urls:
            try:
                response = self.session.head(url, timeout=15)
                response.raise_for_status()
                return response.headers.get("Last-Modified")
            except requests.exceptions.RequestException:
                continue

        return None

    def _timestamp_from_helioviewer(self, source: str) -> Optional[str]:
        source_id = SDO_SOURCES[source]["sourceId"]
        response = self.session.get(
            "https://api.helioviewer.org/v2/getClosestImage/",
            params={
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sourceId": source_id,
            },
            timeout=15,
        )
        response.raise_for_status()
        return response.json().get("date")

    def _save_response(
        self,
        response: requests.Response,
        source: str,
        provider: str,
        image_url: str,
        extension: str,
        observation_time: Optional[str] = None,
        extra_metadata: Optional[Dict] = None,
    ) -> Dict:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"SDO_{source}_{timestamp}{extension}"

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        metadata = {
            "source": source,
            "name": SDO_SOURCES[source]["name"],
            "wavelength": SDO_SOURCES[source]["wavelength"],
            "description": SDO_SOURCES[source]["description"],
            "provider": provider,
            "provider_name": PROVIDER_LABELS[provider],
            "filepath": str(filepath),
            "download_time": datetime.now(timezone.utc).isoformat(),
            "image_url": image_url,
            "observation_time": observation_time,
            "content_type": response.headers.get("Content-Type"),
            "last_modified": response.headers.get("Last-Modified"),
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        metadata_file = filepath.with_suffix(".json")
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Image saved: {filepath}")
        print(f"✓ Metadata saved: {metadata_file}")
        print(f"✓ Provider used: {PROVIDER_LABELS[provider]}")

        return metadata