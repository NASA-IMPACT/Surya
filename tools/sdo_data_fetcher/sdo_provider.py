"""
Shared SDO data provider logic with automatic fallback.
"""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Callable, Dict, Iterable, List, Optional, Union

import requests


SDO_SOURCES = {
    "AIA_94": {
        "sourceId": 8,
        "name": "AIA 94",
        "wavelength": "94Å",
        "description": "AIA 94 Å - Hot flare plasma",
        "nasa_code": "0094",
        "lmsal_code": "0094",
    },
    "AIA_131": {
        "sourceId": 9,
        "name": "AIA 131",
        "wavelength": "131Å",
        "description": "AIA 131 Å - Flaring regions",
        "nasa_code": "0131",
        "lmsal_code": "0131",
    },
    "AIA_171": {
        "sourceId": 10,
        "name": "AIA 171",
        "wavelength": "171Å",
        "description": "AIA 171 Å - Quiet corona and coronal loops",
        "nasa_code": "0171",
        "lmsal_code": "0171",
    },
    "AIA_193": {
        "sourceId": 11,
        "name": "AIA 193",
        "wavelength": "193Å",
        "description": "AIA 193 Å - Hot plasma in active regions",
        "nasa_code": "0193",
        "lmsal_code": "0193",
    },
    "AIA_211": {
        "sourceId": 12,
        "name": "AIA 211",
        "wavelength": "211Å",
        "description": "AIA 211 Å - Active regions",
        "nasa_code": "0211",
        "lmsal_code": "0211",
    },
    "AIA_304": {
        "sourceId": 13,
        "name": "AIA 304",
        "wavelength": "304Å",
        "description": "AIA 304 Å - Chromosphere and prominence",
        "nasa_code": "0304",
        "lmsal_code": "0304",
    },
    "AIA_335": {
        "sourceId": 14,
        "name": "AIA 335",
        "wavelength": "335Å",
        "description": "AIA 335 Å - Active regions",
        "nasa_code": "0335",
        "lmsal_code": "0335",
    },
    "AIA_1600": {
        "sourceId": 15,
        "name": "AIA 1600",
        "wavelength": "1600Å",
        "description": "AIA 1600 Å - Upper photosphere",
        "nasa_code": "1600",
        "lmsal_code": "1600",
    },
    "AIA_1700": {
        "sourceId": 16,
        "name": "AIA 1700",
        "wavelength": "1700Å",
        "description": "AIA 1700 Å - Temperature minimum",
        "nasa_code": "1700",
        "lmsal_code": "1700",
    },
    "AIA_4500": {
        "sourceId": 17,
        "name": "AIA 4500",
        "wavelength": "4500Å",
        "description": "AIA 4500 Å - Visible light photosphere",
        "nasa_code": "4500",
        "lmsal_code": "4500",
    },
    "HMI_Continuum": {
        "sourceId": 18,
        "name": "HMI Continuum",
        "wavelength": "Continuum",
        "description": "HMI Continuum - Solar surface",
        "nasa_code": "HMIIC",
        "lmsal_code": "_HMI_cont_aiascale",
        "jsoc_path": "/data/hmi/images/latest/HMI_latest_Int_1024x1024.gif",
        "jsoc_timestamp_key": "continuum",
    },
    "HMI_Magnetogram": {
        "sourceId": 19,
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


def parse_target_datetime(value: Union[str, datetime], timezone_mode: str = "utc") -> datetime:
    """Parse a target time and return an aware UTC datetime."""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = value.strip()
        if not text:
            raise ValueError("Target datetime cannot be empty")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if "T" not in text and " " in text:
            text = text.replace(" ", "T", 1)
        parsed = datetime.fromisoformat(text)

    if parsed.tzinfo is None:
        if timezone_mode.lower() == "local":
            return parsed.astimezone(timezone.utc)
        else:
            parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def format_utc_datetime(value: datetime) -> str:
    """Format a datetime for Helioviewer APIs."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_slug(value: datetime) -> str:
    """Create a filesystem-safe UTC timestamp."""
    return value.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _parse_helioviewer_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip().replace(" ", "T", 1)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class SDOProviderClient:
    """Download latest SDO imagery from multiple redundant providers."""

    def __init__(self, output_dir: str = "sdo_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def download_image_at(
        self,
        source: str,
        target_time: Union[str, datetime],
        timezone_mode: str = "utc",
        width: int = 1024,
        image_type: str = "png",
        output_subdir: Optional[str] = None,
    ) -> Optional[Dict]:
        """Download the image closest to a requested UTC or local target time."""
        if source not in SDO_SOURCES:
            raise ValueError(f"Invalid source. Choose from: {list(SDO_SOURCES.keys())}")

        image_type = image_type.lower().lstrip(".")
        if image_type not in {"png", "jpg", "webp"}:
            raise ValueError("image_type must be one of: png, jpg, webp")
        if width <= 0:
            raise ValueError("width must be greater than zero")

        target_dt = parse_target_datetime(target_time, timezone_mode=timezone_mode)
        source_info = SDO_SOURCES[source]

        print(f"\nFetching {source} closest to {format_utc_datetime(target_dt)}...")
        print(f"Wavelength: {source_info['wavelength']}")
        print("Provider: Helioviewer API")

        info_response = self._request_with_retries(
            "https://api.helioviewer.org/v2/getClosestImage/",
            params={
                "date": format_utc_datetime(target_dt),
                "sourceId": source_info["sourceId"],
            },
            timeout=45,
        )
        info_response.raise_for_status()
        image_info = info_response.json()
        image_id = image_info.get("id")

        if not image_id:
            return None

        response = self._request_with_retries(
            "https://api.helioviewer.org/v2/downloadImage/",
            params={
                "id": image_id,
                "width": width,
                "type": image_type,
            },
            timeout=90,
            stream=True,
        )
        response.raise_for_status()

        target_dir = self.output_dir / output_subdir if output_subdir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        filepath = target_dir / f"SDO_{source}_{utc_slug(target_dt)}.{image_type}"

        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        actual_dt = _parse_helioviewer_datetime(image_info.get("date"))
        signed_delta = None
        abs_delta = None
        if actual_dt:
            signed_delta = (actual_dt - target_dt).total_seconds()
            abs_delta = abs(signed_delta)

        metadata = {
            "source": source,
            "name": source_info["name"],
            "wavelength": source_info["wavelength"],
            "description": source_info["description"],
            "provider": "helioviewer",
            "provider_name": PROVIDER_LABELS["helioviewer"],
            "filepath": str(filepath),
            "download_time": datetime.now(timezone.utc).isoformat(),
            "image_url": response.url,
            "content_type": response.headers.get("Content-Type"),
            "requested_time": format_utc_datetime(target_dt),
            "observation_time": format_utc_datetime(actual_dt) if actual_dt else image_info.get("date"),
            "actual_observation_time": format_utc_datetime(actual_dt) if actual_dt else image_info.get("date"),
            "delta_seconds": signed_delta,
            "abs_delta_seconds": abs_delta,
            "image_id": image_id,
            "image_width": width,
            "image_type": image_type,
            "helioviewer_metadata": image_info,
        }

        metadata_file = filepath.with_suffix(".json")
        metadata["metadata_filepath"] = str(metadata_file)
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Image saved: {filepath}")
        print(f"✓ Metadata saved: {metadata_file}")
        if metadata["actual_observation_time"]:
            print(f"✓ Observation time: {metadata['actual_observation_time']}")

        return metadata

    def download_samples(
        self,
        sources: Optional[List[str]],
        start_time: Union[str, datetime],
        timezone_mode: str = "utc",
        hours: float = 1.0,
        cadence_minutes: int = 15,
        width: int = 1024,
        image_type: str = "png",
        output_subdir: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        """Download a forward-only time series for the selected SDO sources."""
        if sources is None:
            sources = list(SDO_SOURCES.keys())
        invalid_sources = [source for source in sources if source not in SDO_SOURCES]
        if invalid_sources:
            raise ValueError(f"Invalid sources: {invalid_sources}")
        if hours <= 0:
            raise ValueError("hours must be greater than zero")
        if cadence_minutes <= 0:
            raise ValueError("cadence_minutes must be greater than zero")

        start_dt = parse_target_datetime(start_time, timezone_mode=timezone_mode)
        end_dt = start_dt + timedelta(hours=hours)
        sample_times = []
        current_dt = start_dt
        while current_dt <= end_dt:
            sample_times.append(current_dt)
            current_dt += timedelta(minutes=cadence_minutes)

        run_subdir = output_subdir or f"historical_{utc_slug(start_dt)}"
        run_dir = self.output_dir / run_subdir
        run_dir.mkdir(parents=True, exist_ok=True)

        total = len(sample_times) * len(sources)
        completed = 0
        results = []
        errors = []

        for sample_time in sample_times:
            sample_subdir = f"{run_subdir}/{utc_slug(sample_time)}"
            for source in sources:
                try:
                    result = self.download_image_at(
                        source=source,
                        target_time=sample_time,
                        timezone_mode="utc",
                        width=width,
                        image_type=image_type,
                        output_subdir=sample_subdir,
                    )
                    if result:
                        results.append(result)
                        event = {"type": "result", "completed": completed + 1, "total": total, "result": result}
                    else:
                        error = {
                            "source": source,
                            "requested_time": format_utc_datetime(sample_time),
                            "error": "No image found",
                        }
                        errors.append(error)
                        event = {"type": "error", "completed": completed + 1, "total": total, "error": error}
                except Exception as exc:
                    error = {
                        "source": source,
                        "requested_time": format_utc_datetime(sample_time),
                        "error": str(exc),
                    }
                    errors.append(error)
                    event = {"type": "error", "completed": completed + 1, "total": total, "error": error}

                completed += 1
                if progress_callback:
                    progress_callback(event)

        manifest = {
            "provider": "helioviewer",
            "provider_name": PROVIDER_LABELS["helioviewer"],
            "start_time": format_utc_datetime(start_dt),
            "end_time": format_utc_datetime(end_dt),
            "timezone_mode": timezone_mode,
            "hours": hours,
            "cadence_minutes": cadence_minutes,
            "sample_times": [format_utc_datetime(sample_time) for sample_time in sample_times],
            "sources": sources,
            "total_requested": total,
            "successful_downloads": len(results),
            "failed_downloads": len(errors),
            "output_dir": str(run_dir),
            "results": results,
            "errors": errors,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_file = run_dir / "manifest.json"
        manifest["manifest_filepath"] = str(manifest_file)
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✓ Historical fetch complete: {len(results)}/{total} images")
        print(f"✓ Manifest saved: {manifest_file}")
        return manifest

    def _request_with_retries(self, url: str, retries: int = 2, **kwargs) -> requests.Response:
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self.session.get(url, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt >= retries:
                    raise
                wait_seconds = 1 + attempt
                print(f"Helioviewer request failed, retrying in {wait_seconds}s: {exc}")
                time.sleep(wait_seconds)
        raise last_error

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
