import abc
import os
import shutil
import json
from typing import Optional, Text, Tuple, Dict, Any

import requests
from requests.auth import HTTPBasicAuth

from simatcher.log import logger
from simatcher.exceptions import ActionFailed


def get_persistor(name: Text) -> Optional["Persistor"]:
    """Returns an instance of the requested persistor.

    Currently, `bkrepo`, `cos` and providing module paths are supported remote
    storages.
    """
    if name == "bkrepo":
        return BKRepoPersistor(
            os.environ.get("BUCKET_NAME"), os.environ.get("BK_REPO_ENDPOINT_URL")
        )
    if name == "tencentcloud":
        return TencentCloudPersistor(os.environ.get("BUCKET_NAME"))
    return None


class Persistor(abc.ABC):
    """Store models in cloud and fetch them when needed."""
    @abc.abstractmethod
    def persist(self, model_directory: Text, model_name: Text, project_name: Text) -> None:
        """Uploads a model persisted in the `target_dir` to cloud storage."""
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve(self, model_name: Text, target_path: Text) -> None:
        """Downloads a model that has been persisted to cloud storage."""
        raise NotImplementedError

    def _compress(self, model_directory: Text, model_name: Text) -> Tuple[Text, Text]:
        """Creates a compressed archive and returns key and tar."""
        import tempfile

        dirpath = tempfile.mkdtemp()
        base_name = self._tar_name(model_name, include_extension=False)
        tar_name = shutil.make_archive(
            os.path.join(dirpath, base_name),
            "gztar",
            root_dir=model_directory,
            base_dir=".",
        )
        file_key = os.path.basename(tar_name)
        return file_key, tar_name

    @staticmethod
    def _tar_name(model_name: Text, include_extension: bool = True) -> Text:

        ext = ".tar.gz" if include_extension else ""
        return f"{model_name}{ext}"

    @staticmethod
    def _copy(compressed_path: Text, target_path: Text) -> None:
        shutil.copy2(compressed_path, target_path)

    def _handle_api_result(self, result: Optional[Dict[str, Any]]) -> Any:
        if isinstance(result, dict):
            if result.get('result', False) or result.get('code', 0) == 0:
                return result.get('data')
            logger.error(result)
            raise ActionFailed

    def call_action(self, action: str, method: str, **params) -> Any:
        params.update({'auth': self.basic})
        url = f"{self.api_root}/{action}"
        response = getattr(requests, method)(url, **params)
        try:
            return self._handle_api_result(response.json())
        except (TypeError, json.JSONDecodeError):
            return response


class BKRepoPersistor(Persistor):
    """BlueKing native storage BKrepo"""

    def __init__(
        self,
        bucket_name: Text = None,
    ) -> None:
        super().__init__()
        self.bucket_name = bucket_name or os.getenv('BK_REPO_BUCKET')
        self.api_root = os.getenv('BK_REPO_ROOT')
        self.basic = HTTPBasicAuth(os.getenv('BK_REPO_USERNAME'), os.getenv('BK_REPO_PASSWORD'))

    def persist(self, model_directory: Text, model_name: Text, project_name: Text) -> None:
        if not os.path.isdir(model_directory):
            raise ValueError(f"Target directory '{model_directory}' not found.")

        file_key, tar_path = self._compress(model_directory, model_name)
        with open(tar_path, "rb") as f:
            self._upload('simatcher', f'archive/{project_name}/{file_key}', data=f)

    def retrieve(self, model_name: Text, target_path: Text) -> None:
        pass

    def _upload(self, project: str, abs_path: str, **params):
        """
        files/data
        """
        return self.call_action(f'generic/{self.bucket_name}/{project}/{abs_path}', 'put', **params)

    def _download(self, project: str, abs_path: str, **params):
        return self.call_action(f'generic/{self.bucket_name}/{project}/{abs_path}?download=true',
                                'get', **params)

    def _search(self, rule: Dict):
        return self.call_action('repository/api/node/search',
                                'post',
                                json={
                                    "page": {"pageNumber": 1, "pageSize": 1000},
                                    "sort": {"properties": ["folder", "lastModifiedDate"],
                                             "direction": "DESC"},
                                    "rule": rule
                                })


class TencentCloudPersistor(Persistor):
    """Tencent cloud cos"""

    def __init__(
        self,
        bucket_name: Text,
        endpoint_url: Optional[Text] = None,
        region_name: Optional[Text] = None,
    ) -> None:
        super().__init__()
        self.bucket_name = bucket_name

    def persist(self, model_directory: Text, model_name: Text, project_name: Text) -> None:
        pass

    def retrieve(self, model_name: Text, target_path: Text) -> None:
        pass
