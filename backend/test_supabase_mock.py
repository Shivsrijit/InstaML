import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import sys

# Add backend dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.app.core.supabase_storage import (
    is_supabase_configured,
    get_relative_storage_path,
    upload_file_to_supabase,
    download_file_from_supabase,
    ensure_local_file_exists
)

class TestSupabaseStorage(unittest.TestCase):

    def setUp(self):
        import backend.app.core.supabase_storage as s_storage
        self.orig_s_url = s_storage.SUPABASE_URL
        self.orig_s_key = s_storage.SUPABASE_KEY
        
        # Cache current env vars
        self.orig_url = os.environ.get("SUPABASE_URL")
        self.orig_key = os.environ.get("SUPABASE_KEY")
        
        # Unset for isolation
        if "SUPABASE_URL" in os.environ:
            del os.environ["SUPABASE_URL"]
        if "SUPABASE_KEY" in os.environ:
            del os.environ["SUPABASE_KEY"]
            
        s_storage.SUPABASE_URL = None
        s_storage.SUPABASE_KEY = None

    def tearDown(self):
        import backend.app.core.supabase_storage as s_storage
        s_storage.SUPABASE_URL = self.orig_s_url
        s_storage.SUPABASE_KEY = self.orig_s_key
        
        # Restore env vars
        if self.orig_url is not None:
            os.environ["SUPABASE_URL"] = self.orig_url
        if "SUPABASE_URL" in os.environ and self.orig_url is None:
            del os.environ["SUPABASE_URL"]
            
        if self.orig_key is not None:
            os.environ["SUPABASE_KEY"] = self.orig_key
        if "SUPABASE_KEY" in os.environ and self.orig_key is None:
            del os.environ["SUPABASE_KEY"]


    def test_supabase_configured_false(self):
        self.assertFalse(is_supabase_configured())

    def test_supabase_configured_true(self):
        os.environ["SUPABASE_URL"] = "https://xyz.supabase.co"
        os.environ["SUPABASE_KEY"] = "super-secret-key"
        
        # Re-import or dynamically evaluate config
        import backend.app.core.supabase_storage as s_storage
        # Dynamically set vars since they are read at module import level
        s_storage.SUPABASE_URL = "https://xyz.supabase.co"
        s_storage.SUPABASE_KEY = "super-secret-key"
        
        self.assertTrue(s_storage.is_supabase_configured())

    def test_relative_path_parsing(self):
        # Path with user_X should map correctly
        p1 = Path("/workspace/backend/storage/user_42/project_7/data/file.csv")
        p2 = Path("C:\\Users\\SSN\\Desktop\\instaml\\backend\\storage\\user_1\\project_3\\models\\model_5.pkl")
        
        rel1 = get_relative_storage_path(p1)
        rel2 = get_relative_storage_path(p2)
        
        self.assertEqual(rel1, "user_42/project_7/data/file.csv")
        self.assertEqual(rel2, "user_1/project_3/models/model_5.pkl")

    @patch("requests.post")
    def test_upload_file_api_call(self, mock_post):
        # Setup mock env
        import backend.app.core.supabase_storage as s_storage
        s_storage.SUPABASE_URL = "https://xyz.supabase.co"
        s_storage.SUPABASE_KEY = "mykey"
        
        # Setup mock response
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_post.return_value = mock_res
        
        # Setup temp local file
        temp_file = Path(__file__).parent / "temp_test_upload.txt"
        with open(temp_file, "w") as f:
            f.write("test data")
            
        try:
            success = upload_file_to_supabase(temp_file)
            self.assertTrue(success)
            
            # Assert mock was called with correct url and authorization headers
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            url = args[0]
            self.assertTrue(url.startswith("https://xyz.supabase.co/storage/v1/object/"))
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer mykey")
            self.assertEqual(kwargs["headers"]["x-upsert"], "true")
        finally:
            if temp_file.exists():
                temp_file.unlink()

    @patch("requests.get")
    def test_download_file_api_call(self, mock_get):
        # Setup mock env
        import backend.app.core.supabase_storage as s_storage
        s_storage.SUPABASE_URL = "https://xyz.supabase.co"
        s_storage.SUPABASE_KEY = "mykey"
        
        # Setup mock response
        mock_res = MagicMock()
        mock_res.status_code = 200
        mock_res.content = b"downloaded text content"
        mock_get.return_value = mock_res
        
        # Setup temp download path
        temp_dest = Path(__file__).parent / "temp_download_dest.txt"
        if temp_dest.exists():
            temp_dest.unlink()
            
        try:
            success = download_file_from_supabase("user_1/project_1/data/test.txt", temp_dest)
            self.assertTrue(success)
            self.assertTrue(temp_dest.exists())
            with open(temp_dest, "r") as f:
                self.assertEqual(f.read(), "downloaded text content")
        finally:
            if temp_dest.exists():
                temp_dest.unlink()

    def test_ensure_local_file_exists_local_already(self):
        # File that exists locally should immediately return path without making API calls
        local_file = Path(__file__).resolve()
        
        import backend.app.core.supabase_storage as s_storage
        s_storage.SUPABASE_URL = "https://xyz.supabase.co"
        s_storage.SUPABASE_KEY = "mykey"
        
        with patch("backend.app.core.supabase_storage.download_file_from_supabase") as mock_dl:
            res_path = ensure_local_file_exists(str(local_file))
            self.assertEqual(res_path, str(local_file))
            mock_dl.assert_not_called()

if __name__ == "__main__":
    unittest.main()
