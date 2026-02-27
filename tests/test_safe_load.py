import unittest
from unittest import mock

from alphalens_forecast.models.safe_load import safe_torch_load


class TestSafeTorchLoad(unittest.TestCase):
    @mock.patch("alphalens_forecast.models.safe_load.torch.load")
    def test_cuda_deserialize_retries_on_cpu(self, mock_load):
        error = RuntimeError(
            "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"
        )
        mock_load.side_effect = [error, {"ok": True}]

        result = safe_torch_load("dummy.pt", prefer_device="cuda")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_load.call_count, 2)
        first_kwargs = mock_load.call_args_list[0].kwargs
        second_kwargs = mock_load.call_args_list[1].kwargs
        self.assertTrue("map_location" not in first_kwargs or first_kwargs["map_location"] is None)
        self.assertEqual(str(second_kwargs.get("map_location")), "cpu")

    @mock.patch("alphalens_forecast.models.safe_load.torch.cuda.is_available", return_value=False)
    @mock.patch("alphalens_forecast.models.safe_load.torch.load")
    def test_auto_cpu_uses_map_location(self, mock_load, _mock_cuda):
        mock_load.return_value = {"ok": True}

        result = safe_torch_load("dummy.pt", prefer_device="auto")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_load.call_count, 1)
        kwargs = mock_load.call_args.kwargs
        self.assertEqual(str(kwargs.get("map_location")), "cpu")


if __name__ == "__main__":
    unittest.main()
