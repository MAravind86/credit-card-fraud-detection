import threading
from datetime import datetime
from typing import Any, Dict

from fraud_detector_interface import SimpleFraudDetector


class ModelService:
	"""Thread-safe singleton-style model service for predictions."""
	_instance = None
	_lock = threading.Lock()

	def __init__(self):
		try:
			self.detector = SimpleFraudDetector()
			# Load/prepare/train once on first use
			self.detector.load_and_prepare_data()
			self.detector.balance_data()
			self.detector.train_model()
			self.detector.save_model()
			self.detector.load_model()
		except Exception as e:
			print(f"Model initialization error: {e}")
			raise

	@classmethod
	def get_instance(cls) -> "ModelService":
		if cls._instance is None:
			with cls._lock:
				if cls._instance is None:
					cls._instance = ModelService()
		return cls._instance

	def parse_time(self, time_input: str) -> float:
		"""Support 'now', HH:MM, or epoch seconds as string."""
		if isinstance(time_input, (int, float)):
			return float(time_input)
		text = str(time_input).strip().lower()
		if text == 'now':
			return datetime.now().timestamp()
		if ':' in text:
			hour, minute = map(int, text.split(':'))
			ts = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0).timestamp()
			return ts
		return float(text)

	def predict(self, transaction_id: str, time_value: Any, amount_value: Any) -> Dict[str, Any]:
		try:
			ts = self.parse_time(time_value)
		except Exception:
			return {
				'error': "Invalid time. Use 'now', 'HH:MM', or epoch seconds.",
				'transaction_id': transaction_id
			}
		try:
			amount = float(amount_value)
		except Exception:
			return {
				'error': 'Invalid amount. Enter numeric value.',
				'transaction_id': transaction_id
			}
		return self.detector.predict_fraud(transaction_id, ts, amount)



