import hashlib
import json

class MerkleSealer:
    """
    Handles the cryptographic sealing of the evidence record.
    Generates the Image Hash and the Record Hash (Merkle Link).
    """
    
    @staticmethod
    def hash_image(image_bytes: bytes) -> str:
        """
        Creates a SHA-256 hash of the raw image bytes.
        This proves the image has not been altered after capture.
        """
        sha256 = hashlib.sha256()
        sha256.update(image_bytes)
        return sha256.hexdigest()
        
    @staticmethod
    def hash_record(record_data: dict, previous_hash: str) -> str:
        """
        Creates the Merkle Chain Record Hash.
        The record_data must include the image_hash.
        It appends the previous_hash to strictly link this record 
        to the previous one in the chain, preventing deletion tampering.
        """
        # We need a deterministic string representation of the data.
        # We extract specific fields in a strict order to ensure consistency.
        
        # Expected fields: officer_badge_id, timestamp, latitude, longitude, 
        # result, substance, confidence, image_hash
        
        ordered_string = (
            f"{record_data.get('officer_badge_id', '')}|"
            f"{record_data.get('timestamp', '')}|"
            f"{record_data.get('latitude', 0.0)}|"
            f"{record_data.get('longitude', 0.0)}|"
            f"{record_data.get('result', '')}|"
            f"{record_data.get('substance', '')}|"
            f"{record_data.get('confidence', 0.0)}|"
            f"{record_data.get('image_hash', '')}|"
            f"{previous_hash}"
        )
        
        sha256 = hashlib.sha256()
        sha256.update(ordered_string.encode('utf-8'))
        return sha256.hexdigest()

# Example Usage for Flutter Developer:
# previous_hash = "0000000000000000000000000000000000000000000000000000000000000000" # Genesis
# image_hash = MerkleSealer.hash_image(raw_bytes)
# data = {
#     "officer_badge_id": "NCB-4421",
#     "timestamp": "2026-09-19T10:00:00Z",
#     "latitude": 19.0760,
#     "longitude": 72.8777,
#     "result": "POSITIVE",
#     "substance": "Cocaine",
#     "confidence": 94.2,
#     "image_hash": image_hash
# }
# record_hash = MerkleSealer.hash_record(data, previous_hash)
