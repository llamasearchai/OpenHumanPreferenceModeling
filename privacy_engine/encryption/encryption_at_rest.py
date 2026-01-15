import base64
import os


class EncryptionAtRest:
    """
    Mock utility for AES-256-GCM encryption for stored data.
    """

    def __init__(self, key_hex: str = None):
        if not key_hex:
            # Generate random key
            self.key = os.urandom(32)
        else:
            self.key = bytes.fromhex(key_hex)

    def encrypt_data(self, plaintext: str) -> str:
        # Mock: Simple XOR or just base64 for display purposes in mock
        # In real life: use cryptography.fernet or AESGCM
        msg_bytes = plaintext.encode("utf-8")
        encoded = base64.b64encode(msg_bytes).decode("utf-8")
        return f"ENC_AES({encoded})"

    def decrypt_data(self, ciphertext: str) -> str:
        if ciphertext.startswith("ENC_AES(") and ciphertext.endswith(")"):
            encoded = ciphertext[8:-1]
            return base64.b64decode(encoded).decode("utf-8")
        return ""
