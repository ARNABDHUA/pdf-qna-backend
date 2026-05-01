from pywebpush import Vapid
import base64

v = Vapid()
v.generate_keys()

# Export public key as Base64url string
public_key_bytes = v.public_key.public_bytes(
    encoding=__import__('cryptography.hazmat.primitives.serialization', fromlist=['Encoding']).Encoding.X962,
    format=__import__('cryptography.hazmat.primitives.serialization', fromlist=['PublicFormat']).PublicFormat.UncompressedPoint
)
public_key_b64 = base64.urlsafe_b64encode(public_key_bytes).rstrip(b'=').decode('utf-8')

# Export private key as Base64url string
private_key_bytes = v.private_key.private_bytes(
    encoding=__import__('cryptography.hazmat.primitives.serialization', fromlist=['Encoding']).Encoding.PEM,
    format=__import__('cryptography.hazmat.primitives.serialization', fromlist=['PrivateFormat']).PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=__import__('cryptography.hazmat.primitives.serialization', fromlist=['NoEncryption']).NoEncryption()
)
private_key_b64 = base64.urlsafe_b64encode(private_key_bytes).rstrip(b'=').decode('utf-8')

print("=" * 60)
print("VAPID_PUBLIC_KEY =", public_key_b64)
print("VAPID_PRIVATE_KEY =", private_key_b64)
print("=" * 60)